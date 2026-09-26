import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from peft_pretraining.evaluation import evaluate_batches


class ControlledLM(nn.Module):
    """The first input token specifies the NLL of all later (class-one) tokens."""
    def __init__(self, consume_rng=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.child = nn.Dropout()
        self.register_buffer("marker", torch.tensor(0.0))
        self.calls = 0
        self.consume_rng = consume_rng

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(1)
        desired_loss = input_ids[:, 0].float()
        probability = torch.exp(-desired_loss)
        logits = torch.log((1.0 - probability) / 3.0)[:, None, None].expand(
            input_ids.shape[0], input_ids.shape[1], 4
        ).clone()
        logits[:, :, 1] = -desired_loss[:, None]
        return SimpleNamespace(logits=logits + 0.0 * self.weight)


def _batch(*rows):
    ids = torch.tensor(rows, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def test_corpus_loss_weights_prediction_targets_not_batches():
    model = ControlledLM()
    loss, count = evaluate_batches(model, [_batch([1, 1, 0, 0]), _batch([2, 1, 1, 1])], 0, "cpu")
    assert count == 4  # 6 nonpadding inputs, but only 4 scored targets.
    assert loss == pytest.approx((1.0 + 3.0 * 2.0) / 4.0, abs=1e-6)


def test_equal_batches_have_no_extra_divisor():
    loss, count = evaluate_batches(ControlledLM(), [_batch([1, 1]), _batch([3, 1])], 0, "cpu")
    assert (loss, count) == pytest.approx((2.0, 2))


def test_zero_target_batches_are_skipped_and_all_empty_is_explicit():
    model = ControlledLM()
    loss, count = evaluate_batches(model, [_batch([1, 0]), _batch([2, 1])], 0, "cpu")
    assert loss == pytest.approx(2.0, abs=1e-6)
    assert count == 1
    assert model.calls == 1
    with pytest.raises(ValueError, match="no valid next-token targets"):
        evaluate_batches(model, [_batch([1, 0]), _batch([1])], 0, "cpu")
    with pytest.raises(ValueError, match="no valid next-token targets"):
        evaluate_batches(model, [], 0, "cpu")


def test_budget_stops_after_first_round_reaching_target():
    model = ControlledLM()
    loss, count = evaluate_batches(model, [_batch([1, 1, 1])] * 5, 0, "cpu", target_tokens=3)
    assert count == 4
    assert model.calls == 2
    assert loss == pytest.approx(1.0, abs=1e-6)


def test_loss_is_invariant_to_batch_partition_across_upcast_chunks():
    rows = [[1, 0, 0, 0]] * 16 + [[2, 1, 1, 1], [3, 1, 0, 0]]
    together = evaluate_batches(ControlledLM(), [_batch(*rows)], 0, "cpu")
    separated = evaluate_batches(ControlledLM(), [_batch(row) for row in rows], 0, "cpu")
    assert together == pytest.approx(separated, abs=1e-6)
    assert together == pytest.approx((9.0 / 4.0, 4), abs=1e-6)


def test_attention_mask_excludes_nonpadding_masked_targets():
    batch = {"input_ids": torch.tensor([[2, 1, 1]]), "attention_mask": torch.tensor([[1, 1, 0]])}
    loss, count = evaluate_batches(ControlledLM(), [batch], 0, "cpu")
    assert count == 1
    assert loss == pytest.approx(2.0, abs=1e-6)


def test_modes_and_global_rng_are_restored_even_if_iteration_uses_rng():
    model = ControlledLM(consume_rng=True)
    model.train()
    model.child.eval()
    random.seed(51)
    np.random.seed(52)
    torch.manual_seed(53)
    py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()

    def batches():
        random.random()
        np.random.random()
        torch.rand(1)
        yield _batch([1, 1])

    evaluate_batches(model, batches(), 0, "cpu")
    assert model.training is True
    assert model.child.training is False
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    assert np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    model.eval()
    with pytest.raises(ValueError, match="no valid"):
        evaluate_batches(model, [], 0, "cpu")
    assert model.training is False


def _distributed_evaluation_worker(rank, init_file, output_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(ControlledLM())
        unequal = [_batch([1, 1])] if rank == 0 else [_batch([2, 1, 1]), _batch([3, 1, 1, 1])]
        result = {
            "unequal": evaluate_batches(model, unequal, 0, "cpu", target_tokens=100),
            "budget": evaluate_batches(model, unequal, 0, "cpu", target_tokens=3),
            "empty_rank": evaluate_batches(model, [] if rank == 0 else [_batch([2, 1, 1, 1])], 0, "cpu"),
        }
        try:
            evaluate_batches(model, [_batch([1, 0])] if rank == 0 else [], 0, "cpu")
        except ValueError as exc:
            result["all_empty"] = str(exc)
        result["training_restored"] = model.training and model.module.training
        with open(f"{output_dir}/{rank}.json", "w") as handle:
            json.dump(result, handle)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_distributed_uneven_shards_and_global_budget(tmp_path):
    mp.spawn(_distributed_evaluation_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    assert results[0] == results[1]
    assert results[0]["unequal"] == pytest.approx([14.0 / 6.0, 6], abs=1e-6)
    assert results[0]["budget"] == pytest.approx([5.0 / 3.0, 3], abs=1e-6)
    assert results[0]["empty_rank"] == pytest.approx([2.0, 3], abs=1e-6)
    assert "no valid next-token targets" in results[0]["all_empty"]
    assert results[0]["training_restored"] is True
