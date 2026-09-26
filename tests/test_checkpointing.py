"""Disk checkpoint tests: complete optimizer/data/RNG continuation, not weights alone."""
import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from galore_torch.adamw import AdamW
from peft_pretraining.checkpointing import (
    capture_rng_state,
    load_model_weights,
    load_training_checkpoint,
    load_trusted_checkpoint,
    optimizer_to_parameter_devices,
    replay_data_iterator,
    restore_rng_state,
    save_training_checkpoint,
    update_data_digest,
)


class DeterministicBatches(Dataset):
    def __len__(self):
        return 30

    def __getitem__(self, index):
        x = (torch.arange(4, dtype=torch.float32) + index) / 15
        return {"x": x, "target": x.flip(0) * .7}


class WorkerRandomBatches(DeterministicBatches):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        if torch.utils.data.get_worker_info() is not None:
            # Worker-local RNGs are replayed from their original worker seeds.
            noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
            batch["x"] = batch["x"] + noise * .005
        return batch


class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.35)
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.linear(self.dropout(x))


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _objects():
    model = DropoutModel()
    parameter = model.linear.weight
    optimizer = AdamW([{
        "params": [parameter],
        "rank": 2,
        "update_proj_gap": 2,
        "scale": .25,
        "proj_type": "right",
        "optimizer_state_ablation": "reset_v_local_v_age",
        "param_name_by_id": {id(parameter): "linear.weight"},
        "param_index_by_id": {id(parameter): 0},
    }], lr=.03, betas=(.9, .99), eps=1e-6, no_deprecation_warning=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.9)
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler, batch):
    # Deliberately exercise all three CPU RNG streams in the executed gradient.
    perturbation = 1 + .01 * random.random() + .01 * float(np.random.rand())
    loss = ((model(batch["x"]) - batch["target"]) ** 2).mean() * perturbation
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return float(loss.detach())


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    elif hasattr(left, "__dict__"):
        assert type(left) is type(right)
        _assert_nested_equal(vars(left), vars(right))
    else:
        assert left == right


@pytest.mark.parametrize("num_workers", [0, 2])
def test_disk_resume_matches_continuous_dropout_optimizer_projector_scheduler(tmp_path, num_workers):
    _seed(73)
    model, optimizer, scheduler = _objects()
    dataloader = DataLoader(WorkerRandomBatches(), batch_size=2, num_workers=num_workers)
    initial_data_rng = capture_rng_state()
    iterator = iter(dataloader)
    data_digest = hashlib.sha256()
    for _ in range(3):
        batch = next(iterator)
        update_data_digest(data_digest, batch)
        _step(model, optimizer, scheduler, batch)
    resume_rng = capture_rng_state()
    metadata = {"world_size": 1, "workers": num_workers, "batch_size": 2, "data": "fixed-test-v1"}
    checkpoint = tmp_path / "checkpoint-3"
    save_training_checkpoint(
        checkpoint, model, optimizer, scheduler,
        {"global_step": 3, "update_step": 3}, metadata,
        [{"rng_state": resume_rng, "data_initial_rng_state": initial_data_rng,
          "consumed_microbatches": 3, "data_digest": data_digest.hexdigest()}],
    )
    _assert_nested_equal(capture_rng_state(), resume_rng)
    expected_losses = [_step(model, optimizer, scheduler, next(iterator)) for _ in range(4)]
    expected_weights = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(next(iter(optimizer.state.values())))
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_rng = capture_rng_state()
    assert expected_optimizer["step"] == 7
    assert expected_optimizer["v_age"] == 1
    assert expected_optimizer["projector"].refresh_count >= 3

    # Different initialization/RNG history before loading must have no effect.
    _seed(999)
    resumed_model, resumed_optimizer, resumed_scheduler = _objects()
    payload = load_training_checkpoint(checkpoint, expected_resume_metadata=metadata)
    load_model_weights(resumed_model, checkpoint)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    optimizer_to_parameter_devices(resumed_optimizer)
    resumed_scheduler.load_state_dict(payload["scheduler"])
    rank = payload["rank_states"][0]
    resumed_digest = hashlib.sha256()
    resumed_iterator = replay_data_iterator(
        DataLoader(WorkerRandomBatches(), batch_size=2, num_workers=num_workers),
        rank["consumed_microbatches"],
        initial_rng_state=rank["data_initial_rng_state"], resume_rng_state=rank["rng_state"],
        expected_data_digest=rank["data_digest"], data_digest=resumed_digest,
    )
    assert resumed_digest.hexdigest() == rank["data_digest"]
    actual_losses = [
        _step(resumed_model, resumed_optimizer, resumed_scheduler, next(resumed_iterator))
        for _ in range(4)
    ]
    assert actual_losses == expected_losses
    _assert_nested_equal(resumed_model.state_dict(), expected_weights)
    _assert_nested_equal(next(iter(resumed_optimizer.state.values())), expected_optimizer)
    _assert_nested_equal(resumed_scheduler.state_dict(), expected_scheduler)
    _assert_nested_equal(capture_rng_state(), expected_rng)


def test_model_loader_supports_safetensors_and_legacy_bin(tmp_path):
    from safetensors.torch import save_file
    source = nn.Linear(4, 3)
    restored = nn.Linear(4, 3)
    safe = tmp_path / "safe"
    safe.mkdir()
    save_file(source.state_dict(), str(safe / "model.safetensors"))
    load_model_weights(restored, safe)
    _assert_nested_equal(source.state_dict(), restored.state_dict())
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    torch.save(source.state_dict(), legacy / "pytorch_model.bin")
    load_model_weights(restored, legacy)
    _assert_nested_equal(source.state_dict(), restored.state_dict())
    with pytest.raises(ValueError, match="exact resume is unavailable"):
        load_training_checkpoint(legacy)


def test_checkpoint_rejects_incomplete_rng_changed_metadata_and_overwrite(tmp_path):
    model, optimizer, scheduler = _objects()
    rng = capture_rng_state()
    rank = {"rng_state": rng, "data_initial_rng_state": rng, "consumed_microbatches": 0}
    checkpoint = tmp_path / "complete"
    metadata = {"world_size": 1, "workers": 0}
    with pytest.raises(ValueError, match="data_initial_rng_state"):
        save_training_checkpoint(checkpoint, model, optimizer, scheduler, {}, metadata,
                                 [{"rng_state": rng, "consumed_microbatches": 0}])
    assert not checkpoint.exists()
    save_training_checkpoint(checkpoint, model, optimizer, scheduler, {}, metadata, [rank])
    with pytest.raises(ValueError, match="workers"):
        load_training_checkpoint(checkpoint, {"world_size": 1, "workers": 2})
    with pytest.raises(FileExistsError):
        save_training_checkpoint(checkpoint, model, optimizer, scheduler, {}, metadata, [rank])
    assert load_trusted_checkpoint(checkpoint / "optimizer.pt")["checkpoint_format_version"] == 1


def test_replay_detects_different_prefix_and_restores_rng_on_failure():
    initial = capture_rng_state()
    _seed(942)
    resumed = capture_rng_state()
    with pytest.raises(ValueError, match="prefix differs"):
        replay_data_iterator(DataLoader(DeterministicBatches(), batch_size=2), 3,
                             initial_rng_state=initial, resume_rng_state=resumed,
                             expected_data_digest="not-the-saved-data")
    _assert_nested_equal(capture_rng_state(), resumed)
    with pytest.raises(ValueError, match="Data stream ended"):
        replay_data_iterator([], 1, initial_rng_state=initial, resume_rng_state=resumed)
    _assert_nested_equal(capture_rng_state(), resumed)


def test_data_digest_distinguishes_values_dtype_shape_and_fields():
    def digest(batch):
        return update_data_digest(hashlib.sha256(), batch).hexdigest()
    baseline = {"a": torch.tensor([1, 2], dtype=torch.int64), "b": torch.tensor(3.)}
    assert digest(baseline) == digest(dict(reversed(list(baseline.items()))))
    alternatives = [
        {**baseline, "a": torch.tensor([1, 3], dtype=torch.int64)},
        {**baseline, "a": baseline["a"].int()},
        {**baseline, "a": baseline["a"].reshape(1, 2)},
        {"renamed": baseline["a"], "b": baseline["b"]},
    ]
    assert all(digest(batch) != digest(baseline) for batch in alternatives)


def test_restore_rng_rejects_cuda_topology_change_before_cpu_mutation():
    rng = capture_rng_state()
    changed = dict(rng)
    changed["torch_cuda"] = list(rng["torch_cuda"]) + [torch.zeros(4, dtype=torch.uint8)]
    with pytest.raises(ValueError, match="CUDA device count differs"):
        restore_rng_state(changed)
    _assert_nested_equal(capture_rng_state(), rng)


@pytest.mark.parametrize("safe", [False, True])
def test_model_loader_supports_sharded_indexes(tmp_path, safe):
    from safetensors.torch import save_file
    source = nn.Linear(4, 3)
    target = nn.Linear(4, 3)
    mapping = {}
    for index, (key, tensor) in enumerate(source.state_dict().items()):
        filename = f"model-{index}.safetensors" if safe else f"model-{index}.bin"
        if safe:
            save_file({key: tensor}, str(tmp_path / filename))
        else:
            torch.save({key: tensor}, tmp_path / filename)
        mapping[key] = filename
    index_name = "model.safetensors.index.json" if safe else "pytorch_model.bin.index.json"
    (tmp_path / index_name).write_text(json.dumps({"weight_map": mapping}))
    load_model_weights(target, tmp_path)
    _assert_nested_equal(source.state_dict(), target.state_dict())
