"""Offline CPU integration of the real trainer, scheduler, evaluation and resume.

Only remote data/tokenizer access and W&B are replaced. Model, optimizer,
DataLoader, checkpoint serialization and one-rank Gloo/DDP execute normally.
"""
import importlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from transformers import LlamaConfig

from peft_pretraining.checkpointing import load_trusted_checkpoint


class SyntheticTextDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def shuffle(self, seed):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return SyntheticTextDataset(rows)

    def __iter__(self):
        return iter(self.rows)

    def map(self, function, batched, remove_columns):
        assert batched
        mapped = function({"text": [row["text"] for row in self.rows]})
        return SyntheticTextDataset([
            {name: values[index] for name, values in mapped.items()}
            for index in range(len(self.rows))
        ])


class SyntheticTokenizer:
    pad_token_id = 0

    def __call__(self, text, max_length, **kwargs):
        texts = [text] if isinstance(text, str) else text
        rows = []
        for item in texts:
            index = int(item.split()[-1])
            row = [1, 2 + index % 9, 2 + (index * 3) % 9, 2 + (index * 7) % 9]
            if index % 2:
                row.append(2)
            row = row[:max_length]
            rows.append(row + [0] * (max_length - len(row)))
        ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class SilentProgress:
    def update(self, *args):
        pass

    def close(self):
        pass


def _assert_identical(left, right, path="root"):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype, path
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if key in {"param_name_by_id", "param_index_by_id"}:
                # In-memory parameter addresses differ across independent models.
                assert sorted(left[key].values()) == sorted(right[key].values()), f"{path}.{key}"
            else:
                _assert_identical(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_identical(a, b, f"{path}[{index}]")
    elif hasattr(left, "__dict__"):
        assert type(left) is type(right), path
        _assert_identical(vars(left), vars(right), path)
    else:
        assert left == right, path


@pytest.fixture
def offline_trainer(monkeypatch, tmp_path):
    trainer = importlib.import_module("torchrun_main")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def load_data(*args, split, **kwargs):
        count = 64 if split == "train" else 4
        return SyntheticTextDataset([
            {"text": f"document {i}", "timestamp": "", "url": ""} for i in range(count)
        ])

    monkeypatch.setattr(trainer.datasets, "load_dataset", load_data)
    monkeypatch.setattr(trainer.datasets.distributed, "split_dataset_by_node", lambda data, **kw: data)
    monkeypatch.setattr(trainer.AutoTokenizer, "from_pretrained", lambda *a, **kw: SyntheticTokenizer())
    monkeypatch.setattr(trainer, "wandb", SimpleNamespace(
        init=lambda **kw: None, config=SimpleNamespace(update=lambda *a, **kw: None),
        save=lambda *a, **kw: None, log=lambda *a, **kw: None,
    ))
    monkeypatch.setattr(trainer, "logger", SimpleNamespace(info=lambda *a, **kw: None,
                                                           warning=lambda *a, **kw: None,
                                                           remove=lambda *a, **kw: None))
    monkeypatch.setattr(trainer, "tqdm", lambda *a, **kw: SilentProgress())
    original_init = dist.init_process_group
    initialization_number = 0

    def initialize_local_group(*args, **kwargs):
        nonlocal initialization_number
        initialization_number += 1
        kwargs["init_method"] = f"file://{tmp_path / ('gloo_' + str(initialization_number))}"
        return original_init(*args, **kwargs)

    monkeypatch.setattr(trainer.dist, "init_process_group", initialize_local_group)
    configuration = tmp_path / "tiny_llama.json"
    configuration.write_text(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=16, pad_token_id=0, use_cache=False,
    ).to_json_string())
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield trainer, configuration
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.set_num_threads(old_threads)


def _run(trainer, configuration, output, condition, workers, *, stop=None, resume=None):
    output = Path(output)
    command = [
        "--model_config", str(configuration), "--device", "cpu", "--dtype", "float32",
        "--batch_size", "2", "--total_batch_size", "4", "--max_length", "6",
        "--optimizer", "galore_adamw", "--optimizer_state_ablation", condition,
        "--rank", "2", "--update_proj_gap", "2", "--galore_scale", "0.25",
        "--lr", "0.01", "--warmup_steps", "1", "--num_training_steps", "6",
        "--eval_every", "2", "--save_every", "3", "--workers", str(workers),
        "--seed", "17", "--save_dir", str(output),
        "--training_metrics_jsonl", str(output / "metrics.jsonl"),
    ]
    if stop is not None:
        command += ["--stop_after_updates", str(stop)]
    if resume is not None:
        command += ["--continue_from", str(resume)]
    args = trainer.parse_args(command)
    try:
        trainer.main(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return output


@pytest.mark.parametrize("condition,workers", [
    ("none", 0), ("reset_m", 0), ("reset_v", 0), ("reset_all", 0),
    ("reset_v_local_v_age", 0), ("reset_v_local_v_age", 2),
])
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_real_trainer_resume_matches_uninterrupted(offline_trainer, tmp_path, condition, workers):
    trainer, configuration = offline_trainer
    uninterrupted = _run(trainer, configuration, tmp_path / "full", condition, workers)
    interrupted = _run(trainer, configuration, tmp_path / "split", condition, workers, stop=3)
    _run(trainer, configuration, interrupted, condition, workers, resume=interrupted / "model_3")

    # Full serialization, not merely optimizer step counters or final model loss.
    for step in (3, 6):
        reference_dir, resumed_dir = uninterrupted / f"model_{step}", interrupted / f"model_{step}"
        _assert_identical(load_trusted_checkpoint(reference_dir / "pytorch_model.bin"),
                          load_trusted_checkpoint(resumed_dir / "pytorch_model.bin"), f"model_{step}")
        reference, actual = [load_trusted_checkpoint(directory / "optimizer.pt")
                             for directory in (reference_dir, resumed_dir)]
        for field in ("optimizer", "scheduler", "training_state", "rank_states", "resume_metadata"):
            _assert_identical(reference[field], actual[field], f"checkpoint_{step}.{field}")
        assert actual["rank_states"][0]["consumed_microbatches"] == step * 2
        assert len(actual["rank_states"][0]["data_digest"]) == 64

    def future_events(folder):
        return [row for line in (folder / "metrics.jsonl").read_text().splitlines()
                if (row := json.loads(line))["event"] in {"train", "eval"} and row["update_step"] > 3]

    assert future_events(uninterrupted) == future_events(interrupted)
