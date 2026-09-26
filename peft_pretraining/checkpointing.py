"""Complete checkpoints and deterministic replay for the released C4 trainer.

Only load checkpoints you trust: optimizer checkpoints contain Python projector
objects. Replay requires the same data/configuration and worker count and a data
pipeline independent of the training RNG, as in this release's C4 preprocessing.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state():
    """Snapshot all RNGs used by training, separately for each distributed rank."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def _validate_rng_state(state):
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Checkpoint lacks complete Python/NumPy/Torch RNG state; exact resume is unavailable")


def restore_rng_state(state, strict_cuda=True):
    """Restore saved RNGs; reject changed CUDA topology for exact continuation."""
    _validate_rng_state(state)
    cuda_states = state["torch_cuda"]
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if strict_cuda and len(cuda_states) != device_count:
        raise ValueError(
            "CUDA device count differs from checkpoint "
            f"({len(cuda_states)} saved, {device_count} current); exact resume is unavailable"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_states and device_count:
        if len(cuda_states) != device_count:
            raise ValueError("Cannot map saved CUDA RNG states to current devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def load_trusted_checkpoint(path, map_location="cpu"):
    """Load our tensor/custom-projector checkpoint across PyTorch default changes."""
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_model_weights(model, directory):
    """Save either a Hugging Face model or a plain torch module, without DDP prefix."""
    model = unwrap_model(model)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        # Explicit format prevents a library default from changing the loader contract.
        model.save_pretrained(str(directory), max_shard_size="100GB", safe_serialization=False)
    else:
        torch.save(model.state_dict(), directory / "pytorch_model.bin")


def load_model_weights(model, checkpoint_dir):
    """Load .bin or safetensors, including their standard Hugging Face shard indexes.

    This is also usable for legacy *weight-only* checkpoints. It does not assert
    that a legacy checkpoint can resume optimizer, data, or RNG state exactly.
    """
    root = Path(checkpoint_dir)
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = root / filename
        if path.is_file():
            shards = [path]
            break
    else:
        for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = root / filename
            if index_path.is_file():
                with index_path.open() as handle:
                    index = json.load(handle)
                shards = [root / name for name in sorted(set(index["weight_map"].values()))]
                break
        else:
            raise FileNotFoundError(f"No Hugging Face .bin or safetensors model weights in {root}")
    state = {}
    for path in shards:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            shard = load_file(str(path), device="cpu")
        else:
            shard = load_trusted_checkpoint(path)
        duplicates = state.keys() & shard.keys()
        if duplicates:
            raise ValueError(f"Duplicate model weight keys across shards: {sorted(duplicates)}")
        state.update(shard)
    unwrap_model(model).load_state_dict(state, strict=True)


def _move_state_value(value, device, memo):
    if id(value) in memo:
        return memo[id(value)]
    if torch.is_tensor(value):
        moved = value.to(device=device)
        memo[id(value)] = moved
        return moved
    if isinstance(value, dict):
        memo[id(value)] = value
        for key, item in value.items():
            value[key] = _move_state_value(item, device, memo)
    elif isinstance(value, list):
        memo[id(value)] = value
        value[:] = [_move_state_value(item, device, memo) for item in value]
    elif isinstance(value, tuple):
        moved = tuple(_move_state_value(item, device, memo) for item in value)
        memo[id(value)] = moved
        return moved
    elif hasattr(value, "__dict__"):
        # torch.optim.load_state_dict does not recurse into GaLoreProjector objects.
        memo[id(value)] = value
        for key, item in vars(value).items():
            setattr(value, key, _move_state_value(item, device, memo))
    return value


def optimizer_to_parameter_devices(optimizer):
    """Move moment and custom projector tensors to their corresponding parameter.

    Preserve dtype: checkpoint/projector precision is part of the algorithm.
    Call after optimizer.load_state_dict and model.to(device).
    """
    for parameter, state in optimizer.state.items():
        _move_state_value(state, parameter.device, {})


def update_data_digest(hasher, batch):
    """Hash a minibatch's sorted tensor keys, dtype, shape, and exact CPU bytes."""
    if not isinstance(batch, dict) or not all(torch.is_tensor(value) for value in batch.values()):
        raise TypeError("Data replay digest expects a mapping of tensor minibatch fields")
    for key in sorted(batch):
        value = batch[key].detach().cpu().contiguous()
        metadata = json.dumps([key, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        hasher.update(len(metadata).to_bytes(8, "big"))
        hasher.update(metadata)
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
    # Explicit boundary distinguishes consecutive batches with different fields.
    hasher.update(b"\x00MINIBATCH-END\x00")
    return hasher


def replay_data_iterator(dataloader, consumed_microbatches, *, initial_rng_state, resume_rng_state,
                         expected_data_digest=None, data_digest=None):
    """Recreate worker seeds and replay the deterministic stream to the next batch.

    Iteration and skipped batches must not advance the resumed training RNG.
    Save initial_rng_state immediately before the original iter(dataloader).
    Replaying scans and tokenizes all consumed batches; it trades startup time
    for an exact stream position without adding a stateful-loader dependency.
    When expected_data_digest is supplied, check the entire consumed prefix;
    data_digest is an initially empty hashlib.sha256 object the caller can keep
    extending after this function returns.
    A separate explicit DataLoader generator, if used, must also be rewound by
    the caller. Stochastic main-process transforms sharing the model RNG are
    outside this contract; use an independent dataset RNG for those transforms.
    """
    if not isinstance(consumed_microbatches, int) or consumed_microbatches < 0:
        raise ValueError("consumed_microbatches must be a non-negative integer")
    _validate_rng_state(initial_rng_state)
    _validate_rng_state(resume_rng_state)
    if data_digest is None and expected_data_digest is not None:
        data_digest = hashlib.sha256()
    try:
        restore_rng_state(initial_rng_state)
        iterator = iter(dataloader)
        for index in range(consumed_microbatches):
            try:
                batch = next(iterator)
                if data_digest is not None:
                    update_data_digest(data_digest, batch)
            except StopIteration as error:
                raise ValueError(
                    "Data stream ended while replaying saved position "
                    f"({index}/{consumed_microbatches} microbatches); check data/configuration"
                ) from error
        if expected_data_digest is not None and data_digest.hexdigest() != expected_data_digest:
            raise ValueError("Data replay prefix differs from checkpoint; exact resume is unavailable")
        return iterator
    finally:
        restore_rng_state(resume_rng_state)


def _validate_rank_states(rank_states):
    if not isinstance(rank_states, (list, tuple)) or not rank_states:
        raise ValueError("Exact checkpoint requires RNG/data state for every rank")
    for rank, state in enumerate(rank_states):
        if not isinstance(state, dict):
            raise ValueError(f"Missing checkpoint state for rank {rank}")
        for key in ("rng_state", "data_initial_rng_state"):
            if key not in state:
                raise ValueError(f"Rank {rank} lacks {key}; exact resume is unavailable")
            _validate_rng_state(state[key])
        consumed = state.get("consumed_microbatches")
        if not isinstance(consumed, int) or consumed < 0:
            raise ValueError(f"Rank {rank} lacks a valid consumed_microbatches position")


def save_training_checkpoint(directory, model, optimizer, scheduler, training_state,
                             resume_metadata, rank_states):
    """Publish one complete checkpoint atomically; caller gathers ranks first.

    Call only on rank zero at an optimizer-update boundary, after zero_grad.
    The target must not exist. Save into a sibling directory and rename only
    after every model/state file has finished, so partial checkpoints cannot be
    mistaken for complete ones. Checkpoint saving preserves the calling RNG.
    """
    _validate_rank_states(rank_states)
    if resume_metadata.get("world_size", len(rank_states)) != len(rank_states):
        raise ValueError("Saved rank state count does not match world_size")
    # Metadata should be stable, comparable configuration, not tensors or objects.
    json.dumps(resume_metadata, sort_keys=True)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent))
    entry_rng = capture_rng_state()
    try:
        save_model_weights(model, temporary)
        payload = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_state": dict(training_state),
            "resume_metadata": dict(resume_metadata),
            "rank_states": list(rank_states),
        }
        with (temporary / "optimizer.pt").open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        with (temporary / "training_state.json").open("w") as handle:
            json.dump(training_state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        restore_rng_state(entry_rng)


def load_training_checkpoint(directory, expected_resume_metadata=None):
    """Read a complete checkpoint; reject legacy partial resumes explicitly."""
    path = Path(directory) / "optimizer.pt"
    if not path.is_file():
        raise ValueError(
            "Checkpoint lacks optimizer/scheduler/data/RNG state; exact resume is unavailable. "
            "Legacy model weights can only be loaded as a fresh initialization."
        )
    payload = load_trusted_checkpoint(path)
    required = {"optimizer", "scheduler", "training_state", "resume_metadata", "rank_states"}
    if (payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION
            or not required.issubset(payload)):
        raise ValueError("Legacy/incomplete checkpoint has no verified data/RNG state; exact resume is unavailable")
    _validate_rank_states(payload["rank_states"])
    metadata = payload["resume_metadata"]
    if metadata.get("world_size", len(payload["rank_states"])) != len(payload["rank_states"]):
        raise ValueError("Checkpoint rank state count does not match world_size")
    if expected_resume_metadata is not None and metadata != expected_resume_metadata:
        keys = sorted(set(metadata) | set(expected_resume_metadata))
        differences = [key for key in keys if metadata.get(key) != expected_resume_metadata.get(key)]
        raise ValueError("Exact resume configuration mismatch: " + ", ".join(differences))
    return payload
