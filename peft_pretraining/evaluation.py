"""Corpus perplexity evaluation with an explicit next-token denominator."""

from contextlib import contextmanager
import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


@contextmanager
def _preserve_rng():
    """Evaluation must not advance the subsequent training RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _next_token_nll(logits, labels):
    """Sum float32 token losses without materializing a full float32 logits copy."""
    nll = torch.zeros((), dtype=torch.float64, device=logits.device)
    # A 256-sequence C4 batch has very large logits. Bound the temporary upcast
    # to 16 sequences while retaining the same token-wise cross entropy.
    for start in range(0, labels.shape[0], 16):
        shifted_logits = logits[start:start + 16, :-1, :].float()
        shifted_labels = labels[start:start + 16, 1:]
        losses = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        nll += losses.double().sum()
    return nll


@torch.no_grad()
def evaluate_batches(model, batches, pad_idx, device, target_tokens=10_000_000):
    """Return (global mean next-token NLL, global number of prediction targets).

    ``batches`` is an iterable of model input dictionaries, sharded per rank if
    distributed. Labels default to input_ids; padding and masked positions are
    excluded. The first token of each sequence is context, not a scored target.

    All ranks participate in one reduction per iteration, including ranks whose
    shards are exhausted. Stop jointly once at least ``target_tokens`` targets
    have been scored, or all shards end. The budget can be exceeded by at most
    the last global round of batches. DDP is unwrapped during evaluation so an
    exhausted rank need not perform a dummy forward or synchronize buffers.

    An evaluation containing no valid targets raises ValueError. The model's
    original module modes and Python/NumPy/Torch RNG states are always restored.
    """
    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool) or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    device = torch.device(device)
    distributed = dist.is_available() and dist.is_initialized()
    reduction_device = device if distributed and dist.get_backend() == "nccl" else torch.device("cpu")
    eval_model = model.module if isinstance(model, DistributedDataParallel) else model
    previous_modes = [(module, module.training) for module in model.modules()]
    total_nll = 0.0
    total_targets = 0

    with _preserve_rng():
        try:
            model.eval()
            iterator = iter(batches)
            exhausted = False
            while total_targets < target_tokens:
                batch = None
                if not exhausted:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        exhausted = True
                local_nll = 0.0
                local_targets = 0
                if batch is not None:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    labels = batch.pop("labels", batch["input_ids"]).clone()
                    if pad_idx is not None:
                        labels.masked_fill_(batch["input_ids"] == pad_idx, -100)
                    if "attention_mask" in batch:
                        labels.masked_fill_(batch["attention_mask"] == 0, -100)
                    local_targets = int((labels[..., 1:] != -100).sum().item())
                    if local_targets:
                        outputs = eval_model(**batch)
                        local_nll = _next_token_nll(outputs.logits, labels).item()
                # Float64 exactly represents the integer counts at these budgets.
                # One collective also communicates exhaustion, so uneven shards
                # terminate together without mismatched forward collectives.
                totals = torch.tensor(
                    [local_nll, local_targets, int(not exhausted)],
                    dtype=torch.float64,
                    device=reduction_device,
                )
                if distributed:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                round_nll, round_targets, active_ranks = totals.tolist()
                if not math.isfinite(round_nll):
                    raise ValueError("Evaluation produced a non-finite token NLL")
                total_nll += round_nll
                total_targets += int(round_targets)
                if not active_ranks:
                    break
            if total_targets == 0:
                raise ValueError("Evaluation contained no valid next-token targets")
            return total_nll / total_targets, total_targets
        finally:
            # Restore mixed parent/child modes, not only the top-level flag.
            for module, was_training in previous_modes:
                module.training = was_training
