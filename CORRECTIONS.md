# Evaluation and checkpoint corrections in the minimal release

## Evaluation

- Replace the sum of `n` batch losses divided by `n+1` with total next-token
  negative log likelihood divided by valid prediction targets.
- Weight unequal batches and distributed shards by their actual target counts.
- Count shifted, unmasked targets for the global evaluation budget.
- Handle exhausted or empty ranks without mismatched DDP forwards.
- Preserve module modes and Python, NumPy and Torch RNG during evaluation.
- Use the same `allenai/c4` dataset identifier for training and validation.

## Release scope

Dedicated norm diagnostics, counterfactual/E3/E4 rollout modules, their tests,
command-line options and training hooks are removed. The package retains the
five-condition training experiment, evaluation and continuation. Norm figures
and tables are outside its reproduction scope.

## Checkpoint continuation

- Atomically save model, full optimizer/projector state, scheduler, counters,
  per-rank RNG, data-origin RNG, consumed minibatches and their checksum.
- Restore state at optimizer-update boundaries, replay the consumed data prefix,
  validate its checksum and restore training RNG after replay.
- Persist local LR origins in optimizer state; rebind
  parameter names/indices after loading or copying.
- Reject incomplete legacy checkpoints and incompatible numerical environments.
- Gather rank-local state and propagate rank-zero save errors to other ranks.
- Add an optional absolute stop update that preserves the full LR schedule,
  and pass stop/resume settings through the reproduction launcher.

## Validation and limits

Final minimal package: **45 tests and 2 subtests passing** on CPU
with Python 3.10.21, PyTorch 2.14.0, Transformers 4.31.0,
NumPy 2.2.6 and Datasets 5.0.1.

The automated suite covers the original intervention rules, unequal-token
evaluation, two-process Gloo/DDP evaluation with uneven shards,
optimizer-state round trips, and real disk checkpoint continuation
with dropout and stochastic DataLoader workers. An offline tiny-LLaMA trainer
test compares uninterrupted training with stop/resume for all five conditions,
including a two-worker run, checking weights, optimizer, projectors, scheduler,
RNG, data checksum and future losses exactly.

An additional independent arithmetic oracle compared 210 optimizer steps
across FP32/BF16, all five conditions, square/rectangular matrices and multiple
refreshes; the original update arithmetic was preserved.

No full C4 training, CUDA/NCCL checkpoint continuation, scientific-checkpoint
re-evaluation or regeneration of paper figures was performed. Corrected code
does not retroactively validate published perplexities or diagnostic values.
Keep data and environment fixed for continuation; replay checks consumed data
but cannot guarantee an unchanged future remote stream.

The archive is exported without Git history, AppleDouble files, bytecode,
test caches or personal archive owner metadata. Other issues identified by
the broader code/manuscript audit are outside this change.
