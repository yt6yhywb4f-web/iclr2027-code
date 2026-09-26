# GaLore state reconstruction

Code for `carry`, `reset_m`, `reset_v`, `reset_all`, and `local_v_age`.
Other retained options are outside this protocol; norm diagnostics are excluded.
Based on GaLore, under the Apache-2.0 license.

## Installation

Requires a CUDA GPU supporting BF16 and access to C4/en and `t5-base`.
Run from the repository root:

```bash
pip install -r exp_requirements.txt
pip install -e .
export WANDB_MODE=disabled
```

Omit the last line to use your configured Weights & Biases account.

## Training

Use the default auxiliary settings, without refresh/projector environment
overrides. Start fresh with `CONTINUE_FROM` and `STOP_AFTER_UPDATES` unset.

```bash
bash scripts/run_llama60_reconstruction.sh CONDITION SEED
```

Replace the placeholders according to the manuscript, which specifies the
experimental settings.
Outputs are saved under `outputs/llama60_reconstruction/`.

## Resume

```bash
CONTINUE_FROM=/path/to/checkpoint \
  bash scripts/run_llama60_reconstruction.sh CONDITION SEED
```

Keep the original run settings and environment. Only complete
checkpoints produced by this version support continuation.

## Tests

```bash
pip install -r requirements-dev.txt
OMP_NUM_THREADS=1 pytest -q
```

Validation covers CPU tests, not full C4/CUDA runs or published endpoints.
See [corrections and validation](CORRECTIONS.md) and [provenance](PROVENANCE.md).
