# Source provenance

This release combines two validated source lines without including scientific
outputs, cluster-specific paths, credentials, or scheduler metadata.

## Local-v-age source

- Source commit and tree identifiers are withheld during anonymous review.
- The scientific source and local archival source resolve to the same Git tree.
- The exported optimizer implementation had SHA-256
  `787816c8432c27c3cc2754c917404aa3b68665174f2c9079f6fe2df1dd93209e`
  before the release-only addition of `reset_m` and `reset_all`.

## Historical refresh interventions

The historical campaign implemented the following mutations at a non-initial
SVD refresh, before incorporating the current gradient:

- `reset_m`: zero `exp_avg`.
- `reset_v`: zero `exp_avg_sq`.
- `reset_all`: zero both tensors.

All three preserve the global optimizer and bias-correction counter. The
release adds these two missing choices (`reset_m`, `reset_all`) to the
validated local-v-age source at the same intervention point. No scheduler,
projector, parameter, RNG, or data-order semantics are changed.

## Scientific configuration

The checked-in `configs/llama_60m_reconstruction.json` is transcribed from
the effective run metadata. The training data shuffle seed is fixed to 42 in
`torchrun_main.py`, while `--seed` controls model/training randomness.

## Evaluation, continuation and release scope

This version starts from the supplied release archive with SHA-256
`caf2395224edcf8d928c3b29bb2f8510bebff4ad7411da82b85d4481b559ce8e`
and applies the corrections listed in `CORRECTIONS.md`.

The original five-condition update arithmetic is retained. Evaluation
aggregation and complete checkpoint continuation are corrected. Dedicated norm
diagnostics and their integrations are removed from this minimal release.
These changes do not establish that previously published results were computed
with the corrected metric; scientific checkpoints must be re-evaluated.
