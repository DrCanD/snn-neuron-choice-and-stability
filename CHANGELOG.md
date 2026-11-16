# Changelog

## 0.2.0 — 2026-11-16

Complete the experiment-code scope beyond the revision controls included in 0.1.0.

- Add all seven original experiment programs: ablation, search/CNN, mixed/official SHD, and cross-domain ranking programs.
- Add S1–S9 supplementary definitions: 195 experiment runs and numerical S5 analysis.
- Bundle required model sources with provenance hashes and original MIT/Apache-2.0 attribution.
- Add numerical SHD contrast and paired equivalence analyses; no tables or figures.
- Require real inputs, protect fresh outputs, preserve historical stage order, and reject unsupported legacy resume.
- Document missing historical data-preparation sources and environment limits explicitly.

Existing research results and frozen revision computation remain unchanged. Validation is offline, with training operations blocked. No new training or end-to-end GPU rerun was performed.

## 0.1.0 — 2026-11-16

Initial packaged release of the DIAT and SHD revision experiment runners.

- One entry point for capacity, input-representation, E–I stability, the separate DIAT reference, and controlled SHD experiments.
- Fixed model-source commits, input fingerprints, environment checks, and per-seed experiment plans.
- Numerical results, validation histories, predictions, and model checkpoints as outputs.
- Input-only checks, isolated output directories, and protocol-specific recovery behavior.
- Usage documentation and software citation metadata for GitHub and Zenodo.

Training code, experiment settings, and existing measured results were not changed while preparing the documentation and release. Packaging checks were performed without new training. An end-to-end GPU rerun of the portable package has not been performed.
