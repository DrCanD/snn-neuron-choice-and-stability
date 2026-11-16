# Neuron Choice and Initialization Stability in SNNs

**One repository for the original study, supplementary experiments, and final revision controls on DIAT-μSAT and SHD.**

[![Release v0.2.0](https://img.shields.io/badge/release-v0.2.0-2ea44f.svg)](https://github.com/DrCanD/snn-neuron-choice-and-stability/releases/tag/v0.2.0)
[![DOI](https://zenodo.org/badge/1460193939.svg)](https://doi.org/10.5281/zenodo.24285109)

**Authors:** İsmail Can Dikmen · Teoman Karadağ

This package produces numerical results, predictions, histories, checkpoints, and numerical statistical analyses. All required model sources are included locally. No second research repository, automatic source download, or Google Drive mount is needed.

Version **0.1.0 covered only the final revision controls**. Version **0.2.0 adds the original study programs and S1–S9 supplementary experiment/analysis code**. Their distinct historical and revision protocols are retained; their results must not be pooled as interchangeable repetitions.

**Reproduction boundary:** supply the study's processed DIAT arrays, official SHD files, and—for the original mixed-speaker experiments—the historical SHD cache. Datasets and old measured results are not bundled. Original raw-radar preprocessing and historical mixed-SHD cache construction are not available in the verified source archive. This is an experiment-code package for those study inputs, not a reconstruction of the missing data-preparation steps. Historical GPU/software/RNG trajectories are not fully recoverable; exact reproduction of published scores is not asserted.

## Setup

```bash
git clone https://github.com/DrCanD/snn-neuron-choice-and-stability.git
cd snn-neuron-choice-and-stability
python run.py --help
python run.py plan --experiment paper-diat-ablation
python run.py plan --experiment shd --seed 42
```

`plan` and `check` use the Python standard library only and never load models or start training. Training requires **Linux or Colab**. Use Python **3.13.16** and install:

```bash
python -m pip install -r requirements.txt
```

Final revision controls require their recorded PyTorch **2.14.1+cu130**, CUDA **13.0**, cuDNN **92800**, and package versions. DIAT controls require **NVIDIA RTX PRO 6000 Blackwell Server Edition**; controlled SHD and the separate DIAT reference require **NVIDIA L4**. Those runners stop on environment mismatches. Installing dependencies alone does not provision matching hardware or cuDNN.

Original programs retain historical experiment settings and record the current environment. Their full historical environment was not archived; the Optuna/tqdm lower bounds are not a completely locked historical search environment.

## Inputs

Keep input and output directories separate:

- **DIAT:** the study's processed `X.npy` (`[N, 64, 64]`) and `y.npy`. Expected hashes: [configs/diat.json](configs/diat.json).
- **Official SHD:** decompressed `shd_train.h5` and `shd_test.h5` from [Zenke Lab](https://zenkelab.org/datasets/). Expected hashes: [configs/shd.json](configs/shd.json). The official split is **not fully speaker-disjoint**, despite historical script names containing `disjoint`.
- **Mixed-speaker SHD:** the original `shd_T12_m200_v2.pt`. No independently verified historical file hash was recovered. `check` reports its current hash; training also requires a trusted archive hash via `--data-sha256`. A newly generated cache must not be presented as the historical input.

```bash
python run.py check --experiment paper-diat-ablation --data /data/diat
python run.py check --experiment paper-shd-official --data /data/shd
python run.py check --experiment paper-shd-mixed --data /data/shd_T12_m200_v2.pt
python run.py check --experiment S7_signal --data /data/diat --shd-data /data/shd
```

Missing or mismatched inputs stop execution. There is no synthetic-data fallback.

## Original study

Use `run.py plan --experiment NAME` for source programs, seeds, and run counts. Original programs retain their complete method/seed order; omit `--seed` and `--resume`.

- `paper-diat-ablation`: eight-method homogeneous, vertical, order-permuted, and horizontal-bank comparison; 24 runs.
- `paper-search-cnn`: full Doppler/vanilla-LIF/parametric-LIF search; 90 search trials, 15–30 selected-configuration reevaluations, and five CNN runs. Trials may be pruned early.
- `paper-shd-mixed`: eight-family mixed-speaker SHD comparison; 24 runs.
- `paper-diat-ranking`: Doppler, CrossInhib, and Phase on DIAT; nine runs.
- `paper-shd-official`: eight-family official-split SHD comparison; 24 runs.
- `paper-search` and `paper-shd-probe`: preliminary programs retained separately from the final search/ranking.

Only explicit `train` commands start experiments. Every invocation requires a **new output directory**:

```bash
python run.py train --experiment paper-diat-ablation --data /data/diat --output outputs/paper-ablation
python run.py train --experiment paper-search-cnn --data /data/diat --output outputs/paper-search-cnn
python run.py train --experiment paper-diat-ranking --data /data/diat --output outputs/paper-diat-ranking
python run.py train --experiment paper-shd-official --data /data/shd --output outputs/paper-shd-official
python run.py train --experiment paper-shd-mixed --data /data/shd_T12_m200_v2.pt --data-sha256 TRUSTED_ARCHIVE_SHA256 --output outputs/paper-shd-mixed
```

Original output layouts and selection rules are retained. Some historical checkpoints contain last-epoch weights while selected predictions/metrics refer to the best validation epoch; these checkpoints are not selected-model or exact-resume evidence.

## Supplementary experiments

The archived definitions cover **195 runs**:

- `S1_cap_sweep`: vanilla/parametric/Doppler at five widths; 45 runs.
- `S2_regime`: parameter-matched regimes, unit-matched bank, stable bank; 24 runs.
- `S3_rescue`: initialization/learning-rate controls and STFT/Phase diagnostics; 15 runs.
- `S4_shd`: Doppler and library/stable CrossInhib; nine runs.
- `S6_seeds`: two additional seeds on both domains; 12 runs.
- `S7_signal`: DIAT first differences and SHD cumulative representations; 18 runs.
- `S8_diat` / `S8_shd`: temporal pooling/coupling sweep and 64-bin SHD; 21 / nine runs.
- `S9_lifdiff`: vanilla/parametric LIF on DIAT first differences; six runs.
- `S9_initfix`: STFT/dual-tau initialization controls on both domains; 15 runs.
- `S9_shd_rank`: six remaining original SHD families plus parametric LIF; 21 runs.

Pass DIAT as `--data` for DIAT stages, SHD as `--data` for SHD-only stages. Both-domain stages (`S6_seeds`, `S7_signal`, `S9_initfix`) additionally require `--shd-data`:

```bash
python run.py train --experiment S1_cap_sweep --data /data/diat --output outputs/S1
python run.py train --experiment S4_shd --data /data/shd --output outputs/S4
python run.py train --experiment S7_signal --data /data/diat --shd-data /data/shd --output outputs/S7
```

Run remaining selectors in their own fresh directories. Method/seed order, calibration, loader lifetime within each stage, and the original disposable timing probe remain part of the historical protocol. A fresh process per stage does not reconstruct an uninterrupted notebook RNG trajectory.

**S5 is numerical analysis, not training.** After all supplementary stages finish:

```bash
python analyze.py supplementary --results outputs/S1 outputs/S2 outputs/S3 outputs/S4 outputs/S6 outputs/S7 outputs/S8-diat outputs/S8-shd outputs/S9-lifdiff outputs/S9-initfix outputs/S9-rank --data /data/diat --output supplementary-statistics.json
```

This requires all 195 records from the packaged supplementary protocol. It produces numerical summaries, diagnostics, paired bootstraps, and rank statistics in one new JSON file. Input results are not modified; no manuscript tables or figures are generated.

## Final revision controls

These retain the frozen computation from v0.1.0. Run one seed per invocation:

- `diat-capacity`: Doppler/library-bank/stable-bank at widths 64/128/256; 31 jobs including two extra W128 seeds.
- `diat-signal`: original/first-difference inputs for vanilla/parametric/Doppler; 18 jobs. Three original-input Doppler settings overlap with `diat-capacity` and are not additional independent evidence.
- `diat-stability`: library E–I and couplings 0.20/0.28/0.32 at 16/64 frames; 24 jobs.
- `diat-reference`: separate L4 Doppler/stable-bank reference with its own loader protocol; six jobs. Keep this cohort separate.
- `shd`: two **homogeneous** families × five resolution/update/order conditions × three seeds; 30 jobs. This is not a heterogeneous-bank SHD experiment.

```bash
python run.py train --experiment diat-capacity --seed 42 --data /data/diat --output outputs/capacity-s42
python run.py train --experiment diat-signal --seed 42 --data /data/diat --output outputs/signal-s42
python run.py train --experiment diat-stability --seed 42 --data /data/diat --output outputs/stability-s42
python run.py train --experiment diat-reference --seed 42 --data /data/diat --output outputs/reference-s42
python run.py train --experiment shd --seed 42 --data /data/shd --output outputs/shd/seed42
```

Use other recorded seeds from `plan` with separate directories. For these five selectors only, `--resume` accepts this version's matching output. SHD/reference retain checkpoint recovery; DIAT controls skip completed jobs but reject interrupted individual jobs. Old-version or unrelated output directories are never reused.

SHD `train` includes preflight optimizer steps on disposable models. Neither `plan`, `check`, nor `analyze.py` executes those steps.

After all 30 SHD jobs finish, verify predictions, selected checkpoints, completion, initialization, and minibatch pairing and compute the ten Holm-adjusted contrasts:

```bash
python analyze.py shd --results outputs/shd --output shd-statistics.json
```

For equivalence sensitivity, supply **one protocol cohort at a time**, with identical complete seed sets and test-label order:

```bash
python analyze.py paired --a BANK_SEED42.json BANK_SEED123.json BANK_SEED999.json --b DOPPLER_SEED42.json DOPPLER_SEED123.json DOPPLER_SEED999.json --classes 6 --output equivalence.json
```

Scores are recomputed from predictions. Outputs include paired differences, t intervals, sign-flip probability, and exploratory TOST at ±0.01/0.02/0.03/0.04 macro-F1. Seed pairing does not establish identical minibatch/RNG trajectories. Do not select passing margins after viewing results or pool historical, revision-control, and L4-reference cohorts.

## Integrity and validation

`manifest.json` verifies executable/configuration bytes. [sources/provenance.json](sources/provenance.json) records original and packaged hashes. Sources originate from:

- [neuron-task-matching@96c2c484](https://github.com/DrCanD/neuron-task-matching/tree/96c2c484c45114afed21c0a7e768d3cba9b3c723), whose experiment code matches the old `v1.1.0` tag; the intervening change was documentation.
- [dikmen-spiking-neurons@ebab00ac](https://github.com/DrCanD/dikmen-spiking-neurons/tree/ebab00ac1a177ae08472db87ad5051f0d6ca7ebb), restricted to the seven MS-IF implementations used by this study, with unchanged neuron computation.
- Archived supplementary notebook definitions and audited final-revision statistical calculations; their original hashes are recorded in the provenance file.

Packaging replaces installation/Drive setup with local imports and explicit paths, requires real data, guards legacy scripts against accidental import execution, and removes plotting/presentation blocks. Historical training/model definitions and frozen final-revision computation are retained.

**Validation uses no training:** source/syntax/integrity checks, all 23 CLI plans, missing-input rejection, all 195 supplementary dispatch slots, CPU model state/RNG/forward checks, and independent statistical checks. Optimizer construction and backward are blocked during these checks. Existing research-result fingerprints are checked for changes. **No end-to-end GPU training rerun of this package has been performed.**

## Citation and license

Cite the archived release you use through [Zenodo](https://doi.org/10.5281/zenodo.24285109) or GitHub's **Cite this repository** action using [CITATION.cff](CITATION.cff). Cite datasets separately. The DOI identifies software, not new measurements.

The harness and experiment programs use [MIT](LICENSE). The bundled neuron library retains its [Apache-2.0 license](sources/dikmen_neurons/LICENSE) and [NOTICE](sources/dikmen_neurons/NOTICE). Both licenses apply to the distributed package.
