# Neuron Choice and Initialization Stability in SNNs

**Experiment code for comparing neuron families, heterogeneous banks, input representations, and initialization stability.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13.16](https://img.shields.io/badge/Python-3.13.16-3776AB.svg)](requirements.txt)
[![Release](https://img.shields.io/github/v/release/DrCanD/snn-neuron-choice-and-stability)](https://github.com/DrCanD/snn-neuron-choice-and-stability/releases)
[![DOI](https://zenodo.org/badge/1460193939.svg)](https://zenodo.org/badge/latestdoi/1460193939)

This repository packages the completed revision experiments on **DIAT-μSAT radar** and **Spiking Heidelberg Digits (SHD)**. A single command-line entry point selects an experiment and seed. Each run writes numerical results, validation histories, predictions, and model checkpoints.

The implementation pins the model sources, input fingerprints, and experiment settings. The repository contains code and configuration; datasets, previously measured results, manuscript files, and plotting tools are not bundled.

**Authors:** İsmail Can Dikmen · Teoman Karadağ

[Getting started](#getting-started) · [Experiments](#experiments) · [Data](#data) · [Running experiments](#running-experiments) · [Reproducibility](#reproducibility) · [Citation](#citation)

## Getting started

```bash
git clone https://github.com/DrCanD/snn-neuron-choice-and-stability.git
cd snn-neuron-choice-and-stability
```

Inspect an experiment without loading data or starting training:

```bash
python run.py plan --experiment shd --seed 42
python run.py --help
```

`plan` and `check` use the Python standard library only. Training requires **Linux or Colab**, Git, **Python 3.13.16**, and the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The recorded environments use PyTorch **2.14.1+cu130**, CUDA runtime **13.0**, and cuDNN **92800**. The DIAT controls use **NVIDIA RTX PRO 6000 Blackwell Server Edition**; SHD and the separate DIAT reference use **NVIDIA L4**. Training checks the recorded environment and stops on a mismatch. Installing the Python dependencies alone does not provision the required GPU or guarantee a matching cuDNN installation.

## Experiments

| Selector | Comparison | Recorded seeds | Runs |
| :--- | :--- | :--- | ---: |
| `diat-capacity` | Doppler, library-initialized bank, and stable-initialized bank at widths 64, 128, and 256 | 42, 123, 999; also 7 and 2024 for W128 Doppler/stable bank | 31 |
| `diat-signal` | Original and signed first-difference inputs for vanilla LIF, parametric LIF, and Doppler | 42, 123, 999 | 18 |
| `diat-stability` | Library E–I initialization and couplings 0.20, 0.28, 0.32 at 16 and 64 frames | 42, 123, 999 | 24 |
| `diat-reference` | Separate L4 Doppler/stable-bank reference with its recorded training and loader protocol | 42, 123, 999 | 6 |
| `shd` | Two homogeneous families × five resolution/update/order conditions | 42, 123, 999 | 30 |

Counts describe selectable jobs, not independent replications: the three original-input W128 Doppler settings in `diat-signal` overlap with `diat-capacity`. The separate DIAT reference has a different protocol and must remain a separate cohort.

The SHD conditions are `fine48_N48`, `coarse12_N48`, `fine48_N96`, `coarse12_N96`, and `shuffled48_N48`. Here, the first number specifies input bins and `N` specifies network updates. SHD compares homogeneous Doppler and stable E–I networks; it does not test a heterogeneous bank.

## Data

Supply the study inputs in a folder outside the output directory:

```text
data/
├── diat/
│   ├── X.npy
│   └── y.npy
└── shd/
    ├── shd_train.h5
    └── shd_test.h5
```

- **DIAT-μSAT:** the study's processed `X.npy` (`[N, 64, 64]`) and `y.npy`. Arbitrary preprocessing of the original radar data is not interchangeable with these inputs. The raw-to-processed conversion is outside this repository.
- **SHD:** the official, decompressed HDF5 train/test files from [Zenke Lab](https://zenkelab.org/datasets/). The runner constructs the fixed train/validation split and temporal representations. The official test partition should not be described as fully speaker-disjoint.

Expected SHA-256 fingerprints are stored in [the DIAT configuration](configs/diat.json) and [the SHD configuration](configs/shd.json). Check input files before allocating training time:

```bash
python run.py check --experiment diat-capacity --data data/diat
python run.py check --experiment shd --data data/shd
```

`check` validates raw file hashes. It does not construct a model, validate GPU execution, or perform optimizer updates. Detailed split and runtime checks occur inside the training runner.

## Running experiments

Only the explicit `train` command starts a scientific run. Run one seed per invocation, with a **new output directory**:

```bash
python run.py train --experiment diat-capacity --seed 42 --data data/diat --output outputs/capacity-s42
python run.py train --experiment diat-signal --seed 42 --data data/diat --output outputs/signal-s42
python run.py train --experiment diat-stability --seed 42 --data data/diat --output outputs/stability-s42
python run.py train --experiment diat-reference --seed 42 --data data/diat --output outputs/reference-s42
python run.py train --experiment shd --seed 42 --data data/shd --output outputs/shd-s42
```

Choose the matching GPU environment for each experiment. Replace `42` with another seed from the experiment plan. Separate seed invocations can run in separate environments with distinct output directories.

Pinned upstream code is downloaded into `.sources/` on the first training invocation. Use `--sources /path/to/cache` to choose another source cache. Existing source checkouts are verified and are not silently replaced.

### Output and recovery

Each completed job produces a `result.json` containing macro-F1, accuracy, predictions, validation history, and selection information, together with a selected model checkpoint. The separate DIAT reference retains its original result layout, including metrics under `best`. Additional progress and recovery records are saved by the relevant runner.

To continue this package's own matching output, repeat the command with `--resume`:

```bash
python run.py train --experiment shd --seed 42 --data data/shd --output outputs/shd-s42 --resume
```

- SHD and the DIAT reference retain checkpoint recovery.
- DIAT controls skip verified completed jobs but refuse to overwrite an interrupted individual run. Restart that attempt in a new output directory.
- Output identity includes the experiment, seed, input path, and pipeline manifest. An unrelated existing directory is rejected.

The SHD training command begins with bounded diagnostic checks, including optimizer steps on disposable models. These checks are part of `train`; they are never executed by `plan` or `check`.

## Reproducibility

The model implementations are pinned to:

- [`neuron-task-matching@96c2c484`](https://github.com/DrCanD/neuron-task-matching/tree/96c2c484c45114afed21c0a7e768d3cba9b3c723)
- [`dikmen-spiking-neurons@ebab00ac`](https://github.com/DrCanD/dikmen-spiking-neurons/tree/ebab00ac1a177ae08472db87ad5051f0d6ca7ebb)

The package verifies its executable/configuration files through `manifest.json`. The experiment contracts also fix data hashes, source hashes, numerical flags, calibration, and model-selection rules.

**Pairing is protocol-specific.** DIAT controls use a private seeded loader but calibrate each model/view separately. SHD clones the complete initial state and reference-calibrated thresholds across the five conditions of each model/seed, shares affine initialization across model families, and checks common epoch permutations. The separate DIAT reference retains its own loader lifecycle. Keep these cohorts separate.

**Validation of this packaged release:** syntax and manifest checks, five command-line plans, input-rejection checks, unchanged SHD computation methods, unchanged DIAT reference training definitions, and exact CPU initial-state comparisons for all 31 capacity jobs passed. No new training was started to validate the portable package. An end-to-end GPU training rerun of this packaging has not been performed; numerical identity across other environments is not asserted.

## Repository layout

```text
run.py             Experiment selection, input checks, and output protection
runtime.py         Source pinning and environment verification
diat.py            DIAT capacity, representation, and stability controls
shd.py             Controlled SHD experiment
configs/           Recorded inputs and experiment settings
reference/         Separate DIAT reference runner and frozen definitions
requirements.txt   Pinned Python dependencies
manifest.json      Executable/configuration file fingerprints
```

## Citation

Use the **Zenodo DOI badge** above to cite an archived software release. GitHub's **Cite this repository** action uses [CITATION.cff](CITATION.cff). Cite the version you actually used, and cite the original datasets separately.

The software archive and its DOI identify the code release, not a new experimental dataset or a completed GPU rerun.

## License

[MIT](LICENSE). The upstream neuron library and experiment harness retain their respective attribution and licenses.
