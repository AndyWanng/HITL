# Hemorrhage HITL

Protocol-first hemorrhage segmentation pipeline with a human-in-the-loop review workflow.

The project trains a 5-fold out-of-fold baseline, plans review rounds, imports routine and audit annotations, fine-tunes fold-specific models, and exports reports plus external predictions.

## What it does

- Scans a NIfTI dataset under `data/imagesTr` and `data/labelsTr`
- Validates geometry and label codes before training
- Builds a fixed 5-fold split with strict OOF discipline
- Trains round 0 models and generates soft targets plus uncertainty maps
- Selects routine and audit review cases for each round
- Imports reviewer outputs and updates case state in `workspace/state.db`
- Fine-tunes the model after each completed review round
- Exports per-round logs, checkpoints, masks, and summary reports

## Repository layout

```text
configs/        Runtime, protocol, and model configuration
docs/           Operational documentation
environments/   Conda environment definitions
plans/          Protocol and planning notes
src/            Python package source
tests/          Unit and smoke tests
data/           Local training data (ignored by Git)
workspace/      Generated artifacts, logs, and reports
```

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended for training
- NIfTI data laid out as:

```text
data/
  imagesTr/
    CASE_A_0000.nii.gz
  labelsTr/
    CASE_A.nii.gz
```

The pipeline expects label codes in `{0, 1, 2, 3}` and projects them to a binary hemorrhage mask internally.

## Environment setup

### Option 1: Conda on Windows

```bash
conda env create -f environments/environment.win-cuda.yml
conda activate hemorrhage-hitl-win
```

### Option 2: Editable install with pip

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Runtime configs

- `configs/runtime.local.yaml`: local CUDA run with CPU fallback enabled
- `configs/runtime.server.yaml`: server-oriented CUDA run with CPU fallback disabled

Both configs default to:

- `data_root: data`
- `workspace_root: workspace`

## Quick start

Initialize the project:

```bash
hemorrhage --runtime-config configs/runtime.local.yaml init-project
```

Train the round 0 baseline:

```bash
hemorrhage --runtime-config configs/runtime.local.yaml train-round0
```

Plan the first HITL round:

```bash
hemorrhage --runtime-config configs/runtime.local.yaml plan-round --round 1 --budget 15
```

Import returned reviewer outputs:

```bash
hemorrhage --runtime-config configs/runtime.local.yaml import-routine --round 1 --input path\to\routine_input
hemorrhage --runtime-config configs/runtime.local.yaml import-audit-anchor --round 1 --input path\to\audit_anchor_input
hemorrhage --runtime-config configs/runtime.local.yaml import-audit-final --round 1 --input path\to\audit_final_input
```

Finalize and report the round:

```bash
hemorrhage --runtime-config configs/runtime.local.yaml finalize-round --round 1
hemorrhage --runtime-config configs/runtime.local.yaml report-round --round 1
```

Run external inference after training:

```bash
hemorrhage --runtime-config configs/runtime.local.yaml predict-external --model-tag final --input-dir path\to\images --output-dir path\to\outputs
```

## Main outputs

- `workspace/state.db`: pipeline state database
- `workspace/artifacts/checkpoints/`: best and last checkpoints per fold
- `workspace/artifacts/oof/`: OOF prediction `.npz` files
- `workspace/artifacts/masks/`: exported binary masks
- `workspace/review/`: review bundles for routine and audit flows
- `workspace/reports/`: per-round summaries, metrics, and training traces
- `workspace/logs/`: command logs

## Development

Run tests:

```bash
python -m pytest tests -q
```

Notes:

- `tests/test_smoke.py` requires a local CUDA GPU and is skipped otherwise.
- The package exposes the CLI entrypoint as `hemorrhage`.

## Additional documentation

- Operations guide: `docs/GUIDE.md`
- Protocol notes: `plans/Hemorrhage Segmentation.md`
