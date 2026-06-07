# Hemorrhage HITL Operations Guide

This guide assumes you are already in the project root directory.

## 1. One-time Environment Setup

### Linux server

```bash
conda env create -f environments/environment.linux-cuda.yml
conda activate hemorrhage-hitl-linux
```

Verify PyTorch and CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## 2. Runtime Configuration

By default, the server runtime file expects:

- `data/` under the project root
- `workspace/` under the project root
- CUDA execution

The default server runtime file is:

```text
configs/runtime.server.yaml
```

If your dataset is stored outside the repo, update `configs/runtime.server.yaml` before running:

```yaml
paths:
  data_root: /absolute/path/to/data
  workspace_root: workspace
runtime:
  device: cuda
  allow_cpu_fallback: false
  smoke_mode: false
```

## 3. Full End-to-End Command Flow

All commands below assume you are in the project root.

### Progress And Status

You can inspect overall progress at any time:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml status
```

To inspect one specific round:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml status --round 1
```

The overview shows:

- each known round
- current round status
- budget
- review import progress
- checkpoint count
- report file availability

The single-round view also shows:

- routine/audit set sizes
- import completion flags
- review stats row count and warning count
- report/checkpoint availability
- OOF Dice summary when available
- stop state when available

### Step 1. Initialize the project

```bash
hemorrhage --runtime-config configs/runtime.server.yaml init-project
```

What this does:

- scans `data/imagesTr` and `data/labelsTr`
- validates label codes `{0,1,2,3}`
- creates the initial 5-fold split
- initializes `workspace/state.db`
- copies round 0 labels into `workspace/artifacts/labels/...`
- writes an initialization audit report

### Step 2. Train round 0

```bash
hemorrhage --runtime-config configs/runtime.server.yaml train-round0
```

What this does:

- trains 5 fold-specific models
- inside each training fold, creates a small deterministic validation split
- runs validation every epoch
- selects the best checkpoint by validation Dice
- generates OOF predictions for all cases
- builds round 0 soft targets and uncertainty maps
- writes round 0 checkpoints, reports, masks, and logs

### Step 3. Plan the first HITL round

You must provide the review budget here. Example: total budget `15`.

```bash
hemorrhage --runtime-config configs/runtime.server.yaml plan-round --round 1 --budget 15
```

What this does:

- scores eligible cases using the protocol
- selects routine and audit cases
- exports three review bundles:

```text
workspace/review/round_1/routine/
workspace/review/round_1/audit_anchor/
workspace/review/round_1/audit_final/
```

Bundle semantics:

- `routine/` contains:
  - `images/`
  - `labels_seed/`
  - `model_mask/`
  - `uncertainty/`
  - `manifest.csv`
- `audit_anchor/` contains:
  - `images/`
  - `labels_seed/`
  - `manifest.csv`
  - no model-assisted files
- `audit_final/` contains:
  - `images/`
  - `labels_seed/`
  - `model_mask/`
  - `uncertainty/`
  - `manifest.csv`
  - it is pre-exported at planning time so the model-assisted assets are available immediately
  - before `import-audit-anchor`, its `labels_seed/` is still a provisional copy of the current label
  - after `import-audit-anchor`, the pipeline refreshes `labels_seed/` in place so phase 2 starts from the anchor label

### Step 4. Import routine review results

After the routine reviewers return a normalized directory:

```text
routine_input/
  labels/
    CASE_A.nii.gz
    CASE_B.nii.gz
  metadata.csv
```

run:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml import-routine --round 1 --input /path/to/routine_input
```

`metadata.csv` should include `case_id`; `review_time` is optional but recommended. Missing time values are reported as warnings and do not block import.

### Step 5. Import audit anchor results

After the blind audit pass returns:

```text
audit_anchor_input/
  labels/
    CASE_X.nii.gz
    CASE_Y.nii.gz
  metadata.csv
```

run:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml import-audit-anchor --round 1 --input /path/to/audit_anchor_input
```

`metadata.csv` should include `case_id`; `anchor_time` is optional but recommended. Missing time values are reported as warnings and do not block import.

This does not create a new folder. Instead, it refreshes the existing:

```text
workspace/review/round_1/audit_final/
```

so that `labels_seed/` now comes from the anchor pass while the model-assisted files stay unchanged.

### Step 6. Import assisted audit results

After the assisted audit pass returns:

```text
audit_final_input/
  labels/
    CASE_X.nii.gz
    CASE_Y.nii.gz
  metadata.csv
```

run:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml import-audit-final --round 1 --input /path/to/audit_final_input
```

`metadata.csv` should include `case_id`; `assisted_time` is optional but recommended. Missing time values are reported as warnings and do not block import.

### Step 7. Finalize round 1

```bash
hemorrhage --runtime-config configs/runtime.server.yaml finalize-round --round 1
```

What this does:

- updates raw and binary labels
- updates review counters and re-entry eligibility
- fine-tunes the 5 fold-specific models
- runs per-epoch validation inside each training fold
- selects the best checkpoint by validation Dice
- regenerates OOF predictions, soft targets, uncertainty, postprocessed masks
- computes round metrics and OOF Dice summaries

### Step 8. Export the round report

```bash
hemorrhage --runtime-config configs/runtime.server.yaml report-round --round 1
```

Main outputs:

- `workspace/reports/round_1/summary.json`
- `workspace/reports/round_1/case_metrics.csv`
- `workspace/reports/round_1/review_stats.csv`
- `workspace/reports/round_1/review_warnings.csv`
- `workspace/reports/round_1/oof_summary.json`
- `workspace/reports/round_1/oof_fold_metrics.csv`
- `workspace/reports/round_1/oof_case_metrics.csv`

### Step 9. Continue to round 2 and beyond

Repeat the same sequence with the next round number:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml plan-round --round 2 --budget 15
hemorrhage --runtime-config configs/runtime.server.yaml import-routine --round 2 --input /path/to/routine_input_round2
hemorrhage --runtime-config configs/runtime.server.yaml import-audit-anchor --round 2 --input /path/to/audit_anchor_input_round2
hemorrhage --runtime-config configs/runtime.server.yaml import-audit-final --round 2 --input /path/to/audit_final_input_round2
hemorrhage --runtime-config configs/runtime.server.yaml finalize-round --round 2
hemorrhage --runtime-config configs/runtime.server.yaml report-round --round 2
```

Backward-compatible aliases still exist:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml import-phase1 --round 1 --input /path/to/combined_input
hemorrhage --runtime-config configs/runtime.server.yaml import-phase2 --round 1 --input /path/to/audit_final_input
```

They emit deprecation warnings and map to the new routine/audit commands.

## 4. Background Execution

The CLI runs in the foreground by default. If you close the terminal, the job may stop.

Recommended options:

### Option A. `tmux`

```bash
tmux new -s hemorrhage
hemorrhage --runtime-config configs/runtime.server.yaml train-round0
```

Detach:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t hemorrhage
```

### Option B. `nohup`

```bash
mkdir -p logs
nohup hemorrhage --runtime-config configs/runtime.server.yaml train-round0 > logs/train_round0.out 2>&1 &
```

For later rounds:

```bash
nohup hemorrhage --runtime-config configs/runtime.server.yaml finalize-round --round 1 > logs/finalize_round_1.out 2>&1 &
```

## 5. How to Monitor Progress

### Main command log

Each command writes a timestamped log under `workspace/logs/`.

Examples:

- `workspace/logs/round_0/train-round0.log`
- `workspace/logs/round_1/plan-round.log`
- `workspace/logs/round_1/import-routine.log`
- `workspace/logs/round_1/import-audit-anchor.log`
- `workspace/logs/round_1/import-audit-final.log`
- `workspace/logs/round_1/finalize-round.log`
- `workspace/logs/round_1/report-round.log`

Tail the log:

```bash
tail -f workspace/logs/round_0/train-round0.log
```

### Per-fold training progress

Per-fold training CSV is updated during training:

- `workspace/reports/round_0/fold_1_train.csv`
- `workspace/reports/round_0/fold_2_train.csv`
- ...

The CSV includes:

- training loss per epoch
- validation Dice per epoch
- whether the epoch produced a new best checkpoint
- the current best epoch and best metric value

Watch one fold:

```bash
tail -f workspace/reports/round_0/fold_1_train.csv
```

Per-fold training status JSON:

- `workspace/reports/round_0/fold_1_train_status.json`

Example:

```bash
cat workspace/reports/round_0/fold_1_train_status.json
```

Important fields:

- `epochs_completed`
- `last_epoch_loss`
- `last_val_metrics`
- `best_metric_name`
- `best_metric_value`
- `best_epoch`
- `train_case_ids`
- `val_case_ids`

### Per-fold inference progress

Per-fold inference status JSON:

- `workspace/reports/round_0/fold_1_inference_status.json`

Example:

```bash
cat workspace/reports/round_0/fold_1_inference_status.json
```

### GPU monitoring

```bash
watch -n 5 nvidia-smi
```

## 6. Important Outputs

### Database

```text
workspace/state.db
```

### Checkpoints

```text
workspace/artifacts/checkpoints/round_0/fold_1.pt
workspace/artifacts/checkpoints/round_0/fold_1_last.pt
workspace/artifacts/checkpoints/round_1/fold_1.pt
workspace/artifacts/checkpoints/round_1/fold_1_last.pt
...
```

Checkpoint naming:

- `fold_k.pt` is the selected best checkpoint based on validation Dice
- `fold_k_last.pt` is the final checkpoint from the last epoch

### OOF predictions

```text
workspace/artifacts/oof/round_0/CASE_ID.npz
workspace/artifacts/oof/round_1/CASE_ID.npz
...
```

### Postprocessed masks

```text
workspace/artifacts/masks/round_0/CASE_ID.nii.gz
workspace/artifacts/masks/round_1/CASE_ID.nii.gz
...
```

### Reports

```text
workspace/reports/round_0/
workspace/reports/round_1/
...
```

## 7. External Prediction

After training is complete:

```bash
hemorrhage --runtime-config configs/runtime.server.yaml predict-external --model-tag final --input-dir /path/to/external_images --output-dir /path/to/output
```

Expected input naming:

- `CASE_A_0000.nii.gz`
- `CASE_B_0000.nii.gz`

Expected outputs:

- `CASE_A.npz` with probability map
- `CASE_A.nii.gz` with postprocessed binary mask

## 8. Common Notes

- `init-project` and `train-round0` do not require a review budget.
- The review budget is only provided when you run `plan-round`.
- If you are already in the project root, you do not need `--project-root .` unless you want to be explicit.
- The pipeline currently expects normalized review import directories and does not parse Word files directly.
- `routine` and `audit` are parallel review branches; only `audit` itself has `anchor -> assisted` two-stage review.
- The assisted bundles export a binary model mask for clinical viewing, but the internal training state still uses probability maps and uncertainty maps.
