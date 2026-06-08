# Next Pipeline Implementation Audit

This audit maps `project plan/1.txt` to the current `next/` implementation.

## Scope

The implemented pipeline is a standalone next-stage experiment:

```text
workspace_v0 EpiBios R1 source model
  -> ARAMRA prediction and scoring
  -> fold-local animal selection
  -> fine-tuning with selected ARAMRA labels plus EpiBios replay
  -> ARAMRA animal-level OOF evaluation
  -> EpiBios retention evaluation
  -> auxiliary ARAMRA-internal OOF analysis
```

It does not modify the original HITL workflow and does not assume new human
revision of ARAMRA labels.

## Requirement Mapping

| Requirement from plan/objective | Implementation evidence | Status |
|---|---|---|
| New independent `next` folder | `next/run.py`, `next/scripts/run_next_pipeline.py`, `next/configs/*.yaml`, `next/README.md` | Implemented |
| Use `workspace_v0` model for fine-tuning | `source_checkpoint_path()`, `load_checkpoint_compatible()`, `train_adapted_source_models()` | Implemented |
| No held-out test yet; use OOF | `make_aramra_folds()`, `select_for_fold()`, `evaluate_experiment_fold()` | Implemented |
| Split by ARAMRA animal, not scan | `make_aramra_folds()` assigns one fold per `animal_id_strict` | Implemented |
| Held-out ARAMRA animals cannot be selected or fine-tuned | `select_for_fold()` filters candidates to `record.fold != outer_fold`; `write_selection_tables()` has leakage guard | Implemented |
| Predict ARAMRA with source model | `predict_source_on_aramra()` | Implemented |
| Compute score/rank | `animal_score_table()`, `select_animals_*()` | Implemented |
| Selection must not only choose worst cases | `select_animals_balanced_utility()` mixes non-extreme mid/high utility, representative low/mid, and M5/small-lesion risk cases | Implemented |
| Compare selection strategies | `planned_experiments()` supports `source_only`, `random`, `disagreement`, `uncertainty`, `balanced_utility`, `all_train_pool`; random selection balances M5 coverage, case count, and lesion burden | Implemented |
| Fine-tune selected ARAMRA labels plus EpiBios replay | `train_adapted_source_models()` | Implemented |
| EpiBios replay should avoid direct source-fold leakage | `epibios_replay_mode: source_fold_train_only`; source fold holdout is excluded per source model | Implemented |
| Main ARAMRA OOF evaluation | `evaluate_experiment_fold()`, `run_main_experiments()` | Implemented |
| EpiBios retention evaluation | `evaluate_experiment_fold()` retention branch | Implemented |
| Auxiliary ARAMRA-internal OOF model | `run_aux_aramra_internal_oof()` | Implemented |
| Complete metrics | `mask_metrics()`, `component_metrics()`, `metric_row()` include Dice, Jaccard, HD95, ASSD, RVE, abs RVE, volume similarity, surface Dice, kappa, precision, sensitivity, specificity, lesion-F1, volume error | Implemented |
| Clear output hierarchy | `ensure_run_dirs()`, `README.md` output layout | Implemented |
| Default background launch | `next/run.py`, `main()`, `launch_background()` | Implemented |
| Foreground smoke mode | `--foreground` in `parse_args()` | Implemented |
| Run state and logs | `run_state.json`, `TeeLogger`, `logs/launcher.*.log`, `logs/pipeline.log` | Implemented |
| Early path/checkpoint validation | `preflight_checks()` writes `config/preflight_report.json` before expensive stages | Implemented |
| Final report | `write_report()` writes `report.md` | Implemented |

## Configurations

- `next/configs/server_next.yaml`
  - Full intended server run.
  - Uses `/fs04/ea78/BraTS2021/hemorrhage_hitl/ARAMRA002_standardized`.
  - Keeps the default adaptation learning rate aligned with the original
    workspace_v0 fine-tuning setting (`5.0e-6`).
  - Runs 5 ARAMRA folds, source-only, random, disagreement, uncertainty,
    balanced utility, all-train-pool, EpiBios retention, and auxiliary ARAMRA
    internal OOF.

- `next/configs/local_smoke.yaml`
  - Small foreground test config.
  - Keeps the model channel configuration compatible with `workspace_v0`
    checkpoints.
  - Requires a local ARAMRA standard folder at `E:/Hemorrhage/ARAMRA002_standardized`
    or an edited `paths.aramra_root`.

## Current Static Verification

The following compile checks passed in the current worktree:

```text
python -m py_compile analysis/animalwise_oof_pipeline/scripts/run_animalwise_oof.py next/scripts/run_next_pipeline.py
```

The current Codex runtime does not include `torch`, `yaml`, `nibabel`, or
`scipy`, and this workspace session does not have the ARAMRA standard folder
mounted. Therefore runtime smoke testing must be performed in the
`BraTS_Spacing` environment on the machine/server with the data mounted.

## Recommended Runtime Verification

First run a small foreground check:

```bash
conda activate BraTS_Spacing
python next/scripts/run_next_pipeline.py --config next/configs/local_smoke.yaml --foreground
```

If local ARAMRA is not mounted at the configured path, edit `paths.aramra_root`
or run the server config on the server:

```bash
conda activate BraTS_Spacing
python next/scripts/run_next_pipeline.py --config next/configs/server_next.yaml --foreground
```

After the foreground check reaches `status=completed`, launch the background
server run:

```bash
python next/scripts/run_next_pipeline.py --config next/configs/server_next.yaml
```

## Runtime Completion Evidence To Check

A complete run should produce:

```text
run_state.json                         status=completed
metadata/metadata_master.csv
config/preflight_report.json            status=pass
splits/split_integrity_report.json     status=pass
scores/source_scores_case.csv
selection/selected_animals.csv
metrics/source_only_aramra_case_metrics.csv
metrics/aramra_oof_case_metrics.csv
metrics/aramra_oof_summary.csv
metrics/epibios_retention_case_metrics.csv
metrics/epibios_retention_summary.csv
metrics/aux_aramra_internal_oof_case_metrics.csv
metrics/aux_aramra_internal_oof_summary.csv
metrics/aux_source_internal_difficulty_crosswalk.csv
report.md
```

## Caveats

- The implementation is intentionally simple on loss design: it reuses the
  existing 0.5 weighted BCE plus 0.5 soft Dice loss through the animalwise
  runtime. It does not add trust weighting, EMA consistency, FiLM, or target
  domain conditioning.
- `--resume` reuses an existing run directory but does not yet skip completed
  stages automatically. Treat it as a controlled restart target.
- The full server config is computationally heavy. A first operational run can
  reduce `selection.methods` and `selection.budgets_animals` before launching
  the full matrix.
