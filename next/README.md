# Next ARAMRA Selection Fine-tuning Pipeline

This folder implements the next-stage experiment described in `project plan/1.txt`.

The pipeline is intentionally standalone and launched by a Python script rather than a new package CLI. By default it starts a background worker; use `--foreground` for smoke tests.

## Experiment Question

Can model-guided selection of a small number of labelled ARAMRA animals improve performance more efficiently than random selection, while retaining EpiBios performance?

The implementation keeps the distinction in the plan:

- Original HITL review selection can prioritise the hardest disagreement cases because a human revises them.
- This stage has no new human revise, so fine-tuning selection must avoid using only extreme disagreement cases.

## Main Pipeline

For each ARAMRA animal-level fold:

1. Hold out one ARAMRA animal fold for evaluation only.
2. Predict ARAMRA with the workspace_v0 EpiBios R1 source ensemble.
3. Compute source-model selection features on training-fold animals only:
   - model-label disagreement,
   - ensemble uncertainty,
   - predicted-mask review-cost proxy.
4. Select ARAMRA animals using:
   - source-only baseline,
   - random stratified selection balancing M5 coverage, case count, and lesion burden,
   - disagreement-only selection,
   - uncertainty-only selection,
   - balanced utility selection,
   - all-train-pool upper bound.
5. Fine-tune each workspace_v0 R1 source fold model with:
   - selected ARAMRA animals from the training folds,
   - EpiBios replay data excluding the corresponding source held-out fold.
6. Evaluate the adapted 5-model ensemble on held-out ARAMRA animals.
7. Evaluate EpiBios retention using source-fold OOF-style predictions.

## Auxiliary Pipeline

The auxiliary stage trains an ARAMRA-internal animal-level OOF model from scratch. This is used to distinguish:

- cases hard because of EpiBios-to-ARAMRA transfer,
- cases also hard within ARAMRA supervised training.

## Run

Foreground smoke test:

```bash
python next/scripts/run_next_pipeline.py --config next/configs/local_smoke.yaml --foreground
```

Server/background run:

```bash
python next/run.py
```

Equivalent explicit command:

```bash
python next/scripts/run_next_pipeline.py --config next/configs/server_next.yaml
```

The bundled server config expects the standard ARAMRA copy at:

```text
/fs04/ea78/BraTS2021/hemorrhage_hitl/ARAMRA002_standardized
```

If the standardised ARAMRA folder is mounted elsewhere, update `paths.aramra_root` before launching.

Reuse an existing run directory for a foreground restart:

```bash
python next/run.py --foreground --resume next/runs/<run_name>
```

`--resume` reuses the existing run directory and state/log files, but it does not yet skip already completed stages automatically. Treat it as a controlled restart target, not as a full checkpointed scheduler.

Outputs are written under:

```text
next/runs/{timestamp}_next_aramra_selection_oof/
```

## Output Layout

```text
run/
  config/
    config_resolved.yaml
    preflight_report.json
  run_state.json
  metadata/
    metadata_master.csv
    epibios_cases.csv
    aramra_cases.csv
  splits/
    aramra_animalwise_folds.csv
    epibios_source_folds.csv
    split_integrity_report.json
  source_predictions/
    aramra/
      probabilities/
      masks_raw/
  scores/
    source_scores_case.csv
  selection/
    candidate_animal_scores_by_fold.csv
    selected_animals.csv
  checkpoints/
    {method}_budget_{budget}/outer_fold_{k}/source_fold_{j}_best.pt
    aux_aramra_internal/
  predictions/
    aramra_oof/
    epibios_retention/
    aux_aramra_internal_oof/
  metrics/
    source_only_aramra_case_metrics.csv
    aramra_oof_case_metrics.csv
    aramra_oof_summary.csv
    epibios_retention_case_metrics.csv
    epibios_retention_summary.csv
    aux_aramra_internal_oof_case_metrics.csv
    aux_source_internal_difficulty_crosswalk.csv
  logs/
  report.md
```

## Important Integrity Rules

- ARAMRA held-out animals are not used for selection or fine-tuning in the same outer fold.
- Selection features are normalised within the fold-specific candidate pool, not globally across held-out animals.
- Workspace_v0 R1 checkpoints are used as the fine-tuning starting point.
- EpiBios replay excludes the corresponding source held-out fold by default.
- The ARAMRA labels used here are existing labels, not newly revised labels.
