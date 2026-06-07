# workspace_v0 full external analysis run manifest

Generated on 2026-05-23.

## Scope

This analysis intentionally used the archived `workspace_v0/workspace` artifacts:

- round0 checkpoints: `workspace_v0/workspace/artifacts/checkpoints/round_0/fold_*.pt`
- round1 checkpoints: `workspace_v0/workspace/artifacts/checkpoints/round_1/fold_*.pt`
- round0/round1 OOF probabilities and binary labels under `workspace_v0/workspace/artifacts/`

It did not use later SGRA or other attempt checkpoints for ARAMRA prediction.

## Runtime

- Project root: `C:\Users\22396\PycharmProjects\hemorrhage`
- ARAMRA root: `E:\Hemorrhage`
- Python: `C:\Users\22396\PycharmProjects\BraTS_Spacing\.venv\Scripts\python.exe`
- Device used for external prediction: CUDA GPU
- Prediction status: completed for round0 and round1
- ARAMRA prediction counts: 171 probabilities and 171 masks for each round and postprocess variant

## Commands

```powershell
& 'C:\Users\22396\PycharmProjects\BraTS_Spacing\.venv\Scripts\python.exe' analysis\workspace_v0_full_external_analysis\scripts\workspace_v0_full_analysis.py metadata --project-root C:\Users\22396\PycharmProjects\hemorrhage --out-dir C:\Users\22396\PycharmProjects\hemorrhage\analysis\workspace_v0_full_external_analysis --aramra-root E:\Hemorrhage
```

```powershell
& 'C:\Users\22396\PycharmProjects\BraTS_Spacing\.venv\Scripts\python.exe' analysis\workspace_v0_full_external_analysis\scripts\workspace_v0_full_analysis.py predict --project-root C:\Users\22396\PycharmProjects\hemorrhage --out-dir C:\Users\22396\PycharmProjects\hemorrhage\analysis\workspace_v0_full_external_analysis --aramra-root E:\Hemorrhage --device cuda --rounds 0,1
```

```powershell
& 'C:\Users\22396\PycharmProjects\BraTS_Spacing\.venv\Scripts\python.exe' analysis\workspace_v0_full_external_analysis\scripts\workspace_v0_full_analysis.py analyze --project-root C:\Users\22396\PycharmProjects\hemorrhage --out-dir C:\Users\22396\PycharmProjects\hemorrhage\analysis\workspace_v0_full_external_analysis --aramra-root E:\Hemorrhage
```

```powershell
& 'C:\Users\22396\PycharmProjects\BraTS_Spacing\.venv\Scripts\python.exe' analysis\workspace_v0_full_external_analysis\scripts\workspace_v0_full_analysis.py report --project-root C:\Users\22396\PycharmProjects\hemorrhage --out-dir C:\Users\22396\PycharmProjects\hemorrhage\analysis\workspace_v0_full_external_analysis --aramra-root E:\Hemorrhage
```

## Config drift note

The current local `configs/model.yaml` contains later SGRA-related options and uses `postprocessing.min_component_voxels: 2`.

The GitHub original `AndyWanng/HITL` config used:

- `training.finetune.epochs: 10`
- `training.finetune.lr: 2.0e-5`
- `postprocessing.min_component_voxels: 16`

The archived `workspace_v0` round1 training logs show 50 finetune epochs per fold. To make this explicit, the analysis reports raw, `post_min2`, and `post_min16` metrics.

## Primary outputs

- `report.md`
- `metadata/metadata_master.csv`
- `metadata/data_integrity_summary.json`
- `metadata/workspace_v0_training_summary.csv`
- `metadata/epibios_animalwise_group_folds.csv`
- `metadata/aramra_animal_split_proposal.csv`
- `results/summary.json`
- `results/original_fold_animal_leakage_cases.csv`
- `results/original_fold_animal_leakage_by_fold.csv`
- `results/static_reference_case_metrics.csv`
- `results/static_reference_summary.csv`
- `results/aramra_external_case_metrics.csv`
- `results/aramra_external_group_metrics.csv`
- `results/aramra_external_animal_metrics.csv`
- `results/aramra_longitudinal_pair_and_bootstrap_metrics.csv`
- `predictions/aramra/round_0/`
- `predictions/aramra/round_1/`
