# Animal-wise OOF Pipeline

Standalone pipeline for the necessary experiments:

- EpiBios animal-wise round0 OOF training.
- EpiBios animal-wise round1 finetuning with round1 binary labels and new animal-wise R0 OOF soft targets.
- ARAMRA OOD evaluation with the animal-wise R0/R1 fold ensembles.

The runner is intentionally standalone and Python 3.8-compatible; it does not import the main `hemorrhage.*` package at runtime.

ARAMRA is read from a standardized test layout by default:

```text
E:/Hemorrhage/ARAMRA002_standardized/
  imagesTs/{case_id}_0000.nii.gz
  labelsTs/{case_id}.nii.gz
```

The script still has a legacy recursive scanner as a fallback, but the configs point to the standardized copy.

Run foreground smoke test:

```bash
python analysis/animalwise_oof_pipeline/scripts/run_animalwise_oof.py --config analysis/animalwise_oof_pipeline/configs/local_smoke.yaml --foreground
```

Run server job in the background:

```bash
python analysis/animalwise_oof_pipeline/scripts/run_animalwise_oof.py --config analysis/animalwise_oof_pipeline/configs/server_animalwise.yaml
```

All outputs are written under `analysis/animalwise_oof_pipeline/runs/`.
