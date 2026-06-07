# workspace_v0 full external analysis report

Generated: 2026-05-23 23:43:51

## Scope

This analysis uses the archived `workspace_v0` round0 and round1 5-fold models only. It does not use later correction attempts or the current SGRA revision-policy branch for prediction.

## workspace_v0 training audit

| round_index | fold_id | status_epochs_total | last_logged_epoch | best_epoch | best_metric_value | num_train_cases_after_val_split | num_val_cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 200 | 200 | 129 | 0.667583 | 89 | 13 |
| 0 | 2 | 200 | 200 | 140 | 0.710306 | 89 | 13 |
| 0 | 3 | 200 | 200 | 118 | 0.734694 | 89 | 13 |
| 0 | 4 | 200 | 200 | 149 | 0.614351 | 88 | 13 |
| 0 | 5 | 200 | 200 | 167 | 0.694965 | 88 | 13 |
| 1 | 1 | 50 | 50 | 37 | 0.673745 | 89 | 13 |
| 1 | 2 | 50 | 50 | 44 | 0.751736 | 89 | 13 |
| 1 | 3 | 50 | 50 | 31 | 0.750302 | 89 | 13 |
| 1 | 4 | 50 | 50 | 14 | 0.657510 | 88 | 13 |
| 1 | 5 | 50 | 50 | 44 | 0.713893 | 88 | 13 |

The current local `configs/model.yaml` has later SGRA-related changes and `min_component_voxels=2`; the original GitHub config used `min_component_voxels=16`. To avoid config drift, external metrics are reported for raw, `post_min2`, and `post_min16` masks.

## Data inventory

- EpiBios: 127 cases, 33 animals, 30 reviewed cases (20 routine, 10 audit).
- ARAMRA002: 171 labeled cases, 171 evaluable image-label cases, 96 strict animals.
- ARAMRA unmatched labeled cases: 0. Unmatched images: 1.
- EpiBios field strength: per-case field strength is not encoded in local filenames or workspace_v0 metadata; kept as unknown_epibios_mixed.

## Original fold animal leakage

| fold_id | holdout_cases | holdout_animals | holdout_cases_with_train_sibling | holdout_animals_with_train_sibling | animal_overlap_rate_cases | animal_overlap_rate_animals |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 | 20 | 25 | 20 | 1.000000 | 1.000000 |
| 2 | 25 | 20 | 25 | 20 | 1.000000 | 1.000000 |
| 3 | 25 | 19 | 25 | 19 | 1.000000 | 1.000000 |
| 4 | 26 | 19 | 26 | 19 | 1.000000 | 1.000000 |
| 5 | 26 | 21 | 26 | 21 | 1.000000 | 1.000000 |

Overall, 127/127 holdout cases have same-animal training siblings; case overlap rate = 1.000000.

## Static-reference matrix

| prediction_round | reference_round | postprocess_variant | num_cases | macro_dice | micro_dice | animal_macro_dice | median_hd95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | post_min16 | 127 | 0.653782 | 0.704509 | 0.654636 | 4.242641 |
| 0 | 0 | post_min2 | 127 | 0.690382 | 0.712167 | 0.690490 | 2.582118 |
| 0 | 0 | raw | 127 | 0.690874 | 0.712253 | 0.690990 | 2.638958 |
| 0 | 1 | post_min16 | 127 | 0.672209 | 0.730515 | 0.672365 | 3.741657 |
| 0 | 1 | post_min2 | 127 | 0.709754 | 0.738970 | 0.709127 | 2.236068 |
| 0 | 1 | raw | 127 | 0.710254 | 0.739072 | 0.709636 | 2.000000 |
| 1 | 0 | post_min16 | 127 | 0.664875 | 0.709336 | 0.665632 | 4.123106 |
| 1 | 0 | post_min2 | 127 | 0.695279 | 0.715796 | 0.695495 | 2.468437 |
| 1 | 0 | raw | 127 | 0.695285 | 0.715720 | 0.695526 | 2.449490 |
| 1 | 1 | post_min16 | 127 | 0.682661 | 0.735856 | 0.682744 | 3.534826 |
| 1 | 1 | post_min2 | 127 | 0.713974 | 0.743017 | 0.713482 | 2.236068 |
| 1 | 1 | raw | 127 | 0.713991 | 0.742952 | 0.713524 | 2.236068 |

Interpretation: compare round0 prediction vs round1 reference against round1 prediction vs round1 reference to separate label-reference shift from finetune model gain.

## ARAMRA002 external evaluation

| prediction_round | postprocess_variant | num_cases | num_animals | macro_dice | animal_macro_dice | micro_dice | median_hd95 | mean_lesion_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | post_min16 | 171 | 96 | 0.564590 | 0.566042 | 0.567542 | 20.989280 | 0.279851 |
| 0 | post_min2 | 171 | 96 | 0.571577 | 0.573210 | 0.574252 | 20.099751 | 0.365822 |
| 0 | raw | 171 | 96 | 0.571844 | 0.573482 | 0.574522 | 19.261360 | 0.368210 |
| 1 | post_min16 | 171 | 96 | 0.565611 | 0.567240 | 0.569047 | 21.095023 | 0.282486 |
| 1 | post_min2 | 171 | 96 | 0.572937 | 0.574759 | 0.575938 | 18.688199 | 0.372221 |
| 1 | raw | 171 | 96 | 0.573206 | 0.575019 | 0.576217 | 18.570129 | 0.374933 |

### Timepoint split

| prediction_round | postprocess_variant | time_raw | num_cases | num_animals | macro_dice | animal_macro_dice | micro_dice | median_hd95 | mean_lesion_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | post_min16 | D9 | 100 | 95 | 0.587257 | 0.588731 | 0.590246 | 21.372196 | 0.249076 |
| 0 | post_min16 | M5 | 71 | 71 | 0.532664 | 0.532664 | 0.523989 | 20.598541 | 0.323194 |
| 0 | post_min2 | D9 | 100 | 95 | 0.593952 | 0.595523 | 0.596537 | 20.563292 | 0.343966 |
| 0 | post_min2 | M5 | 71 | 71 | 0.540062 | 0.540062 | 0.531556 | 18.725366 | 0.396605 |
| 0 | raw | D9 | 100 | 95 | 0.594200 | 0.595759 | 0.596785 | 19.970423 | 0.354118 |
| 0 | raw | M5 | 71 | 71 | 0.540356 | 0.540356 | 0.531886 | 18.697589 | 0.388058 |
| 1 | post_min16 | D9 | 100 | 95 | 0.589271 | 0.590825 | 0.592526 | 21.098116 | 0.250108 |
| 1 | post_min16 | M5 | 71 | 71 | 0.532286 | 0.532286 | 0.523629 | 21.095023 | 0.328090 |
| 1 | post_min2 | D9 | 100 | 95 | 0.596211 | 0.597847 | 0.598997 | 18.768793 | 0.352805 |
| 1 | post_min2 | M5 | 71 | 71 | 0.540157 | 0.540157 | 0.531398 | 17.233688 | 0.399568 |
| 1 | raw | D9 | 100 | 95 | 0.596446 | 0.598058 | 0.599245 | 18.760113 | 0.359833 |
| 1 | raw | M5 | 71 | 71 | 0.540474 | 0.540474 | 0.531750 | 16.911535 | 0.396200 |

## Review distribution

| revised_status | animal_family | time_raw | original_fold | num_cases | num_animals |
| --- | --- | --- | --- | --- | --- |
| audit |  |  |  | 10 | 7 |
| none |  |  |  | 97 | 31 |
| routine |  |  |  | 20 | 9 |
| audit | B4C_Rat |  |  | 3 | 2 |
| audit | MHR |  |  | 7 | 5 |
| none | B4C_Rat |  |  | 46 | 14 |
| none | MHR |  |  | 51 | 17 |
| routine | B4C_Rat |  |  | 3 | 2 |
| routine | MHR |  |  | 17 | 7 |
| audit |  | D02 |  | 3 | 3 |
| audit |  | D28 |  | 2 | 2 |
| audit |  | M05 |  | 2 | 2 |
| audit |  | S1 |  | 1 | 1 |
| audit |  | S2 |  | 1 | 1 |
| audit |  | S4 |  | 1 | 1 |
| none |  | D02 |  | 13 | 13 |
| none |  | D09 |  | 14 | 14 |
| none |  | D28 |  | 9 | 9 |
| none |  | M01 |  | 5 | 5 |
| none |  | M05 |  | 12 | 12 |
| none |  | S1 |  | 10 | 10 |
| none |  | S2 |  | 9 | 9 |
| none |  | S3 |  | 11 | 11 |
| none |  | S4 |  | 7 | 7 |
| none |  | W04 |  | 2 | 2 |
| none |  | W20 |  | 5 | 5 |
| routine |  | D02 |  | 5 | 5 |
| routine |  | D09 |  | 5 | 5 |
| routine |  | D28 |  | 2 | 2 |
| routine |  | M01 |  | 1 | 1 |
| routine |  | M05 |  | 4 | 4 |
| routine |  | S2 |  | 2 | 2 |
| routine |  | S3 |  | 1 | 1 |
| audit |  |  | 1 | 2 | 2 |
| audit |  |  | 2 | 2 | 2 |
| audit |  |  | 3 | 2 | 2 |
| audit |  |  | 4 | 2 | 2 |
| audit |  |  | 5 | 2 | 2 |
| none |  |  | 1 | 19 | 16 |
| none |  |  | 2 | 19 | 17 |
| none |  |  | 3 | 20 | 16 |
| none |  |  | 4 | 19 | 15 |
| none |  |  | 5 | 20 | 17 |
| routine |  |  | 1 | 4 | 3 |
| routine |  |  | 2 | 4 | 3 |
| routine |  |  | 3 | 3 | 2 |
| routine |  |  | 4 | 5 | 4 |
| routine |  |  | 5 | 4 | 4 |

## Output files

- `metadata/metadata_master.csv`: unified EpiBios + ARAMRA case table.
- `metadata/workspace_v0_training_summary.csv`: exact logged epoch counts and selected validation checkpoints for archived round0/round1 folds.
- `results/original_fold_animal_leakage_cases.csv` and `results/original_fold_animal_leakage_by_fold.csv`: leakage audit.
- `results/static_reference_case_metrics.csv` and `results/static_reference_summary.csv`: 2x2 prediction/reference matrix.
- `predictions/aramra/round_*/`: external probabilities and masks.
- `results/aramra_external_case_metrics.csv`, `results/aramra_external_group_metrics.csv`, `results/aramra_external_animal_metrics.csv`: external metrics.
- `results/aramra_longitudinal_pair_and_bootstrap_metrics.csv`: D9/M5 pair deltas, repeat-D9 consistency, and animal bootstrap intervals.

## Caveats

- ARAMRA002 is an independent 9.4T target cohort, not a low-field external cohort.
- EpiBios per-case field strength is not recoverable from the local filenames or workspace metadata, so field-strength claims require an external protocol table.
- This report runs existing workspace_v0 model prediction and analysis. Animal-wise retraining is a separate experiment and should use the generated `metadata/epibios_animalwise_group_folds.csv` split table.
