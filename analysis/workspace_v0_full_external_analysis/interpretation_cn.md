# workspace_v0 外部预测与分析中文解读

生成日期：2026-05-23

## 结论摘要

这次分析完整使用 `workspace_v0` 的 round0 / round1 五折模型，对 `E:\Hemorrhage` 中 171 例可评估 ARAMRA002 labeled case 做了 GPU 外部预测，并同时回看 EpiBios 127 例的原始 OOF split、static-reference matrix 和纵向结构。

核心结论比较明确：

1. 原 5-fold OOF 是严重 case-level / session-level OOF，不是 unseen-animal OOF。127/127 个 holdout case 都有同一动物的其他 timepoint 出现在训练 fold 中。
2. EpiBios 内部 0.71 到 0.74 的提升不能直接解释成强泛化提升。raw mask 下，round0 prediction vs round0 reference 的 macro Dice 是 0.6909；round1 prediction vs round1 reference 是 0.7140。但如果固定 round1 reference，round1 相对 round0 只提升 0.0037 macro Dice；更大的差异来自 round0 reference 到 round1 reference 的 label-reference shift，约 0.019。
3. ARAMRA002 外部 9.4T cohort 上，round1 相对 round0 的提升非常小。raw mask 下 macro Dice 从 0.5718 到 0.5732，只增加 0.0014；micro Dice 从 0.5745 到 0.5762，只增加 0.0017。
4. ARAMRA 的 D9 明显好于 M5。round1 raw 下，D9 macro Dice 是 0.5964，M5 是 0.5405。M5 基本没有从 round1 finetune 中受益。
5. 因此，当前最稳的论文叙事不是“完成了多轮 HITL 并收敛”，而是“原始 case-level HITL gain 与 animal-level / external-cohort generalization 之间存在 gap；需要 animal-aware、domain-aware、longitudinal-aware 的评估和训练框架”。

## 数据归类

### EpiBios

- 127 labeled cases
- 33 只动物
- 30 例经过 round1 review，其中 routine 20 例、audit 10 例
- 本地文件名和 `workspace_v0` metadata 不能恢复逐例 field strength，所以本次只保留为 `unknown_epibios_mixed`
- 所有图像和标签 shape/spacing 匹配：`160 x 122 x 80`，`1 x 1 x 1 mm`

### ARAMRA002

- 172 个 image/label row
- 171 个 labeled 且 image-label 可配对的 case
- 96 个 strict animal
- 1 个 unmatched image：`20201210_083104_ARAMRA002_2945_9D_I_1_1_1_10_MGE`
- labeled case 中 D9 100 例，M5 71 例
- animal-level pattern：
  - `D9 / M5`: 69 animals
  - `D9 only`: 21 animals
  - `D9 / D9`: 4 animals
  - `D9 / D9 / M5`: 1 animal
  - `M5 only`: 1 animal

## 原 fold animal leakage

原来的 5 fold 中，每个 fold 的 holdout case 都存在同动物 train sibling：

- fold 1: 25/25 holdout cases
- fold 2: 25/25 holdout cases
- fold 3: 25/25 holdout cases
- fold 4: 26/26 holdout cases
- fold 5: 26/26 holdout cases

总体是 127/127，case overlap rate = 1.0。这个结果不能说明旧实验没价值，但它把旧 OOF 的解释范围限定为 operational case-level HITL metric，而不是 unseen-animal generalization metric。

## Static-reference matrix 的关键解释

raw mask 总体矩阵：

| Prediction | Reference | Macro Dice | Micro Dice |
|---|---:|---:|---:|
| round0 | round0 | 0.6909 | 0.7123 |
| round0 | round1 | 0.7103 | 0.7391 |
| round1 | round0 | 0.6953 | 0.7157 |
| round1 | round1 | 0.7140 | 0.7430 |

拆开看：

- 固定 round1 reference 时，round1 prediction 比 round0 prediction 只高 0.0037 macro Dice。
- 固定 round0 reference 时，round1 prediction 比 round0 prediction 只高 0.0044 macro Dice。
- round0 prediction 从 round0 reference 换到 round1 reference，macro Dice 增加 0.0194。
- round1 prediction 从 round0 reference 换到 round1 reference，macro Dice 增加 0.0187。

所以内部 0.71 到 0.74 的主要解释不是“模型本身大幅变强”，而是 reference label 经过 revise 后与模型预测更一致。这个判断和之前“提升主要体现在 revised labels 部分”的观察一致。

按 reviewed status 看，固定 round1 reference：

- routine：round0 raw 0.7392，round1 raw 0.7491，增加 0.0099
- audit：round0 raw 0.7536，round1 raw 0.7246，下降 0.0290
- none：round0 raw 0.6998，round1 raw 0.7057，增加 0.0058

audit subset 下降这一点需要谨慎解释。它可能反映 audit case 更难、review 后 reference 更不贴近模型，也可能说明 round1 finetune 没有稳定改善最具争议的样本。

## ARAMRA002 外部结果

raw mask 总体：

| Round | Cases | Animals | Macro Dice | Animal Macro Dice | Micro Dice | Median HD95 | Mean Lesion-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 171 | 96 | 0.5718 | 0.5735 | 0.5745 | 19.2614 | 0.3682 |
| 1 | 171 | 96 | 0.5732 | 0.5750 | 0.5762 | 18.5701 | 0.3749 |

round1 外部提升：

- macro Dice：+0.0014
- animal macro Dice：+0.0015
- micro Dice：+0.0017

这个提升非常小，不足以支撑“round1 finetune 显著改善 independent cohort generalization”的 claim。

### D9 vs M5

raw mask：

| Round | Timepoint | Cases | Animals | Macro Dice | Animal Macro Dice | Micro Dice | Median HD95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | D9 | 100 | 95 | 0.5942 | 0.5958 | 0.5968 | 19.9704 |
| 0 | M5 | 71 | 71 | 0.5404 | 0.5404 | 0.5319 | 18.6976 |
| 1 | D9 | 100 | 95 | 0.5964 | 0.5981 | 0.5992 | 18.7601 |
| 1 | M5 | 71 | 71 | 0.5405 | 0.5405 | 0.5317 | 16.9115 |

D9 比 M5 高约 0.056 Dice。round1 对 D9 有轻微提升，对 M5 几乎没有提升。这提示 timepoint / lesion-stage / late-stage appearance shift 是后续论文中值得单独处理的因素。

### D9/M5 paired analysis

完整 D9/M5 pair 组合共 71 个 pair；每个 round 和 postprocess variant 都计算了 pair delta。

raw mask 下：

- round0：M5 Dice 相对 D9 平均低 0.0602，中位低 0.0656
- round1：M5 Dice 相对 D9 平均低 0.0621，中位低 0.0661
- GT volume 的 M5-D9 平均差为 -543 voxels，中位差为 -446 voxels

这说明 ARAMRA 的 M5 不只是模型性能差，标签中的 lesion burden 也通常更小，可能导致 late-stage / small-residual-lesion segmentation 更难。

### Repeat-D9 consistency

repeat-D9 pair 共 5 个 raw-mask pair。round1 raw：

- prediction repeat Dice mean: 0.1423
- prediction repeat Dice median: 0.0865
- label repeat Dice mean: 0.0792
- label repeat Dice median: 0.0509

这个数值很低，说明 repeated D9 并不是简单 scan-rescan 同质重复，更可能包含定位、病灶变化、标签边界和采集差异等因素；后续不能把它当作纯 test-retest reliability，而应作为“重复时间点/重复扫描一致性风险”来分析。

## 方法学判断

这轮结果把论文方向收窄得比较清楚：

1. 旧 HITL workflow 可以作为 operational trace 和 clean-anchor 证据保留。
2. 但不能把旧 5-fold OOF 当作 unbiased animal-level performance。
3. ARAMRA002 是真正有价值的 independent 9.4T target cohort，应该继续保留 locked external / validation / adaptation 的 animal-level 切分，不建议直接全部混进训练后只报 298-case OOF。
4. 下一步最有价值的模型实验是 EpiBios animal-wise CV 和 trust-weighted clean/noisy training，而不是简单重复一轮相同的 EpiBios revise。
5. 如果还能做额外人工 review，优先考虑 ARAMRA paired D9/M5 audit，而不是在原 127 例中继续随机 revise。

## 本次没有完成的部分

本次已完成 `analysis_1` 中最关键的 data audit、original split leakage、static-reference matrix、workspace_v0 round0/round1 ARAMRA external prediction、timepoint/animal-level/longitudinal analysis。

本次没有重新训练新的 animal-wise EpiBios baseline，也没有训练 Epi+AR trust-weighted 模型。原因不是简化分析，而是这些是新的训练实验，计算和实验设计应以本次产出的 `metadata/epibios_animalwise_group_folds.csv` 和 `metadata/aramra_animal_split_proposal.csv` 为基础单独启动，避免把“现有 workspace_v0 模型评估”和“新方法模型训练”混在同一个结果里。

