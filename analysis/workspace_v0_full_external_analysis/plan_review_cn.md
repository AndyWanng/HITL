# 对 ALTA-Hem / animal-aware longitudinal label-trust 方案的复核意见

本报告基于 `analysis/workspace_v0_full_external_analysis/` 中已经生成的 metadata、prediction、metrics、split proposal 和源代码核查结果。工具调用计数按有效子调用计，失败的 PowerShell heredoc、错误环境路径和错误文件名尝试未计入；本轮在最终结论前保守计入超过 250 次有效、有意义的 tool calls。

## 1. 总判断

这份方案的主方向是成立的，而且比继续包装“只做了一轮 HITL 所以 Dice 从约 0.71 到约 0.74”更稳。

最强的可投稿故事不是“多轮 HITL 已经完成”，而是：

1. 旧 5-fold 是 operational scan-level trace，不是 unbiased unseen-animal evaluation。
2. 一轮 expert revise 的主要表观收益很大一部分来自 reference-label shift，而不是 prediction-side generalization。
3. source HITL finetune 几乎没有迁移到 ARAMRA002 external target cohort。
4. ARAMRA002 的 D9/M5 longitudinal structure 暴露了 late-stage / timepoint gap。
5. 因此，30 个 revised labels 更适合作为 clean supervision anchors，而不是作为“完整 HITL loop 已收敛”的证据。

所以，`Animal-aware longitudinal label-trust adaptation` 是合理延申；但目前它仍是 proposal，不是现有代码已经实现的结果。

## 2. 关键本地证据

### 2.1 旧 split 的 animal leakage 是确定事实

`results/original_fold_animal_leakage_by_fold.csv` 显示 5 个 fold 全部是：

- animal overlap rate by cases = 1.0
- animal overlap rate by animals = 1.0
- 合计 127/127 holdout cases 都有同动物 training sibling。

这意味着旧 OOF Dice 可以保留为 operational HITL trace，但不能当作 strict unseen-animal generalization。

### 2.2 static-reference decomposition 支持“reference shift dominates”

raw overall matrix：

| prediction | reference | macro Dice | micro Dice | animal macro Dice |
|---|---:|---:|---:|---:|
| round0 | round0 | 0.690874 | 0.712253 | 0.690990 |
| round0 | round1 | 0.710254 | 0.739072 | 0.709636 |
| round1 | round0 | 0.695285 | 0.715720 | 0.695526 |
| round1 | round1 | 0.713991 | 0.742952 | 0.713524 |

由此得到：

- fixed round1 reference 下，round1 prediction 只比 round0 prediction 高 0.003737 macro Dice。
- fixed round0 reference 下，round1 prediction 只比 round0 prediction 高 0.004411 macro Dice。
- round0 prediction 看到的 reference shift 是 0.019380 macro Dice。
- round1 prediction 看到的 reference shift 是 0.018706 macro Dice。

因此，方案里说“旧 round1 Dice 提升不能直接解释为模型泛化提升”是被数据支持的。

### 2.3 ARAMRA external gain 极小

ARAMRA002 171 labeled cases / 96 strict animals 上：

| model | raw macro Dice | animal macro Dice | micro Dice | median HD95 | mean lesion-F1 |
|---|---:|---:|---:|---:|---:|
| round0 | 0.571844 | 0.573482 | 0.574522 | 19.261 | 0.368210 |
| round1 | 0.573206 | 0.575019 | 0.576217 | 18.570 | 0.374933 |

raw macro Dice 只提升 0.001362。D9 提升约 0.002246，M5 只提升约 0.000118。这个结果非常明确：EpiBios 上的一轮 HITL finetune 没有自动 externalize 到 ARAMRA。

### 2.4 D9/M5 gap 成立，但要表述为 metric-dependent

round1 raw：

- D9 macro Dice = 0.596446。
- M5 macro Dice = 0.540474。
- gap 约 0.056。

但是 lesion-F1 上 M5 均值反而高于 D9：

- D9 mean lesion-F1 = 0.359833。
- M5 mean lesion-F1 = 0.396200。

因此不能写“M5 所有指标都更差”。更准确的说法是：M5 的 overlap Dice 明显更低，且在 D9/M5 paired comparison 中多数动物 M5 worse；但 component-level 指标和 HD95 是 metric-dependent。

### 2.5 D9/M5 paired analysis 支持 longitudinal endpoint

complete pair 上 round1 raw：

- 71 对 D9/M5。
- M5-D9 Dice delta mean = -0.0621，median = -0.0661。
- 57 对 M5 worse，14 对 M5 better。
- GT volume M5-D9 mean = -543 voxels，median = -446 voxels。

这支持把 endpoint 改成 animal-macro、timepoint-balanced 或 D9/M5-balanced，而不是 pooled case macro Dice。

### 2.6 repeat-D9 不能当 clean test-retest reliability

repeat-D9 round1 raw：

- prediction repeat Dice mean = 0.1423。
- label repeat Dice mean = 0.0792。

这个数值太低，不能支持“scan-rescan reliability”表述。它更适合作为 repeated-timepoint / repeated-scan consistency risk 或 irregular stress group。

## 3. proposed ARAMRA split 的问题

当前 split proposal：

| split | animals | cases | pattern |
|---|---:|---:|---|
| adapt pool | 60 | 98 | 34 D9/M5 animals + 21 D9-only + 4 D9/D9 + 1 M5-only |
| validation | 12 | 24 | all D9/M5 |
| locked test | 24 | 49 | 23 D9/M5 + A2838 D9/D9/M5 |

这个设计适合做 paired D9/M5 primary test，但它有一个重要偏差：locked test 和 validation 明显比 adapt pool 更容易。

round1 raw Dice by proposed split：

| split | case macro Dice | animal macro Dice | median HD95 | mean lesion-F1 |
|---|---:|---:|---:|---:|
| adapt pool | 0.536890 | 0.546402 | 23.001 | 0.265876 |
| validation | 0.624464 | 0.624464 | 9.704 | 0.462259 |
| locked test | 0.620731 | 0.621838 | 7.096 | 0.550275 |

这意味着 final AR-Test 如果只用当前 locked split，可能高估最终 target performance。建议二选一：

1. 重切 ARAMRA split，让 locked test 包含一部分 hard / irregular animals。
2. 保留当前 paired locked test，但必须另设 `irregular/hard stress set`，报告 D9-only、D9/D9 repeat、M5-only 和 low-Dice adapt-like cases。

## 4. EpiBios animalwise v1 fold 还不够 publication-grade

已有 `epibios_animalwise_group_folds.csv` 在 positive voxels 上很均衡：

| fold | animals | cases | positive voxels |
|---:|---:|---:|---:|
| 1 | 7 | 27 | 34785 |
| 2 | 7 | 27 | 34058 |
| 3 | 7 | 27 | 33642 |
| 4 | 6 | 22 | 33379 |
| 5 | 6 | 24 | 33796 |

但 family/time/review 分布不均衡。例如 family：

| fold | B4C_Rat | MHR |
|---:|---:|---:|
| 1 | 19 | 8 |
| 2 | 7 | 20 |
| 3 | 3 | 24 |
| 4 | 11 | 11 |
| 5 | 12 | 12 |

因此方案里提出 v2 stratified animal fold 是必要的，不是锦上添花。v1 可以作为 first animal-wise baseline，但 publication primary split 应该优化 family、timepoint、review status 和 lesion burden。

## 5. 代码落地性

当前 repo 已经有：

- residual 3D U-Net backbone；
- case weight / voxel weight；
- round0 / finalize / predict-external；
- revised edit/alignment/SGRA 相关后续补丁；
- external prediction 和本次分析脚本。

但当前代码没有完整实现 proposal 中的 ALTA-Hem：

- 没有 animal-balanced sampler；
- 没有 timepoint-balanced sampler；
- 没有 cohort/timepoint FiLM 或 conditional norm；
- 没有 EMA consistency training；
- 没有 clean/noisy/target label-trust hierarchy；
- 没有 ARAMRA adapt/val/test locked guard；
- 当前 `configs/model.yaml` 是后续 SGRA drift 状态，不能代表 workspace_v0 原始 finetune protocol。

因此下一步不是“解释已有模型就是 ALTA-Hem”，而是新建 experiment layer，明确把 workspace_v0 作为 baseline / evidence foundation。

## 6. 方案中需要降调或修正的 claim

1. 不要说 old 5-fold 是 animal-level OOF。
2. 不要说 round1 HITL 显著提升 external generalization。
3. 不要说 ARAMRA 是 low-field external cohort；当前证据只能写 independent 9.4T target cohort。
4. 不要说 M5 所有指标更差；应该写 overlap Dice 更低，component metrics are metric-dependent。
5. 不要把 repeat-D9 当 scan-rescan reliability。
6. 不要把当前 proposed locked test 当完全代表 ARAMRA hard cases；它偏 complete-pair 且当前 performance 更高。

## 7. 推荐优先级

最高优先级：

1. 生成 EpiBios animalwise v2 stratified folds。
2. 训练 Epi animal-wise round0 / round1 revised-where-available。
3. 重审 ARAMRA split：primary paired locked test + secondary irregular stress set。
4. 做 naive Epi+AR pooled baseline。
5. 做 trust-weighted baseline。

中等优先级：

1. 加 animal-balanced sampler。
2. 加 timepoint-balanced sampler。
3. 加 clean/noisy source/target trust weights。
4. 加 edit-region boost。

靠后优先级：

1. EMA consistency。
2. timepoint conditional FiLM。
3. pair-level temporal loss。

pair-level temporal loss 最要谨慎：没有 registration 和可靠的 biological monotonicity 假设时，不应该强迫 D9/M5 mask voxelwise 相似。它更适合先作为 audit selection / volume-transition auxiliary analysis，而不是主 loss。

## 8. 最终判断

这个 proposal 的研究问题是成立的，甚至比单纯继续做第二轮、第三轮 EpiBios revise 更有论文价值。它最强的地方是：从结果失败中提炼出一个更严肃的问题，即 scan-level HITL gain 在 longitudinal animal MRI 中不等于 animal-level 或 target-cohort generalization。

但要达到较高标准，必须把它从“解释性分析”推进到“动物级 split + target adaptation ablation + locked test / stress test”的完整实验矩阵。仅凭 workspace_v0 的 round0/round1 和 ARAMRA external prediction，足够支撑 motivation，不足以支撑 ALTA-Hem method claim。
