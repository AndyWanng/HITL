# Hemorrhage Segmentation HITL Protocol

**版本**：唯一执行版（供 Codex/工程实现）  
**核心改动**：将每轮 `routine:audit` 比例从 **4:1** 改为 **2:1**  
**适用场景**：人工资源受限；目标是在有限预算下优先修复高价值错误，同时保留足够的长尾监测能力，并基于边际收益递减自动停止。

---

## 1. 目标与原则

本 protocol 不追求 full coverage，也不要求所有 case 都至少被 review 一次。

它的目标是：

1. 固定使用 **5-fold OOF** 结构。
2. 每轮优先处理最有价值的 case（`routine`）。
3. 每轮保留一部分监测样本（`audit`），避免因为纯 top-k 策略误判“已经收敛”。
4. 当 `routine` 集合、`audit` 集合、全局 soft target 都在连续若干个**非空轮**中表现出明显的边际收益递减时，自动停止。

该方案本质上是：

> **utility-driven**，而不是 **coverage-driven**。

---

## 2. 术语映射

原始草案中同时出现了 `exploit/probe` 与 `routine/audit` 两组术语。为避免实现歧义，本版统一采用以下命名：

- `routine` = 原 `exploit`
- `audit` = 原 `probe`

后文仅使用 `routine` / `audit`。

---

## 3. 符号与对象

设数据集共有 $N$ 个 case。对于每个 case $i$：

- 原始影像体数据：$x_i$
- 初始人工标签：$y_i^{(0)} \in \{0,1\}^{\Omega_i}$
- 体素集合：$\Omega_i$

核心状态变量：

- 当前人工标签：$y_i^{(t)}$
- 当前 OOF 概率图：$s_i^{(t)} \in [0,1]^{\Omega_i}$
- 当前 fused soft target：$T_i^{(t)}$
- 当前 fused uncertainty：$H_i^{(t)}$
- review 次数：$r_i$
- 上次被 review 的轮次：$l_i$
- 最早可重新进入候选池的轮次：$e_i$

---

## 4. 固定全局约束

### 4.1 数据与评估约束

1. **不设 locked internal test**。
2. 所有内部 case 的模型预测，**始终使用 OOF prediction**。
3. 对新的外部 case，最终部署预测使用 **5 个 fold 模型的平均 ensemble**。
4. backbone、输入预处理、patch 策略、augmentation 全部固定为当前 baseline；本 protocol 不在这些模块上做搜索或改动。

### 4.2 唯一外部主执行量

只保留一个主执行量：

$$
B = \text{每个非空轮的 nominal 最大 review case 数}
$$

要求：

$$
B \ge 5
$$

实现建议：

- 若希望 `routine:audit = 2:1` 在整数预算上更接近精确比例，建议令 $B$ 为 **3 的倍数**。
- 若希望 audit 在多数轮次中至少能对 5 个 fold 都分到基础配额，建议令 $B \ge 15$；但这不是硬约束。

在第 $t$ 个 nominal round，实际本轮可 review 的 case 数定义为：

$$
B_t = \min(B, |\mathcal{E}_t|)
$$

其中 $\mathcal{E}_t$ 为第 $t$ 轮 eligible pool。

若：

$$
B_t = 0
$$

则该轮为空轮：

- 不做人工 review
- 不做 fine-tune
- 所有状态直接沿用上一轮
- **不计入** early-stop 的连续轮数统计

---

## 5. 每轮预算拆分：`routine:audit = 2:1`

本版直接把每轮预算固定拆成 `routine` 与 `audit` 两部分。

### 5.1 audit 数量

$$
B_{audit,t} = \max\left(1, \operatorname{round}\left(\frac{B_t}{3}\right)\right)
$$

### 5.2 routine 数量

$$
B_{routine,t} = B_t - B_{audit,t}
$$

于是：

$$
|R_t| = B_{routine,t}, \qquad |A_t| = B_{audit,t}
$$

其中：

- $R_t$：routine set
- $A_t$：audit set
- $S_t = R_t \cup A_t$：本轮全部被处理的 case 集合

说明：

- `routine` 是主产出通道，负责优先修复高价值错误。
- `audit` 是监测通道，负责探索长尾池、评估模型辅助是否仍显著改变最终结果，并为停机判据提供证据。

---

## 6. Fold 构造与模型初始化

### 6.1 固定 5-fold 划分

固定使用 **5-fold**。

对每个 case $i$，计算初始标签中的阳性体素数：

$$
V_i = \sum_{v \in \Omega_i} y_i^{(0)}(v)
$$

按 $V_i$ 从大到小排序后，采用蛇形分配（serpentine assignment）将 case 分配到 5 个 folds：

- 第 1 个到 $F_1$
- 第 2 个到 $F_2$
- 第 3 个到 $F_3$
- 第 4 个到 $F_4$
- 第 5 个到 $F_5$
- 第 6 个到 $F_5$
- 第 7 个到 $F_4$
- 第 8 个到 $F_3$
- 第 9 个到 $F_2$
- 第 10 个到 $F_1$

之后重复该来回蛇形模式。

### 6.2 fold-specific 模型

维护 5 个 fold-specific 模型：

$$
M_1, M_2, M_3, M_4, M_5
$$

固定规则：

- 第 $k$ 个模型只在 $D \setminus F_k$ 上训练 / fine-tune。
- 第 $k$ 个模型只对 $F_k$ 产生 OOF prediction。

该 OOF 纪律在整个流程中**绝不改变**。

### 6.3 Round 0 训练

对每个 fold 模型：

- 从头训练
- epoch = **100**
- optimizer = **AdamW**
- learning rate = **1e-4**
- weight decay = **1e-4**
- batch size = **沿用当前 baseline**
- 训练目标 = 初始人工标签 $y^{(0)}$

训练完成后，生成初始 OOF prediction：

$$
s_i^{(0)}(v) \in [0,1]
$$

### 6.4 后续每轮 fine-tune 超参数

对每个 fold 模型：

- 从上一轮模型 warm-start
- epoch = **10**
- optimizer = **AdamW**
- learning rate = **2e-5**
- weight decay = **1e-4**
- batch size = **沿用当前 baseline**

---

## 7. 维护对象与派生量

固定：

$$
\alpha = 0.15
$$

因此人工标签权重恒为 $0.85$。

### 7.1 当前人工标签

对每个 case $i$，维护：

$$
y_i^{(t)} \in \{0,1\}^{\Omega_i}
$$

更新规则：

- 初始为 $y_i^{(0)}$
- 若第 $t$ 轮被 revise，则直接替换为最终 revise 结果
- 若未被 revise，则

$$
y_i^{(t)} = y_i^{(t-1)}
$$

### 7.2 当前 OOF 预测

每轮维护：

$$
s_i^{(t)}(v) \in [0,1]
$$

并严格满足：若 $i \in F_k$，则 $s_i^{(t)}$ 只能来自模型 $M_k^{(t)}$。

### 7.3 fused soft target

最终 soft map 定义为：

$$
T_i^{(t)}(v) = 0.85\, y_i^{(t)}(v) + 0.15\, s_i^{(t)}(v)
$$

再执行裁剪：

$$
T_i^{(t)}(v) \leftarrow \operatorname{clip}\left(T_i^{(t)}(v), 10^{-4}, 1 - 10^{-4}\right)
$$

### 7.4 fused uncertainty map

定义标准化二元熵：

$$
h(p) = \frac{-p \log p - (1-p) \log(1-p)}{\log 2}
$$

由于 $T_i^{(t)} \in [0,0.15] \cup [0.85,1]$，因此 uncertainty map 固定定义为：

$$
H_i^{(t)}(v) = \frac{h\left(T_i^{(t)}(v)\right)}{h(0.15)}
$$

于是：

$$
H_i^{(t)}(v) \in [0,1]
$$

该图仅用于：

1. review 界面的 heatmap
2. 未 review case 的 voxel-level loss 降权
3. 全局过程监测

**不用于 case ranking。**

### 7.5 本轮训练用的临时 soft target

在第 $t$ 轮人工 revise 完成后、但新模型尚未生成前，训练目标固定为：

$$
\tilde{T}_i^{(t)}(v) = 0.85\, y_i^{(t)}(v) + 0.15\, s_i^{(t-1)}(v)
$$

再裁剪：

$$
\tilde{T}_i^{(t)}(v) \leftarrow \operatorname{clip}\left(\tilde{T}_i^{(t)}(v), 10^{-4}, 1 - 10^{-4}\right)
$$

训练用 uncertainty map 为：

$$
H_{i,\mathrm{train}}^{(t)}(v) = \frac{h\left(\tilde{T}_i^{(t)}(v)\right)}{h(0.15)}
$$

### 7.6 模型不确定性：固定 4-TTA variance

每个 case 在每轮都执行固定 4 个 TTA：

1. identity
2. 左右翻转
3. 前后翻转
4. 左右 + 前后翻转

将 4 次预测反变换回原坐标，记为：

$$
s_{i,1}^{(t)},\ s_{i,2}^{(t)},\ s_{i,3}^{(t)},\ s_{i,4}^{(t)}
$$

4-TTA 平均为：

$$
s_i^{(t)} = \frac{1}{4} \sum_{m=1}^{4} s_{i,m}^{(t)}
$$

模型不确定性图定义为：

$$
Q_i^{(t)}(v) = 4 \cdot \operatorname{Var}\left(s_{i,1}^{(t)}(v), \dots, s_{i,4}^{(t)}(v)\right)
$$

因此：

$$
Q_i^{(t)}(v) \in [0,1]
$$

---

## 8. Case priority score：唯一固定定义

case ranking 只使用三路信息：

1. human-model disagreement
2. model uncertainty
3. estimated review cost

**fused uncertainty 不进入 ranking**，避免与 disagreement 双重计权。

### 8.1 top-voxel 聚合规模

对每个 case $i$，定义：

$$
n_i = \max\left(100, \left\lceil 0.01 |\Omega_i| \right\rceil\right)
$$

后续 case-level 聚合全部基于 top-$n_i$ 规则。

### 8.2 case-level human-model disagreement

在第 $t$ 轮选样时，使用上一轮状态：

$$
D_i^{(t)} = \operatorname{mean\ of\ top\ } n_i \operatorname{\ values\ in\ } \left| y_i^{(t-1)} - s_i^{(t-1)} \right|
$$

### 8.3 case-level model uncertainty

$$
U_i^{(t)} = \operatorname{mean\ of\ top\ } n_i \operatorname{\ values\ in\ } Q_i^{(t-1)}
$$

### 8.4 case-level review cost

#### (1) 预测阳性体素比例

$$
P_i^{(t)} = \frac{1}{|\Omega_i|} \sum_{v \in \Omega_i} \mathbf{1}\left[s_i^{(t-1)}(v) > 0.5\right]
$$

#### (2) 预测阳性切片比例

$$
L_i^{(t)} = \frac{\#\{\text{slices containing any voxel with } s_i^{(t-1)} > 0.5\}}{\#\{\text{all slices}\}}
$$

#### (3) 连通域复杂度

定义：

$$
\hat{y}_i^{(t)} = \mathbf{1}\left[s_i^{(t-1)} > 0.5\right]
$$

对 $\hat{y}_i^{(t)}$ 做 3D connected components，记连通域个数为 $CC_{i,\mathrm{raw}}^{(t)}$，归一化为：

$$
CC_i^{(t)} = \frac{\min\left(CC_{i,\mathrm{raw}}^{(t)}, 20\right)}{20}
$$

#### (4) 最终 review cost

$$
C_i^{(t)} = 1 + 2 P_i^{(t)} + L_i^{(t)} + CC_i^{(t)}
$$

### 8.5 base score

在当前 eligible pool 内，对 $D_i^{(t)}$、$U_i^{(t)}$、$C_i^{(t)}$ 分别做 min-max 归一化到 $[0,1]$：

$$
\bar{D}_i^{(t)},\ \bar{U}_i^{(t)},\ \bar{C}_i^{(t)}
$$

若某项 `max = min`，则该项归一化结果全部置为 0。

定义：

$$
\operatorname{Benefit}_i^{(t)} = 0.70\, \bar{D}_i^{(t)} + 0.30\, \bar{U}_i^{(t)}
$$

$$
\operatorname{Score}_i^{(t)} = \frac{\operatorname{Benefit}_i^{(t)}}{1 + \bar{C}_i^{(t)}}
$$

这是每个 case 的 **base score**。

### 8.6 routine 选择用的重复 review 惩罚

由于本版不追求 full coverage，为避免策略长期只咬住同一小撮 hardest cases，routine 选择时对重复 review 的 case 加入轻度惩罚：

$$
\operatorname{Score}_{i,\mathrm{eff}}^{(t)} = \frac{\operatorname{Score}_i^{(t)}}{1 + 0.5\, r_i}
$$

其中 $r_i$ 是 case $i$ 截至当前的累计 review 次数。

解释：

- 第一次 review：不惩罚
- review 次数增加：有效分数自然下降
- 但**不硬性禁止**重复 review

---

## 9. Case 状态与 eligibility

维护以下状态：

- review count：$r_i$，初始为 0
- last review round：$l_i$，初始为 $-1$
- earliest eligible round：$e_i$，初始为 1

在 nominal round $t$ 中，case $i$ eligible 当且仅当：

$$
t \ge e_i
$$

eligible pool 记为：

$$
\mathcal{E}_t = \{ i : t \ge e_i \}
$$

---

## 10. 每轮 case 选择：固定算法

每个非空轮按两步完成样本选择：

1. 先选 `routine set`：$R_t$
2. 再从剩余 eligible case 中选 `audit set`：$A_t$

最终：

$$
S_t = R_t \cup A_t, \qquad |S_t| = B_t
$$

### 10.1 routine 选择

在当前 eligible pool $\mathcal{E}_t$ 内，按以下顺序排序：

1. $\operatorname{Score}_{i,\mathrm{eff}}^{(t)}$ 从高到低
2. $r_i$ 从小到大
3. $l_i$ 从小到大
4. case ID 从小到大

取前 $B_{routine,t}$ 个 case，得到：

$$
R_t
$$

### 10.2 audit 候选池定义

audit 不直接取 top-k，而是优先从**未/少 review** 的剩余 case 中抽取监测样本。

首先定义：

$$
\mathcal{C}_{t}^{audit,(0)} = \{ i \in \mathcal{E}_t \setminus R_t : r_i \le 1 \}
$$

若：

$$
|\mathcal{C}_{t}^{audit,(0)}| \ge B_{audit,t}
$$

则定义 audit 候选池为：

$$
\mathcal{C}_{t}^{audit} = \mathcal{C}_{t}^{audit,(0)}
$$

否则放宽为：

$$
\mathcal{C}_{t}^{audit,(1)} = \{ i \in \mathcal{E}_t \setminus R_t : r_i \le 2 \}
$$

若：

$$
|\mathcal{C}_{t}^{audit,(1)}| \ge B_{audit,t}
$$

则定义 audit 候选池为：

$$
\mathcal{C}_{t}^{audit} = \mathcal{C}_{t}^{audit,(1)}
$$

若仍不足，则 audit 候选池取所有剩余 eligible case：

$$
\mathcal{C}_{t}^{audit} = \mathcal{E}_t \setminus R_t
$$

### 10.3 audit 选择：fold-aware 等间隔抽取

audit 的目标是监测长尾池，而不是再次只追 hardest cases，因此采用 `fold-aware + score-spectrum` 抽取。

#### Step 1：每个 fold 分配基础 audit 配额

定义：

$$
q_{audit,t} = \left\lfloor \frac{B_{audit,t}}{5} \right\rfloor, \qquad u_t = B_{audit,t} - 5 q_{audit,t}
$$

对于每个 fold $F_k$，取：

$$
\mathcal{C}_{t,k}^{audit} = \mathcal{C}_{t}^{audit} \cap F_k
$$

将 $\mathcal{C}_{t,k}^{audit}$ 按 **base score** $\operatorname{Score}_i^{(t)}$ 从高到低排序。若该 fold 中有 $m_k$ 个候选：

- 若 $m_k \le q_{audit,t}$：全部选入
- 若 $m_k > q_{audit,t}$：按等间隔索引选 $q_{audit,t}$ 个

$$
\operatorname{idx}_{k,j} = \left\lceil \frac{j \cdot m_k}{q_{audit,t} + 1} \right\rceil, \qquad j = 1, \dots, q_{audit,t}
$$

这一步的作用：

- 避免 audit 完全偏向某个 fold
- 在每个 fold 内覆盖不同 score 层级，而不是只盯最高分

#### Step 2：全局补足剩余 audit 名额

若经过 Step 1 后仍差 $u_t$ 个，则从 $\mathcal{C}_{t}^{audit}$ 中尚未被选中的候选继续按 base score 排序。设剩余列表长度为 $m_{rem}$，取等间隔索引：

$$
\operatorname{idx}_j = \left\lceil \frac{j \cdot m_{rem}}{u_t + 1} \right\rceil, \qquad j = 1, \dots, u_t
$$

最终得到：

$$
A_t
$$

---

## 11. Human review protocol

### 11.1 routine case 的处理方式

对每个 $i \in R_t$，界面显示：

1. 原始影像 $x_i$
2. 当前人工标签 $y_i^{(t-1)}$ 作为可编辑 mask
3. 当前 OOF 模型概率图 $s_i^{(t-1)}$ 的 0.5 等值轮廓
4. 当前 fused uncertainty map $H_i^{(t-1)}$ 作为可切换 heatmap overlay

医生从当前人工标签出发，完成整例 revise，得到：

$$
y_i^{\mathrm{final}}
$$

然后更新：

$$
y_i^{(t)} = y_i^{\mathrm{final}}
$$

并记录：

- `review_time`
- `edit_ratio`
- `modified_slices_count`

### 11.2 audit case 的处理方式

对每个 $i \in A_t$，执行两阶段 review。

#### Phase 1：anchor-only review

界面只显示：

1. 原始影像 $x_i$
2. 当前人工标签 $y_i^{(t-1)}$ 作为可编辑 mask

**不显示模型预测，不显示 uncertainty heatmap。**

医生完成第一阶段 revise，得到：

$$
y_i^{\mathrm{anchor}}
$$

#### Phase 2：assisted review

随后在同一 case 上继续显示：

1. 原始影像 $x_i$
2. $y_i^{\mathrm{anchor}}$ 作为可编辑 mask
3. 当前 OOF 模型概率图 $s_i^{(t-1)}$ 的 0.5 等值轮廓
4. uncertainty heatmap $H_i^{(t-1)}$

医生可以继续修改，得到最终版本：

$$
y_i^{\mathrm{final}}
$$

然后更新：

$$
y_i^{(t)} = y_i^{\mathrm{final}}
$$

同时保存：

- $y_i^{\mathrm{anchor}}$
- $y_i^{\mathrm{final}}$
- `anchor_time`
- `assisted_time`
- `anchor_assisted_dice`

其中：

$$
\operatorname{anchor\_assisted\_dice}_i^{(t)} = \operatorname{Dice}\left(y_i^{\mathrm{anchor}}, y_i^{\mathrm{final}}\right)
$$

---

## 12. Review 后的状态更新

对所有本轮被处理的 case：

$$
r_i \leftarrow r_i + 1
$$

$$
l_i \leftarrow t
$$

然后更新 `earliest eligible round`。

### 12.1 routine case 的 re-entry

对所有 $i \in R_t$：

$$
e_i \leftarrow t + 2
$$

即：

- routine case 下一轮不可再次被选
- 最早在下下轮重新进入候选池

### 12.2 audit case 的 re-entry

对所有 $i \in A_t$，根据 $\operatorname{Dice}(y_i^{\mathrm{anchor}}, y_i^{\mathrm{final}})$ 决定回池速度：

若

$$
\operatorname{Dice}(y_i^{\mathrm{anchor}}, y_i^{\mathrm{final}}) \ge 0.95
$$

则

$$
e_i \leftarrow t + 3
$$

若

$$
0.90 \le \operatorname{Dice}(y_i^{\mathrm{anchor}}, y_i^{\mathrm{final}}) < 0.95
$$

则

$$
e_i \leftarrow t + 2
$$

若

$$
\operatorname{Dice}(y_i^{\mathrm{anchor}}, y_i^{\mathrm{final}}) < 0.90
$$

则

$$
e_i \leftarrow t + 1
$$

解释：

- 越稳定的 audit case，越慢回池
- 越不稳定的 audit case，越快回池

---

## 13. 本轮训练 target 与 fine-tune

### 13.1 本轮训练 target

第 $t$ 轮 review 完成后，构造：

$$
\tilde{T}_i^{(t)} = 0.85\, y_i^{(t)} + 0.15\, s_i^{(t-1)}
$$

这是本轮 fine-tune 唯一使用的 soft target。

### 13.2 case-level loss 权重

$$
w_{i,\mathrm{case}}^{(t)} =
\begin{cases}
2.0, & i \in S_t \\
1.0, & i \notin S_t
\end{cases}
$$

即：

- 本轮被人工处理过的 case：训练权重翻倍
- 其他 case：正常权重

### 13.3 voxel-level loss 权重

#### 对本轮被 review 的 case

$$
w_{i,\mathrm{voxel}}^{(t)}(v) = 1, \qquad i \in S_t
$$

即：刚修过的 case 不再做 uncertainty 降权。

#### 对本轮未被 review 的 case

$$
w_{i,\mathrm{voxel}}^{(t)}(v) = 1 - 0.5\, H_{i,\mathrm{train}}^{(t)}(v), \qquad i \notin S_t
$$

因此：

$$
w_{i,\mathrm{voxel}}^{(t)}(v) \in [0.5, 1]
$$

### 13.4 Loss 定义

对每个训练 case：

$$
\mathcal{L}_i^{(t)} = w_{i,\mathrm{case}}^{(t)}
\left[
0.5\, \mathcal{L}_{\mathrm{WBCE}}\big(s_i, \tilde{T}_i^{(t)}, w_{i,\mathrm{voxel}}^{(t)}\big)
+ 0.5\, \big(1 - \operatorname{SoftDice}(s_i, \tilde{T}_i^{(t)})\big)
\right]
$$

其中：

- `WBCE` = voxel-weighted binary cross entropy
- `SoftDice` 直接对 soft target $\tilde{T}_i^{(t)}$ 计算

### 13.5 Fine-tune 规则

对每个 fold $k$：

- 从 $M_k^{(t-1)}$ warm-start
- 只在 $D \setminus F_k$ 上 fine-tune
- 使用 $\tilde{T}^{(t)}$ 作为训练 target
- epoch = 10
- lr = 2e-5
- optimizer = AdamW

**严格约束**：

> 绝对不能用 $F_k$ 内任何 case 去训练 $M_k$，无论是 hard label 还是任何 soft target 都不允许。

---

## 14. 重新生成 OOF prediction 与 fused soft target

若第 $t$ 轮为非空轮：

### 14.1 更新 OOF 预测

对每个 fold $k$：

1. 用更新后的 $M_k^{(t)}$ 只预测 $F_k$
2. 使用固定 4-TTA
3. 模型内先做 TTA 平均，得到新的：

$$
s_i^{(t)}
$$

### 14.2 更新 fused soft target 与 uncertainty

$$
T_i^{(t)} = 0.85\, y_i^{(t)} + 0.15\, s_i^{(t)}
$$

再 clip，并计算：

$$
H_i^{(t)}(v) = \frac{h\left(T_i^{(t)}(v)\right)}{h(0.15)}
$$

### 14.3 空轮传播

若第 $t$ 轮为空轮，则直接定义：

$$
y_i^{(t)} = y_i^{(t-1)}, \qquad
s_i^{(t)} = s_i^{(t-1)}, \qquad
T_i^{(t)} = T_i^{(t-1)}, \qquad
H_i^{(t)} = H_i^{(t-1)}
$$

并且空轮**不计入** early-stop 连续轮数。

---

## 15. 停止规则：边际收益递减驱动

本版不再使用 coverage-driven stop。停止逻辑改为：

> 只有当 `routine` 集合、`audit` 集合、全局 fused soft target 都连续若干个非空轮显示边际收益很低时，才停止。

### 15.1 基础统计

#### (1) unique reviewed fraction

$$
\operatorname{Cov}_t = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[r_i > 0]
$$

它只作为 breadth monitor，不再主导 selection。

#### (2) 允许 early stop 的最低 breadth

$$
C_{\min} = \min\left(1, \max\left(\frac{3B}{N}, 0.2\right)\right)
$$

解释：

- 至少积累到相当于 3 个满负荷轮的 unique breadth
- 且至少覆盖 20% 数据
- 两者取更大者

### 15.2 每个非空轮记录的 stop 指标

#### (1) routine 集合的编辑收益

$$
\operatorname{Edit}_t^{routine} = \operatorname{median}_{i \in R_t}
\frac{\lVert y_i^{(t)} - y_i^{(t-1)} \rVert_1}{\lVert y_i^{(t)} \lor y_i^{(t-1)} \rVert_1 + 10^{-6}}
$$

#### (2) audit 集合的编辑收益

$$
\operatorname{Edit}_t^{audit} = \operatorname{median}_{i \in A_t}
\frac{\lVert y_i^{(t)} - y_i^{(t-1)} \rVert_1}{\lVert y_i^{(t)} \lor y_i^{(t-1)} \rVert_1 + 10^{-6}}
$$

#### (3) 全局 fused soft target 变化

$$
\Delta T_t = \frac{1}{N} \sum_{i=1}^{N} \operatorname{MAE}\left(T_i^{(t)}, T_i^{(t-1)}\right)
$$

#### (4) audit 稳定性

$$
\operatorname{Stab}_t^{audit} = \operatorname{median}_{i \in A_t} \operatorname{Dice}\left(y_i^{\mathrm{anchor}}, y_i^{\mathrm{final}}\right)
$$

### 15.3 固定 stop 阈值

固定阈值如下：

$$
\tau_{routine} = 0.03
$$

$$
\tau_{audit} = 0.015
$$

$$
\tau_{\Delta} = 0.005
$$

$$
\tau_A = 0.97
$$

patience 固定为：

$$
p = 3
$$

### 15.4 唯一停止条件

只有在满足以下全部条件时才停止：

#### 条件 1：至少已经完成 3 个非空轮

#### 条件 2：已经达到最低 breadth

$$
\operatorname{Cov}_t \ge C_{\min}
$$

#### 条件 3：连续 $p = 3$ 个非空轮同时满足

$$
\operatorname{Edit}_t^{routine} < \tau_{routine}
$$

$$
\operatorname{Edit}_t^{audit} < \tau_{audit}
$$

$$
\Delta T_t < \tau_{\Delta}
$$

$$
\operatorname{Stab}_t^{audit} > \tau_A
$$

一旦这 4 个条件在连续 3 个非空轮中全部成立，则 protocol 停止。

---

## 16. 每轮固定记录的过程指标

除 stop 指标外，每个非空轮还需记录以下过程指标。

### 16.1 mean fused uncertainty

$$
E_t = \frac{1}{\sum_i |\Omega_i|} \sum_i \sum_{v \in \Omega_i} H_i^{(t)}(v)
$$

### 16.2 high-uncertainty fraction

$$
HU_t = \frac{\#\{(i,v): H_i^{(t)}(v) > 0.5\}}{\sum_i |\Omega_i|}
$$

### 16.3 model-to-final Dice on routine

$$
D_t^{mf,routine} = \operatorname{median}_{i \in R_t} \operatorname{Dice}\left(\mathbf{1}[s_i^{(t-1)} > 0.5], y_i^{(t)}\right)
$$

### 16.4 model-to-final Dice on audit

$$
D_t^{mf,audit} = \operatorname{median}_{i \in A_t} \operatorname{Dice}\left(\mathbf{1}[s_i^{(t-1)} > 0.5], y_i^{(t)}\right)
$$

### 16.5 review time

记录：

- routine review 的中位数时间
- audit `anchor-only` 的中位数时间
- audit `assisted` 的中位数时间

### 16.6 unique reviewed fraction

$$
\operatorname{Cov}_t = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[r_i > 0]
$$

虽然它不再是停止条件主体，但仍建议每轮可视化并持续跟踪。

---

## 17. 最终输出

protocol 结束后，输出以下三类结果。

### 17.1 最终标签集

$$
\{ y_i^{(R)} \}_{i=1}^{N}
$$

### 17.2 最终 fused soft target 集

$$
\{ T_i^{(R)} \}_{i=1}^{N}
$$

### 17.3 最终模型

$$
M_1^{(R)}, M_2^{(R)}, M_3^{(R)}, M_4^{(R)}, M_5^{(R)}
$$

对新的外部 case，最终预测固定为：

1. 5 个模型各推理一次
2. 每个模型做 4-TTA
3. 模型内先做 TTA 平均
4. 再对 5 个模型输出取平均

得到最终外部概率图。

---

## 18. 单轮迭代的规范流程

若第 $t$ 轮为非空轮，则必须按以下 8 步执行：

1. **计算评分**：基于上一轮状态，为所有 eligible case 计算 `base score` 与 `effective score`。
2. **选择 routine set**：按 $\operatorname{Score}_{i,\mathrm{eff}}^{(t)}$ 选出前 $B_{routine,t}$ 个 case。
3. **选择 audit set**：从剩余 eligible case 中，优先从少 review 池出发，按 `fold-aware + 等间隔` 规则抽取 $B_{audit,t}$ 个 case。
4. **人工 review**：
   - routine：assisted revise
   - audit：anchor-only + assisted 两阶段 revise
5. **更新标签与状态**：更新 $y_i^{(t)}$、$r_i$、$l_i$、$e_i$。
6. **构造训练 target**：
   $$
   \tilde{T}^{(t)} = 0.85\, y^{(t)} + 0.15\, s^{(t-1)}
   $$
7. **fine-tune 5 个 fold-specific 模型**：保持严格 OOF 训练纪律。
8. **重生成 OOF / soft target / uncertainty**：更新 $s^{(t)}$、$T^{(t)}$、$H^{(t)}$，记录 stop 指标与过程指标，并检查是否触发停止。

---

## 19. 实现约束清单（必须满足）

以下约束对工程实现是强制性的：

1. **任何 fold 模型都不能在自己的 held-out fold 上训练**，包括 hard label、soft target、任何衍生标签。
2. **第 $t$ 轮训练 target 必须使用 $s^{(t-1)}$，不能偷用 $s^{(t)}$。**
3. **内部预测必须始终是 OOF prediction。**
4. **空轮不参与 patience 统计。**
5. **audit 的第一阶段必须对模型输出与 uncertainty 完全盲。**
6. **fused uncertainty 不参与 case ranking。**
7. **routine 的重复 review 惩罚只能是软惩罚，不能把多次 review 的 case 永久踢出池子。**
8. **所有 min-max 归一化必须仅在当前 eligible pool 内进行。**
9. **audit 抽样必须先做 per-fold 基础配额，再做全局补足；不能退化为简单 top-k。**
10. **early-stop 的连续计数单位是非空轮，不是 nominal round。**

---

## 20. 面向实现的伪代码

```python
# Round 0
initialize y[i] = y0[i] for all i
initialize r[i] = 0, l[i] = -1, e[i] = 1 for all i
train M_k^(0) from scratch on D \ F_k for k in {1..5}
generate OOF predictions s^(0), 4-TTA variance Q^(0)
compute T^(0), H^(0)

# Subsequent rounds
non_empty_round_counter = 0
stop_window = []   # keep only non-empty rounds

t = 1
while True:
    E_t = {i for i in D if t >= e[i]}
    B_t = min(B, len(E_t))

    if B_t == 0:
        y^(t) = y^(t-1)
        s^(t) = s^(t-1)
        T^(t) = T^(t-1)
        H^(t) = H^(t-1)
        t += 1
        continue

    non_empty_round_counter += 1

    # budget split: routine : audit = 2 : 1
    B_audit = max(1, round(B_t / 3))
    B_routine = B_t - B_audit

    # compute ranking features from state at t-1
    for i in E_t:
        compute D_i^(t), U_i^(t), C_i^(t)
    minmax-normalize D, U, C within E_t
    Score_i = (0.70 * Dbar_i + 0.30 * Ubar_i) / (1 + Cbar_i)
    Score_eff_i = Score_i / (1 + 0.5 * r[i])

    # select routine set
    R_t = top_k(
        E_t,
        k=B_routine,
        key=(Score_eff desc, r asc, l asc, case_id asc)
    )

    # build audit candidate pool
    A0 = {i for i in E_t - R_t if r[i] <= 1}
    if len(A0) >= B_audit:
        A_pool = A0
    else:
        A1 = {i for i in E_t - R_t if r[i] <= 2}
        if len(A1) >= B_audit:
            A_pool = A1
        else:
            A_pool = E_t - R_t

    # select audit set by fold-aware equal-spacing
    A_t = fold_aware_equal_spacing(A_pool, B_audit, base_score=Score)

    # human review
    # routine: assisted revise
    # audit: anchor-only -> assisted revise
    update y^(t) on reviewed cases
    keep y^(t) = y^(t-1) on unreviewed cases

    # update states
    for i in R_t ∪ A_t:
        r[i] += 1
        l[i] = t
    for i in R_t:
        e[i] = t + 2
    for i in A_t:
        d = Dice(y_anchor[i], y_final[i])
        if d >= 0.95:
            e[i] = t + 3
        elif d >= 0.90:
            e[i] = t + 2
        else:
            e[i] = t + 1

    # build training target from y^(t) and previous OOF prediction s^(t-1)
    T_tilde^(t) = 0.85 * y^(t) + 0.15 * s^(t-1)

    # fine-tune fold-specific models under strict OOF discipline
    for k in {1..5}:
        fine_tune M_k^(t-1) -> M_k^(t) on D \ F_k using T_tilde^(t)

    # regenerate OOF predictions and derived maps
    generate s^(t), Q^(t) using fixed 4-TTA
    compute T^(t), H^(t)

    # log metrics and stop indicators
    compute Cov_t, Edit_t^routine, Edit_t^audit, ΔT_t, Stab_t^audit, etc.
    update stop_window with this non-empty round only

    if non_empty_round_counter >= 3 and Cov_t >= C_min:
        if last 3 non-empty rounds all satisfy:
            Edit^routine < 0.03
            Edit^audit < 0.015
            ΔT < 0.005
            Stab^audit > 0.97
        then:
            break

    t += 1
```

---

## 21. 与原始版本相比的唯一实质改动

相较于你之前的版本，本版的唯一实质改动是：

### 原设置

- `routine:audit = 4:1`
- 即近似 `80% : 20%`

### 新设置

- `routine:audit = 2:1`
- 即近似 `66.7% : 33.3%`

对应实现层面的变化只有两类：

1. **预算拆分公式改变**
   - 原先：`B_audit ≈ round(0.2 * B_t)`
   - 现在：`B_audit = max(1, round(B_t / 3))`
2. **停机证据来源更强**
   - 因为 audit 占比提高，长尾监测密度增加
   - 但 routine 主产出预算相应下降
   - 所以该版本更偏向“谨慎停机”，而不是“极致追求短期编辑产出”

---

## 22. 一句话总结

这套方案的本质是：

> 固定 5-fold 与严格 OOF discipline；每轮把预算拆成 **2/3 的 routine** 与 **1/3 的 audit**，前者按带重复惩罚的高分优先，后者从未/少 review 的剩余池中做 fold-aware 等间隔监测抽取；review 后用“新标签 + 旧 OOF 预测”构造 soft target 对 5 个模型做 fine-tune，再重新生成 OOF prediction、fused soft target 和 uncertainty；当 routine 编辑收益、audit 编辑收益、全局 fused target 变化在连续 3 个非空轮都很低，且 audit 稳定性很高时停止。
