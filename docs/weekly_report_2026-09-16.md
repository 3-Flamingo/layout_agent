# 2D 图纸布局优化 · 周报

**周期：** 2026-09-09 ～ 2026-09-16  
**范围：** Decide-only 仿真环境搭建；公开工程图纸数据集筛选与验证  
**主仓库：** `E:\workspace\layout_agent`

---

## 本周结论

仿真环境已经能跑通「采集 → 行为克隆 → 掩码 PPO」的训练器链路，但**还没有可发版策略**。公开数据里，NIST 与 ECCV TechDraw **不符合** Decide-only 布局需求；CMU 两张带 `DIMENSION` 的种子 DXF 可以映射成 `LayoutState`，仅适合作为 train-only 补充，尚未入库。

人工 gold 协议已冻结（80 对 T2 + 15 条 T1，seed 17），状态为 `tasks_frozen; awaiting human labels`。人类奖励系数 **β = 0**，在 80/15 门禁通过前不会打开。同事挑战 100 对与 gold `view_key` **不进训练、不选 checkpoint**。

---

## 1. 仿真环境搭建

设计约束不变：策略只替换 `DecisionEngine`。规则 `Evaluator`、`Executor`、确定性 `Schedule` 继续作为安全层。训练时 LLM 关闭。

### 1.1 `LayoutEnv`

实现位置：`src/layout_agent/training/env.py`。

- Gymnasium 接口：`reset → (obs, info)`，`step → (obs, reward, terminated, truncated, info)`。
- 观测：3 通道 ICM、全局特征、候选动作特征（padding）、动作 mask。
- **`legal_bbox` 在 reset 时冻结**，用于 `H_OOB`；Executor 不得改写。
- **`render_bbox` 仅服务栅格化**：尺寸挪出原框后可以扩大，好画 ICM；评估约束不读它。
- 回放校验：`runs/rl_env_validation.json` 记录 2 个 episode / 19 步，`deterministic: true`，`mismatches: []`。

动作空间（合法候选，去重、去掉已尝试与几何 no-op）：

| 动作 | 作用 |
|---|---|
| `ACCEPT` | 结束 episode |
| `SET_LINEAR_SIDE` | 线性尺寸翻到对侧（L/R 或 U/D） |
| `SET_LINEAR_LAYER_ORDER` | 线性分层 |
| `BALANCE_LINEAR_SIDES` | 线性左右/上下平衡 |
| `SET_RADIAL_SECTOR` | 直径/半径引线 8 扇区 |

向量化：`SyncLayoutVectorEnv`；PPO 默认用 spawn 子进程，便于 Windows / CUDA。

### 1.2 数据隔离

文件：`data/evalsys/forbidden_train_views.txt`（**127** 个 `view_key`）。

并集来源：perturb holdout、冻结 human gold、同事挑战。采集、BC、PPO、rollout eval 共用同一隔离表。Gold 与同事挑战的 `view_key` 交叉检查为空。

### 1.3 人工 gold 与人类奖励

| 项 | 状态 |
|---|---|
| T1 | 15 个 eval-only 视图，1–5 分 |
| T2 | 80 对 eval-only 偏好 |
| 选择种子 | 17 |
| 标签 | **尚未回收** |
| 旧 67/84 包 | 保留在 `legacy_proxy_holdout`，不再当正式 gold |
| 同事挑战 | 排除出 gold，永不选 checkpoint |
| 人类奖励 | `human_reward_beta` 默认 0；门禁要求冻结标签且模型超过公式基线 |

相关代码：`src/layout_agent/training/human_reward.py`、`src/layout_agent/evalset/isolation.py`、`data/evalsys/human_gold/manifest.json`。

### 1.4 训练链路与 smoke

采集（`runs/rl_smoke/trajectories/summary.json`）：

- 32 episode / 105 transition；train 73 / val 32；`view_overlap: []`
- 专家：rules / greedy / beam / random；扰动：clean / light
- 4 个源视图，隔离 127 个 forbidden views
- random 轨迹入库作基线，**默认不进 BC**

行为克隆（`runs/rl_smoke/bc/report.json`）：

- 离线 val top-1 ≈ 16.7%，离线门禁通过（smoke 阈值）
- **生产 rollout 门禁未过 greedy**：同一 held-out 视图上 policy 公式分 P ≈ 1184，greedy ≈ 883（越低越好）。见 `runs/rl_smoke/bc_gate.json`

掩码 PPO（`runs/rl_smoke/ppo/report.json`）：

- 从 BC 权重初始化，16 step / 2 update，CPU，2 个 sync env
- 动过 `SetRadialSector`、`SetLinearSide`、`ACCEPT`
- **这是训练器通路验证，不是可上线策略**

测试：`tests/test_layout_env.py`、`test_rl_pipeline.py`、`test_rl_training.py`、`test_human_gold.py`、`test_human_reward.py` 等合计 **37** 个用例（本周联调通过）。

### 1.5 代理奖励（非发版 KPI）

Bradley-Terry 代理奖励在 perturb val 上严格准确率约 **56%**，同事挑战 100 对上：

- 重算 ICM：**28%**
- 存图 ICM：**30%**

该数字只说明「占位 occupancy 代理 ≠ 人的布局偏好」。**不以 56% 或 30% 作为发版指标**，也不用同事挑战选 checkpoint。

---

## 2. 公共数据集筛选

筛选原则：Decide-only 需要 **2D 视图 + 可动尺寸（线性 / 直径 / 半径）+ 轮廓障碍**，能进 `LayoutState`。图框、GD&T、粗糙度目前不是动作空间。下载目录：`E:\workspace\data\extra_public_data`。清单：`inventory.json`。

### 2.1 结论总表

| 数据 | 形态 | 布局可用性 | 处理 |
|---|---|---|---|
| CMU Data-Augmentation | DXF + DIMENSION | **可用，train-only，量极小** | 仓库已在工作区；demo 已转 `LayoutState`，**未入库** |
| NIST MBE PMI | PDF 图纸 + 几何 STEP + 3D PMI STEP | **不能直接进 LayoutEnv** | 已下载验证 |
| ECCV 2026 TechDraw DXF | 轮廓 DXF/SVG，无尺寸 | **不适用** | 只抽检 25 个 DXF，**不下完整约 4.5 GB** |
| MechVQA / Kaggle JPG 等 | 光栅 / VQA | 不用 | 未下载 |
| Drawing2CAD / FloorPlanCAD / 专利 AutoDim 等 | 3D 重建或建筑平面 | 不用 | 未下载 |

### 2.2 NIST MBE PMI

路径：`E:\workspace\data\extra_public_data\nist_pmi`（约 223 MB）。

已解压定义包：CTC 01–05、FTC 06–11 的标注 PDF、无 PMI 的几何 STEP、xlsx 定义。AP242 STEP 含 3D PMI（`DIMENSIONAL_SIZE`、`GEOMETRIC_TOLERANCE`、`DRAUGHTING_CALLOUT` 等），不是 2D 布局动作。原生 CAD 大包保留 zip、未解压。

**结论：** 2D PDF 最多当视觉参考 / sim-to-real，转 `LayoutState` 需要 PDF/CAD 导入，不是 drop-in。

### 2.3 ECCV 2026 CAD Challenge

路径：`E:\workspace\data\extra_public_data\eccv2026_cad_challenge`。

下载了 README、license、`train.csv`（8344 条）以及 25 个 DXF（000000–000019，另抽查 000500 / 002000 / 004000 / 006000 / 008000）。**全部 0 个 `DIMENSION` / `AcDbDimension`**。任务是「三视图/渲染 → STEP」，不是标注布局。

**结论：** 不要下全量。对 Decide-only 没有可优化尺寸。

### 2.4 CMU Data-Augmentation

路径：`E:\workspace\github\Data-Augmentation-Engineering-Drawing`。

原论文做的是**零件分割增强**：轮廓基本不动，随机换「量哪两个点」。这与我们要的「同一组尺寸、换摆放」不是一回事，不能直接跑 `draw_random_dimension`。

真正能接上的只有：

1. 两张种子 DXF 已有 DIMENSION → `obstacles` + `linears` / `radials`。
2. 借用「偏移」思路但锁死测量对象：`p1/p2/anchor` 不变，只改边和距离（与现有 perturb pair 同类）。

种子核实：

| 文件 | DIMENSION | 映射结果 |
|---|---|---|
| `raw_data/0030_f.DXF` | 6（5 线性 dimtype=160，1 半径 164） | 5 linear + 1 radial |
| `raw_data/0053_f.DXF` | 2 线性 | 2 linear |

没有图框、GD&T、粗糙度。仓库内没有现成 `generated_dxf` / `generated_json`。

Demo（仅供查看，未写入训练 / gold / 冻结表）：

`E:\workspace\data\extra_public_data\cmu_layout_demo`

| 文件 | 含义 |
|---|---|
| `0030_f.preview.png` | 原图布局（尺寸在零件外，较干净） |
| `0030_f_sideflip.preview.png` | 同特征翻边并压近，文字叠轮廓 |
| `0053_f.preview.png` / `_sideflip` | 30×20 矩形；翻边后尺寸进零件 |
| `*.state.json` | `schema_version: agent1.1.0` 的 `LayoutState` |
| `*.cmu.json` | CMU 原始字段 |
| `*.instance.png` | 与现有 Perceiver 同一套 ICM |

字段对应：`linear_dimension` → `linears`（`p1/p2/point/value/text_point`）；非 160 的半径 → `radials`（CMU reader 会误当成直径，demo 已按 dimtype 纠正）。

**体量判断：** 两张简单零件补不了生产数据缺口。若入库，必须用独立 `view_key` 前缀（`CMU_DataAug/...`），且不得进入 127 个冻结 key。

---

## 3. 明确没有做的事

- 没有把 NIST / ECCV / CMU 写入 `layout_agent` 训练或评测集。
- 没有打开人类奖励（β 仍为 0）。
- 没有把 PPO smoke 或代理 56%/30% 当成发版结果。
- 没有改 `docs/decide_only_rl.md` 中的方案正文。

---

## 4. 下周建议

1. **回收 T1/T2 人工标签**，跑 `evalset_human_gold.py validate`；未过门禁前人类奖励保持关闭。
2. **BC 生产门禁**：smoke 未过 greedy，需要更大轨迹、更多视图后再谈 PPO。
3. **CMU**：若确认字段可用，再以隔离 train-only 方式入库两张种子 + 同特征 perturb；不要跑他们的随机换测量对象合成。
4. 继续以生产 OSS 图纸为主数据源；公开集只作补充，不再为 ECCV 全量或 NIST 原生 CAD 占磁盘。

---

## 5. 关键路径

| 用途 | 路径 |
|---|---|
| 仿真环境 | `src/layout_agent/training/env.py` |
| 采集 / BC / PPO | `src/layout_agent/training/collect.py`、`train_bc.py`、`train_ppo.py` |
| 隔离表 | `data/evalsys/forbidden_train_views.txt` |
| Gold 协议 | `data/evalsys/human_gold/manifest.json` |
| Smoke 产物 | `runs/rl_smoke/` |
| 公开数据清单 | `E:\workspace\data\extra_public_data\inventory.json` |
| CMU demo | `E:\workspace\data\extra_public_data\cmu_layout_demo\` |
| 方案文档（未改） | `docs/decide_only_rl.md` |
