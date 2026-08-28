# Layout Agent → drawing-2d 标注应用接口

> 用途：Layout Agent 完成优化后，由 **drawing-2d** 的标注布局/写回算法把几何改动落到真实图纸，并刷新显示。  
> 对接原则：**Agent 负责「改哪些、改成什么」；drawing-2d 负责「按 CAD 语义写回并重绘」**。

---

## 1. 职责边界

| 系统 | 做什么 | 不做什么 |
|------|--------|----------|
| **Layout Agent** | 感知 / 评价 / 决策 /（内部模拟执行）→ 产出最终布局目标 | 不直接改 CadDocument、不写 OSS |
| **drawing-2d** | 按本接口把目标几何应用到 `CadAlignedDimension` / `CadDiameterDimension` / `CadRadialDimension`，刷新图元与视图 | 不重新跑 Agent 决策 |

Agent 内部当前会维护两份结果（见 §5）：

1. **决策动作** `actions_applied`：离散动作（翻边、扇区等），便于审计与回放。  
2. **最终几何** `best.state`：视图局部坐标系下的绝对位置。

对接 **写回图纸** 时，请以 **§2「应用补丁 LayoutApplyPatch」** 为准（可由 `before.state` + `best.state` 差分生成）。  
不要只把 `SetLinearSide` 等离散动作丢给 drawing-2d 再实现一遍布局——那会与 Agent 已算好的终点不一致。

---

## 2. 推荐接口：`LayoutApplyPatch`（几何写回）

一次 Agent 运行对应 **一个视图** 的一份补丁。drawing-2d 按 `changes[]` 逐条写回即可。

### 2.1 顶层结构

```json
{
  "schema_version": "layout_apply.1.0.0",
  "solution_id": "1511437367716814848",
  "tenant_id": "250",
  "part_number": "TEF648102101",
  "view": {
    "view_id": "42288",
    "view_uuid": "E5749cOT",
    "view_type": "front"
  },
  "coords": {
    "space": "view_local",
    "y_up": true,
    "units": "mm",
    "note": "与 CadView 局部坐标一致；写回前需按视图 matrix 变换到图纸/模型空间（与现有 dim_layout collect/commit 一致）"
  },
  "summary": {
    "changed": 3,
    "unchanged": 26,
    "improved": true,
    "agent_reason": "StopAccept"
  },
  "changes": [ /* LayoutChange */ ]
}
```

### 2.2 单条变更 `LayoutChange`

每条对应 **一个 CAD 标注**（`id` = 元素 `uuid`，与生产拉取一致）。

#### 线性标注 `category: "linear"` → `CadAlignedDimension`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | CAD `uuid` |
| `category` | `"linear"` | |
| `cad_ctor` | `"CadAlignedDimension"` | 便于实现侧分派 |
| `before.dim_line_point` | `{x,y}` | 优化前尺寸线参考点（对应 Agent `LinearAnno.point`） |
| `after.dim_line_point` | `{x,y}` | **写回目标** → `db.dimLinePoint` |
| `before.dir` / `after.dir` | `{x,y}` | 单位法向（侧向），仅辅助校验；**不必单独写 CAD 字段** |
| `offset` | `{dx,dy}` | `after - before`，便于日志/动画；**写回请用 `after` 绝对坐标** |
| `unchanged` | string[] | 明确不应改动的语义，如 `["xLine1Point","xLine2Point","value"]` |

示例：

```json
{
  "id": "L_overall",
  "category": "linear",
  "cad_ctor": "CadAlignedDimension",
  "before": {
    "dim_line_point": { "x": 100.0, "y": 16.0 },
    "dir": { "x": 0.0, "y": -1.0 }
  },
  "after": {
    "dim_line_point": { "x": 100.0, "y": -8.0 },
    "dir": { "x": 0.0, "y": -1.0 }
  },
  "offset": { "dx": 0.0, "dy": -24.0 },
  "unchanged": ["xLine1Point", "xLine2Point", "measurement"],
  "hint": {
    "side": "D",
    "text_position": { "x": 100.0, "y": -8.0 }
  }
}
```

**drawing-2d 写回建议**（对齐现有 `aligned/collect.ts` 的 `commit`）：

```ts
dimension.init({
  ...dump,
  dimLinePoint: after.dim_line_point,          // [x, y]
  textPosition: hint.text_position ?? after.dim_line_point,
});
```

- **改**：`dimLinePoint`（必改）；通常同步 `textPosition` 到尺寸线中点或沿尺寸线的文字位。  
- **不改**：`xLine1Point` / `xLine2Point`（测量边）、公差与样式 override（除非另有约定）。

#### 直径 / 半径 `category: "diameter" | "radial"`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | CAD `uuid` |
| `category` | `"diameter"` \| `"radial"` | |
| `cad_ctor` | `"CadDiameterDimension"` \| `"CadRadialDimension"` | |
| `before.anchor` / `after.anchor` | `{x,y}` | 圆心/锚点；**应保持不变** |
| `before.path` / `after.path` | `{x,y}[]` | 引线折线，末点为文字位 |
| `before.text_position` / `after.text_position` | `{x,y}` | = `path` 末点 |
| `offset` | `{dx,dy}` | 文字位位移 |

示例：

```json
{
  "id": "7e2-gLyo",
  "category": "diameter",
  "cad_ctor": "CadDiameterDimension",
  "before": {
    "anchor": { "x": 80.0, "y": 60.0 },
    "path": [
      { "x": 80.0, "y": 60.0 },
      { "x": 100.0, "y": 8.0 },
      { "x": 100.0, "y": 8.0 }
    ],
    "text_position": { "x": 100.0, "y": 8.0 }
  },
  "after": {
    "anchor": { "x": 80.0, "y": 60.0 },
    "path": [
      { "x": 80.0, "y": 60.0 },
      { "x": 80.0, "y": 85.0 },
      { "x": 80.0, "y": 85.0 }
    ],
    "text_position": { "x": 80.0, "y": 85.0 }
  },
  "offset": { "dx": -20.0, "dy": 77.0 },
  "unchanged": ["center", "chordPoint", "farChordPoint"]
}
```

**drawing-2d 写回建议**（对齐 `diameter/collect.ts` / `radial/collect.ts`）：

- 直径：更新 `textPosition`；必要时按现有算法由文字位反推 jog / `chordPoint` 可见几何（与 `DiameterDimCommonUtil` 一致）。  
- 半径：更新 `textPosition`（及若有的折点）；**保持 `center`/`anchor` 不变**。

---

## 3. 坐标系与 ID 约定

### 3.1 坐标

- Agent 状态与补丁均使用 **`view_local`，Y 向上，单位 mm**（`LayoutState.coords`）。  
- 与 OSS/`CadView` 局部空间一致；drawing-2d 写回时按该视图的 `matrix` 变换到图纸空间，**变换规则必须与现有 dim_layout 的 collect/commit 相同**。  
- `offset.dx/dy` 也在同一 `view_local` 下；勿与屏幕像素混淆。

### 3.2 标注 ID

| 来源 | `id` 规则 |
|------|-----------|
| 生产图纸 | CAD 元素 `uuid`（可能含 `*` 等字符，需原样匹配） |
| 合成样例 | 如 `L_overall`、`D_hole` |

查找：`doc.getElementByUuid(id)` 或现有 uuid → element 映射。  
**不要用 ICM 颜色作为写回主键**（颜色仅用于 Agent 栅格感知）。

### 3.3 视图定位

补丁 `view.view_uuid` / `view.view_id` 对应 `CadView`。只在该视图所属标注上应用；勿跨视图写回。

---

## 4. 应用语义（drawing-2d 侧）

建议实现函数（命名示例）：

```ts
applyLayoutPatch(doc: IDocument, patch: LayoutApplyPatch): ApplyResult
```

伪流程：

1. 按 `view.view_uuid` 定位 `CadView`，准备坐标变换。  
2. 对 `changes[]`：  
   - 按 `id` 取元素；类型与 `cad_ctor`/`category` 不符则记入 `errors` 并跳过。  
   - **线性**：写 `dimLinePoint`（+ `textPosition`）。  
   - **直径/半径**：写 `textPosition`（及引线相关点，按现有 util）。  
3. 触发重绘 / 脏标记 /（如需要）格网刷新。  
4. 返回：

```json
{
  "ok": true,
  "applied": ["L_overall", "L_seg"],
  "skipped": [],
  "errors": [{ "id": "???", "message": "element not found" }]
}
```

### 4.1 偏移量怎么用

| 用法 | 建议 |
|------|------|
| **权威写回** | 使用 `after.*` **绝对坐标** |
| 动画、diff UI、日志 | 使用 `offset.{dx,dy}` |
| 校验 | `after ≈ before + offset`（浮点容差内） |

不推荐：仅下发 `offset` 而无 `after`（容易与视图矩阵、二次布局叠加出错）。

---

## 5. Agent 现有产物（对照）

运行目录：`{dataset}/runs/{state_id}/`

| 文件 | 内容 | 对接用途 |
|------|------|----------|
| `before.state.json` | 优化前 `LayoutState` | 差分起点 |
| `best.state.json` | 优化后 `LayoutState` | **几何终点（绝对坐标）** |
| `result.json` | `AgentResult` | 含 `actions_applied`、评价报告 |
| `report_before.json` / `report_after.json` | 评价指标 | 展示用，非写回 |

HTTP（本地 Demo）：

- `POST /api/cases/{case_id}/run` → `AgentResult`  
- `GET /api/cases/{case_id}` → `{ state, best_state, run }`  
- `GET /api/run-file/{case_id}/best.state.json`

### 5.1 决策动作 `Action`（审计用，非写回主契约）

```ts
type ActionType =
  | "SetLinearSide"
  | "SetLinearLayerOrder"
  | "BalanceLinearSides"
  | "SetRadialSector"
  | "ReorderRadial"
  | "Accept"
  | "Revert";

interface Action {
  type: ActionType;
  target_id?: string | null;  // 标注 uuid
  side?: "L" | "R" | "U" | "D" | null;
  sector?: number | null;     // 0..7，径向/直径引线扇区
  group_id?: string | null;
  order?: string[] | null;
  reason: string;
}
```

`result.json` 中 `actions_applied: Action[][]`：外层 = 每一轮 Decide，内层 = 该轮并行/顺序动作。

| type | Agent 内部几何效果 | drawing-2d |
|------|-------------------|------------|
| `SetLinearSide` | 改线性 `point`/`dir`，同侧重排层 | **勿直接执行**；看补丁 `after` |
| `BalanceLinearSides` | 同轴两侧再平衡 + 分层 | 同上 |
| `SetLinearLayerOrder` | 外推一层再打包 | 同上 |
| `SetRadialSector` | 改直径/半径 `path` | 同上 |
| `Accept` / `Revert` | 停止 / 放弃 | 无几何 |

---

## 6. 从 `best.state` 生成补丁（算法约定）

伪代码（Agent 侧或对接适配层均可）：

```
for each anno in union(linears, diameters, radials):
  b = before[anno.id]; a = best[anno.id]
  if category == linear:
    if |a.point - b.point| > eps:
      emit linear change (dim_line_point, dir, offset)
  else:
    if path/text_position changed:
      emit radial/diameter change
```

线性字段映射：

| Agent `LinearAnno` | CAD |
|--------------------|-----|
| `id` | `uuid` |
| `p1` / `p2` | `xLine1Point` / `xLine2Point`（只读） |
| `point` | `dimLinePoint` |
| `dir` | 侧向提示（无单独 db 字段） |
| `text.content` | 显示值（只读） |

径向/直径：

| Agent `RadialAnno` | CAD |
|--------------------|-----|
| `id` | `uuid` |
| `anchor` | 圆心 / 锚点（只读） |
| `path[-1]` | `textPosition` |
| `path` | 引线折线（实现可选写 jog） |

---

## 7. 完整最小示例（线性）

```json
{
  "schema_version": "layout_apply.1.0.0",
  "solution_id": "SYN_case_crowded",
  "tenant_id": "0",
  "part_number": "",
  "view": {
    "view_id": "V_front",
    "view_uuid": "uuid-case_crowded",
    "view_type": "Front"
  },
  "coords": { "space": "view_local", "y_up": true, "units": "mm" },
  "summary": { "changed": 1, "unchanged": 2, "improved": true, "agent_reason": "no_conflict" },
  "changes": [
    {
      "id": "L_overall",
      "category": "linear",
      "cad_ctor": "CadAlignedDimension",
      "before": {
        "dim_line_point": { "x": 100, "y": 16 },
        "dir": { "x": 0, "y": -1 }
      },
      "after": {
        "dim_line_point": { "x": 100, "y": -8 },
        "dir": { "x": 0, "y": -1 }
      },
      "offset": { "dx": 0, "dy": -24 },
      "unchanged": ["xLine1Point", "xLine2Point"],
      "hint": { "side": "D", "text_position": { "x": 100, "y": -8 } }
    }
  ]
}
```

---

## 8. 版本与扩展

- 当前补丁 schema：`layout_apply.1.0.0`  
- Agent 状态 schema：`agent1.1.0`（`LayoutState.schema_version`）  
- 后续可扩展：形位公差、粗糙度、文字块、多视图批量 `patches[]`  
- 若 drawing-2d 需要「仅偏移」模式，可增加 `apply_mode: "absolute" | "delta"`，默认 **`absolute`**

---

## 9. 联调检查清单

- [ ] `id` 能在目标 `CadView` 下解析到唯一元素  
- [ ] 坐标经视图 matrix 变换后与屏幕/图纸一致（抽样 1～2 个尺寸人工核对）  
- [ ] 线性只动尺寸线，测量边 `p1/p2` 不变  
- [ ] 直径/半径锚点不变，仅文字/引线变化  
- [ ] 应用后图纸刷新可见；可撤销（可选：写回前 snapshot）  
- [ ] 无变更时 `changes` 为空数组，drawing-2d 为 no-op  

---

## 附录 A：与现有 dim_layout 的关系

drawing-2d 现有流水线（`DefaultDimensionsLayoutStrategyPlugin` → aligned / diameter / radial `collect` + `commit`）已经用：

- 线性：`point` → `dimLinePoint`，`tp` → `textPosition`  
- 直径/半径：`textPosition` 等  

本接口产出的 `after.*` 可视为 **外部布局器给出的目标 `IDAligned` / 直径收集结果**，走同一套 `commit`，避免再发明写回路径。
