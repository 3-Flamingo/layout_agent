# Layout Agent（一期）

单视图标注布局优化 Agent：感知 → 评价 → 决策 → 执行。

- **Active**：线性 / 直径 / 半径（着色实例图 + 颜色字典）
- **Obstacle**：其它图元
- **决策**：规则引擎 + 可选 LLM（对齐焊装交付平台 OpenAI 兼容接口，默认 `deepseek-v4-pro`）

## 快速开始

```bash
cd E:\workspace\layout_agent
pip install -r requirements.txt

# 从生产环境拉取真实图纸（默认方案 070-3579 / solutionId=2082016429787918336）
python scripts/fetch_production_dataset.py
# 已下载过 OSS 时可加速：
# python scripts/fetch_production_dataset.py --reuse-raw

# 跑一遍 Agent（默认数据集见 config/default.yaml）
python scripts/run_agent.py --dataset data/datasets/prod_070-3579

# 启动前端
python -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

生产数据来源：
- 图纸：https://drawing.designorder.cn/?solutionId=2082016429787918336&tenantId=0
- API：`api.designorder.cn/designBackend/doMistServer/mistSolution/selectById`
- 文档：OSS `solutionSolutionWebDocument`（与 2d_service `SolutionLoadUtil` 相同链路）

合成样例（仅调试）仍可用：`python scripts/generate_dataset.py`。

## LLM 配置（可选）

与 `businessplatform_indi` 交付平台一致，使用百炼 / 私有 OpenAI 兼容网关：

```bash
# .env
DASHSCOPE_API_KEY=your_key
OPENAI_BASE_URL=https://llm-e61q4da7pbj3wxqv.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=deepseek-v4-pro
LAYOUT_AGENT_USE_LLM=1
```

未配置 Key 时决策自动回退规则引擎。

## 目录

```
src/layout_agent/   # 四系统 + LLM 客户端
scripts/            # 数据生成、批跑
web/                # FastAPI 前端
data/datasets/      # 生成的数据集
config/             # 默认配置
```

## 模型选型说明

| 系统 | 模型 |
|------|------|
| 感知 / 评价 / 执行 | 无 LLM（几何 + 像素规则） |
| 决策 | `deepseek-v4-pro`（与交付平台相同）：结构化 JSON 动作规划；低温；失败回退规则 |

详见 `config/default.yaml`。
