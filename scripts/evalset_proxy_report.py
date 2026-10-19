#!/usr/bin/env python3
"""Assemble the reproducible weak-supervision proxy training report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def best_row(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {}
    return max(history, key=lambda row: float((row.get("val") or {}).get("all", 0.0)))


def artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def metric(report: Dict[str, Any], level: str = "overall") -> Dict[str, Any]:
    return ((report.get("t2") or {}).get(level) or {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-dir", default="data/evalsys/proxy")
    ap.add_argument("--smoke-run", default="runs/reward_proxy_smoke")
    ap.add_argument("--run", default="runs/reward_proxy_a800_strict")
    ap.add_argument("--legacy-run", default="runs/reward_v0")
    ap.add_argument("--hardware", default="8x NVIDIA A800 80GB PCIe")
    ap.add_argument("--torch", default="2.6.0+cu124")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    args = ap.parse_args()

    proxy = (ROOT / args.proxy_dir).resolve()
    smoke = (ROOT / args.smoke_run).resolve()
    run = (ROOT / args.run).resolve()
    legacy = (ROOT / args.legacy_run).resolve()
    out_json = Path(args.out_json).resolve() if args.out_json else proxy / "proxy_report.json"
    out_md = Path(args.out_md).resolve() if args.out_md else proxy / "REPORT.md"

    frozen = load(proxy / "manifest.json")
    formula_val = load(proxy / "formula_perturb_val.json")
    formula_challenge = load(proxy / "formula_colleague_challenge.json")
    model_val = load(run / "eval_perturb_val.json")
    model_challenge = load(run / "eval_colleague_challenge.json")
    t1 = load(run / "eval_t1_diagnostic.json")
    smoke_history = load(smoke / "history.json")
    smoke_config = load(smoke / "config.json")
    history = load(run / "history.json")
    config = load(run / "config.json")
    legacy_history = load(legacy / "history.json")

    best = best_row(history)
    smoke_best = best_row(smoke_history)
    legacy_best = best_row(legacy_history)
    model_overall = metric(model_val)
    model_heavy = metric(model_val, "heavy")
    overall = float(model_overall.get("strict_accuracy", 0.0))
    heavy = float(model_heavy.get("strict_accuracy", 0.0))
    if overall >= 0.65 and heavy >= 0.75:
        gate = "engineering_gate_pass"
    elif overall < 0.55:
        gate = "stop_tuning_add_data"
    else:
        gate = "pipeline_only"

    report = {
        "schema_version": 1,
        "title": "弱监督代理评测，不是人工测试 KPI",
        "warning": (
            "Labels are synthetic, formula-derived, or role-derived. "
            "Do not report these results as a human test KPI or shipping claim."
        ),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "decision": {
            "status": gate,
            "engineering_gate": {
                "overall_strict_accuracy_min": 0.65,
                "heavy_strict_accuracy_min": 0.75,
            },
            "observed": {
                "overall_strict_accuracy": overall,
                "heavy_strict_accuracy": heavy,
                "colleague_strict_accuracy": metric(model_challenge).get(
                    "strict_accuracy"
                ),
            },
            "conclusion": (
                "Pipeline is operational, but the model does not meet the "
                "engineering gate and does not generalize to colleague pairs."
            ),
        },
        "data": frozen,
        "environment": {
            "hardware": args.hardware,
            "torch": args.torch,
            "world_size": config.get("world_size"),
            "global_batch_size": config.get("global_batch_size"),
            "seed": config.get("seed"),
            "amp": config.get("amp"),
            "config": rel(run / "config.json"),
        },
        "smoke": {
            "run": rel(smoke),
            "best": smoke_best,
            "passed": float((smoke_best.get("val") or {}).get("all", 0.0)) > 0.5,
            "config": smoke_config,
            "history": smoke_history,
        },
        "training": {
            "run": rel(run),
            "best": best,
            "early_stop_epoch": history[-1].get("epoch") if history else None,
            "history": history,
        },
        "evaluation": {
            "formula_perturb_val": formula_val,
            "formula_colleague_challenge": formula_challenge,
            "model_perturb_val": model_val,
            "model_colleague_challenge": model_challenge,
            "model_t1_diagnostic": t1,
            "strict_accuracy_delta": {
                "model_minus_formula_perturb_val": overall
                - float(metric(formula_val).get("strict_accuracy", 0.0)),
                "model_minus_formula_colleague": float(
                    metric(model_challenge).get("strict_accuracy", 0.0)
                )
                - float(metric(formula_challenge).get("strict_accuracy", 0.0)),
            },
        },
        "legacy_comparison": {
            "reward_v0_recorded_best_val": (legacy_best.get("val") or {}).get("all"),
            "reward_v0_best_epoch": legacy_best.get("epoch"),
            "current_minus_legacy": overall
            - float((legacy_best.get("val") or {}).get("all", 0.0)),
            "caveat": (
                "reward_v0 used the older 67-pair validation split and its "
                "checkpoint is unavailable, so this is not an apples-to-apples "
                "re-evaluation on the frozen 87-pair proxy."
            ),
        },
        "artifacts": {
            "checkpoint": artifact(run / "best.pt"),
            "last_checkpoint": artifact(run / "last.pt"),
            "config": artifact(run / "config.json"),
            "history": artifact(run / "history.json"),
            "eval_perturb_val": artifact(run / "eval_perturb_val.json"),
            "eval_colleague_challenge": artifact(
                run / "eval_colleague_challenge.json"
            ),
            "eval_t1_diagnostic": artifact(run / "eval_t1_diagnostic.json"),
        },
        "reproduce": [
            "python scripts/evalset_prepare_proxy.py",
            (
                "python scripts/evalset_baseline_consistency.py "
                "--t2 data/evalsys/pairs_perturb/manifest.jsonl --split val "
                "--out data/evalsys/proxy/formula_perturb_val.json"
            ),
            (
                "NPROC_PER_NODE=8 bash scripts/train_reward_a800.sh "
                "data/evalsys/pairs_perturb/manifest.jsonl "
                "runs/reward_proxy_a800_strict"
            ),
            (
                "python -m layout_agent.training.eval "
                "--ckpt runs/reward_proxy_a800_strict/best.pt "
                "--t2 data/evalsys/proxy/perturb_val.jsonl --split val"
            ),
        ],
    }

    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    val_formula = metric(formula_val)
    challenge_formula = metric(formula_challenge)
    challenge_model = metric(model_challenge)
    t1_metrics = t1.get("t1") or {}
    markdown = f"""# 弱监督代理评测，不是人工测试 KPI

> {report["warning"]}

## 结论
- 状态：`{gate}`。
- perturb val：模型 strict {pct(overall)}，公式 strict {pct(val_formula.get("strict_accuracy"))}。
- heavy：模型 strict {pct(heavy)}，门槛为 75%。
- colleague challenge：模型 strict {pct(challenge_model.get("strict_accuracy"))}，公式 strict {pct(challenge_formula.get("strict_accuracy"))}。
- T1 自动标签诊断：Spearman {t1_metrics.get("spearman", "n/a")}（仅诊断，不是门槛）。
- 结论：训练管线可用，但未通过工程门槛，且对同事真实扰动对没有泛化。

## 数据完整性
- perturb：{frozen["sources"]["pairs_perturb"]["train_pairs"]} train / {frozen["sources"]["pairs_perturb"]["val_pairs"]} val，view_key 交叉为 0。
- colleague challenge：{frozen["sources"]["colleague"]["pairs"]} 对，不参与训练或选 checkpoint。
- T1：{frozen["sources"]["t1"]["tasks"]} 条；已删除重复 ID。
- 数据清单：`{rel(proxy / "manifest.json")}`。

## 训练
- 环境：{args.hardware}，PyTorch {args.torch}，bf16 DDP。
- smoke：最佳 epoch {smoke_best.get("epoch")}，val strict {pct((smoke_best.get("val") or {}).get("all"))}。
- A800：最佳 epoch {best.get("epoch")}，val strict {pct((best.get("val") or {}).get("all"))}；epoch {history[-1].get("epoch")} patience early stop。
- checkpoint：`{rel(run / "best.pt")}`。
- SHA-256：`{report["artifacts"]["checkpoint"]["sha256"]}`。

## 与旧模型比较
- reward_v0 记录值：{pct((legacy_best.get("val") or {}).get("all"))}（旧 67-pair val）。
- 当前 frozen val：{pct(overall)}（87-pair val）。
- 两者 split 不同，旧 checkpoint 缺失，不能把差值解释为模型回退或提升。

## 下一步
- 停止继续调参；优先增加与 colleague 分布一致的数据。
- 对外声称质量提升前，仍需最小人工测试集：68 条 T2 + 13 条 T1。
"""
    out_md.write_text(markdown, encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
