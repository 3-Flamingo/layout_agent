#!/usr/bin/env python3
"""Gate the learned reward against frozen human T1/T2 and formula reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.training.human_reward import build_human_reward_gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--formula-report", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument(
        "--gold-manifest",
        default=str(ROOT / "data" / "evalsys" / "human_gold" / "manifest.json"),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-t1", type=int, default=15)
    parser.add_argument("--min-t2", type=int, default=80)
    parser.add_argument("--min-t1-gain", type=float, default=0.02)
    parser.add_argument("--min-t2-gain", type=float, default=0.02)
    args = parser.parse_args()
    report = build_human_reward_gate(
        json.loads(Path(args.model_report).read_text(encoding="utf-8")),
        json.loads(Path(args.formula_report).read_text(encoding="utf-8")),
        reward_checkpoint=Path(args.reward_checkpoint),
        gold_manifest=Path(args.gold_manifest),
        min_t1=args.min_t1,
        min_t2=args.min_t2,
        min_t1_gain=args.min_t1_gain,
        min_t2_gain=args.min_t2_gain,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
