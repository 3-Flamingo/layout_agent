#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.agent import LayoutAgent
from layout_agent.config import load_config
from layout_agent.perceive import Perceiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/datasets/demo")
    ap.add_argument("--case", default=None, help="optional state_id filter")
    args = ap.parse_args()

    cfg = load_config()
    ds = Path(args.dataset)
    if not ds.is_absolute():
        ds = ROOT / ds
    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    agent = LayoutAgent(cfg)
    perceiver = Perceiver(cfg)
    out_dir = ds / "runs"
    out_dir.mkdir(exist_ok=True)

    results = []
    for item in manifest["items"]:
        state = perceiver.load_state(ds / item["state_path"], ds)
        if args.case and state.state_id != args.case:
            continue
        # fix absolute image paths for eval
        state.raster.instance_image = str((ds / item["instance_image_path"]).resolve())
        if item.get("obstacle_image_path"):
            state.raster.obstacle_image = str((ds / item["obstacle_image_path"]).resolve())

        run_dir = out_dir / state.state_id
        result = agent.run(state, run_dir)
        results.append(result.model_dump(mode="json"))
        print(
            f"{state.state_id}: improved={result.improved} "
            f"P {result.report_before.metrics.get('P'):.2f} -> "
            f"{(result.report_after.metrics.get('P') if result.report_after else None)} "
            f"reason={result.reason}"
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
