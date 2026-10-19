"""Score a single layout state (0-100 display + unclipped ranking score).

  python -m layout_agent.training.score --ckpt runs/reward_human_pairs_residual/best.pt \\
      --state path/to/view.state.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layout_agent.models import LayoutState
from layout_agent.training.eval import CheckpointScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True, help="LayoutState json")
    ap.add_argument("--device")
    args = ap.parse_args()
    path = Path(args.state)
    raw = json.loads(path.read_text(encoding="utf-8"))
    state = LayoutState.model_validate(raw)
    scorer = CheckpointScorer(Path(args.ckpt), args.device)
    arr = scorer.encoder.encode(state, base_dir=path.parent.parent)
    parts = scorer.score_parts(arr, state)
    out = {
        "state": str(path.resolve()),
        "score": round(float(parts["score"]), 4),
        "display": round(float(parts["display"]), 2),
        "formula_score": round(float(parts["formula_score"]), 4),
        "residual": round(float(parts["residual"]), 4),
        "formula_residual": bool(scorer.formula_residual),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
