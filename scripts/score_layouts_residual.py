"""Score the external-score corpus with the trained formula-residual checkpoint.

Writes runs/external_scores/residual.jsonl. The ranking `score` is formula +
residual and may lie outside 0–100. `display` is that score clipped to 0–100.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from layout_agent.models import LayoutState
from layout_agent.training.eval import CheckpointScorer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "external_scores"
CKPT = ROOT / "runs" / "reward_human_pairs_residual" / "best.pt"
PATH = OUT / "residual.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _done(path: Path) -> set[str]:
    return {row["id"] for row in _read_jsonl(path) if row.get("ok")}


def main() -> None:
    items = _read_jsonl(OUT / "corpus.jsonl")
    done = _done(PATH)
    pending = [item for item in items if item["id"] not in done]
    print(f"residual pending {len(pending)} done {len(done)}", flush=True)
    if not pending:
        return
    scorer = CheckpointScorer(CKPT, "cpu")
    scorer.encoder.work = OUT / "_encode_work"
    started = time.time()
    with PATH.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(pending, start=1):
            path = Path(item["state_path"])
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                state = LayoutState.model_validate(raw)
                arr = scorer.encoder.encode(state, base_dir=path.parent.parent)
                parts = scorer.score_parts(arr, state)
                row = {
                    "id": item["id"],
                    "ok": True,
                    "score": round(float(parts["score"]), 4),
                    "display": round(float(parts["display"]), 2),
                    "formula_score": round(float(parts["formula_score"]), 4),
                    "residual": round(float(parts["residual"]), 4),
                }
            except Exception as exc:  # noqa: BLE001
                row = {"id": item["id"], "ok": False, "error": str(exc)[:400]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(pending):
                handle.flush()
                print(f"residual {index}/{len(pending)} {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
