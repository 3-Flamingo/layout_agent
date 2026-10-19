#!/usr/bin/env python3
"""Per-pair frozen-test predictions for several reward checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.training.eval import CheckpointScorer  # noqa: E402


def _pred(sa: float, sb: float, margin: float) -> str:
    if abs(sa - sb) <= margin:
        return "tie"
    return "a" if sa > sb else "b"


def score_ckpt(
    name: str,
    ckpt: Path,
    rows: List[Dict[str, Any]],
    test_path: Path,
    device: str,
    margin: float,
) -> Dict[str, Any]:
    scorer = CheckpointScorer(ckpt, device)
    pairs = []
    for rec in rows:
        prefer = str((rec.get("label") or {}).get("prefer") or "").lower()
        sa = scorer.logit_pair_side(rec, "a", test_path.parent)
        sb = scorer.logit_pair_side(rec, "b", test_path.parent)
        if sa is None or sb is None:
            continue
        pred = _pred(sa, sb, margin)
        ok = pred == prefer and pred != "tie"
        pairs.append(
            {
                "pair_id": rec["pair_id"],
                "view_key": rec.get("view_key"),
                "source": rec.get("source"),
                "n_ann": rec.get("n_ann"),
                "pose_delta": rec.get("pose_delta"),
                "view_type": rec.get("view_type"),
                "prefer": prefer,
                "sa": round(float(sa), 6),
                "sb": round(float(sb), 6),
                "pred": pred,
                "ok": ok,
            }
        )
    n = len(pairs)
    n_ok = sum(1 for p in pairs if p["ok"])
    return {
        "name": name,
        "ckpt": str(ckpt),
        "n": n,
        "n_ok": n_ok,
        "n_err": n - n_ok,
        "strict_accuracy": (n_ok / n) if n else 0.0,
        "pred_counts": dict(Counter(p["pred"] for p in pairs)),
        "prefer_b_when_wrong": sum(1 for p in pairs if not p["ok"] and p["pred"] == "a"),
        "pairs": pairs,
    }


def err_ids(block: Dict[str, Any]) -> set:
    return {p["pair_id"] for p in block["pairs"] if not p["ok"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/evalsys/human_pairs/test.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--margin", type=float, default=1e-6)
    ap.add_argument("--out", default="runs/reward_test_error_compare.json")
    ap.add_argument(
        "--ckpt",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="repeatable: --ckpt real runs/reward_human_pairs/best.pt",
    )
    args = ap.parse_args()
    test_path = Path(args.test)
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    rows = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(r.get("split") for r in rows):
        rows = [r for r in rows if r.get("split") in (None, "test", "")]

    ckpts = args.ckpt or [
        ("real", "runs/reward_human_pairs/best.pt"),
        ("synth_light", "runs/reward_human_pairs_synth_light/best.pt"),
        ("synth_full", "runs/reward_human_pairs_synth/best.pt"),
        ("synth_dense", "runs/reward_human_pairs_synth_dense/best.pt"),
    ]
    blocks = []
    for name, rel in ckpts:
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT / path
        print(f"scoring {name} {path}")
        blocks.append(score_ckpt(name, path, rows, test_path, args.device, args.margin))
        print(
            f"  {name}: {blocks[-1]['n_ok']}/{blocks[-1]['n']} "
            f"acc={blocks[-1]['strict_accuracy']:.4f} pred={blocks[-1]['pred_counts']}"
        )

    by_name = {b["name"]: b for b in blocks}
    names = [b["name"] for b in blocks]
    pairwise = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ea, eb = err_ids(by_name[a]), err_ids(by_name[b])
            same_pred = sum(
                1
                for pa, pb in zip(by_name[a]["pairs"], by_name[b]["pairs"])
                if pa["pair_id"] == pb["pair_id"] and pa["pred"] == pb["pred"]
            )
            pairwise.append(
                {
                    "a": a,
                    "b": b,
                    "n_err_a": len(ea),
                    "n_err_b": len(eb),
                    "err_intersect": len(ea & eb),
                    "err_only_a": len(ea - eb),
                    "err_only_b": len(eb - ea),
                    "identical_error_set": ea == eb,
                    "a_errors_subset_of_b": ea <= eb,
                    "b_errors_subset_of_a": eb <= ea,
                    "pred_agree": same_pred,
                    "pred_agree_rate": same_pred / max(1, by_name[a]["n"]),
                }
            )
            print(
                f"  {a} vs {b}: err ∩={len(ea & eb)} only_{a}={len(ea - eb)} "
                f"only_{b}={len(eb - ea)} identical={ea == eb} "
                f"pred_agree={same_pred}/{by_name[a]['n']}"
            )

    # pairs all four wrong / all four right / split by 67 vs 54 groups
    err_map = {n: err_ids(by_name[n]) for n in names}
    all_ids = [p["pair_id"] for p in blocks[0]["pairs"]]
    union = set().union(*err_map.values()) if err_map else set()
    all_wrong = set(all_ids)
    for s in err_map.values():
        all_wrong &= s
    all_right = [i for i in all_ids if i not in union]

    meta_by_id = {p["pair_id"]: p for p in blocks[0]["pairs"]}

    def bucket(ids):
        src = Counter(meta_by_id[i].get("source") for i in ids if i in meta_by_id)
        return {"n": len(ids), "source": dict(src)}

    out = {
        "test": str(test_path),
        "n_test": len(blocks[0]["pairs"]) if blocks else 0,
        "models": [
            {k: v for k, v in b.items() if k != "pairs"}
            | {"error_pair_ids": sorted(err_ids(b))}
            for b in blocks
        ],
        "pairwise": pairwise,
        "all_four_wrong": bucket(all_wrong),
        "all_four_right": bucket(all_right),
        "union_wrong": bucket(union),
    }
    # attach full pair table
    table = []
    for rec in blocks[0]["pairs"]:
        row = {
            "pair_id": rec["pair_id"],
            "view_key": rec["view_key"],
            "source": rec["source"],
            "n_ann": rec["n_ann"],
            "pose_delta": rec["pose_delta"],
            "prefer": rec["prefer"],
        }
        for b in blocks:
            p = next(x for x in b["pairs"] if x["pair_id"] == rec["pair_id"])
            row[f"{b['name']}_pred"] = p["pred"]
            row[f"{b['name']}_ok"] = p["ok"]
        table.append(row)
    out["pairs"] = table

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print("all_four_wrong", out["all_four_wrong"])
    print("all_four_right", out["all_four_right"])
    print("union_wrong", out["union_wrong"])


if __name__ == "__main__":
    main()
