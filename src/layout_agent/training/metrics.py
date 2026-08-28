"""Evaluation metrics — same protocol as scripts/evalset_baseline_consistency.py."""

from __future__ import annotations

from typing import Dict, List, Tuple


def spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    def ranks(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(xs) < 3:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx * dy else 0.0


def pairwise_agreement(
    scored: List[Tuple[float, float, str]], margin: float = 1e-6
) -> Dict[str, float]:
    """scored: (score_a, score_b, human_prefer 'a'|'b'|'tie') per pair."""
    agree = disagree = model_tie = human_tie = 0
    for sa, sb, prefer in scored:
        if prefer == "tie":
            human_tie += 1
            continue
        if abs(sa - sb) <= margin:
            model_tie += 1
            continue
        if ("a" if sa > sb else "b") == prefer:
            agree += 1
        else:
            disagree += 1
    decided = agree + disagree
    return {
        "n": len(scored),
        "agree": agree,
        "disagree": disagree,
        "model_tie": model_tie,
        "human_tie": human_tie,
        "agreement_rate": (agree / decided) if decided else 0.0,
    }
