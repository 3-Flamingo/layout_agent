"""Evaluation metrics — same protocol as scripts/evalset_baseline_consistency.py."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

Score = Union[float, Sequence[float]]


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


def _model_prefers(sa: Score, sb: Score, *, margin: float, force_choice: bool) -> str:
    """Return ``a``, ``b``, or ``tie`` for which side the model ranks higher."""
    if isinstance(sa, Sequence) and not isinstance(sa, (str, bytes)):
        if tuple(sa) > tuple(sb):
            return "a"
        if tuple(sb) > tuple(sa):
            return "b"
        return "tie"
    diff = float(sa) - float(sb)
    if force_choice:
        if diff > 0:
            return "a"
        if diff < 0:
            return "b"
        return "tie"
    if abs(diff) <= margin:
        return "tie"
    return "a" if diff > 0 else "b"


def pairwise_agreement(
    scored: List[Tuple[Score, Score, str]],
    margin: float = 1e-6,
    *,
    force_choice: bool = False,
) -> Dict[str, float | int]:
    """Summarize pairwise ranking quality.

    ``strict_accuracy`` is the main proxy metric: model ties and human ties
    count as failures. ``agreement_rate`` is retained for compatibility, but
    only measures pairs where the model made a non-tied decision.

    ``force_choice`` matches the later human_pairs protocol (no 「差不多」):
    any nonzero difference picks a side. Tuple ranking keys compare
    lexicographically and only tie when every component is identical.
    """
    agree = disagree = model_tie = human_tie = 0
    for sa, sb, prefer in scored:
        if prefer == "tie":
            human_tie += 1
            continue
        choice = _model_prefers(sa, sb, margin=margin, force_choice=force_choice)
        if choice == "tie":
            model_tie += 1
            continue
        if choice == prefer:
            agree += 1
        else:
            disagree += 1
    decided = agree + disagree
    n = len(scored)
    return {
        "n": n,
        "agree": agree,
        "disagree": disagree,
        "model_tie": model_tie,
        "human_tie": human_tie,
        "agreement_rate": (agree / decided) if decided else 0.0,
        "strict_accuracy": (agree / n) if n else 0.0,
        "decided_coverage": (decided / n) if n else 0.0,
    }
