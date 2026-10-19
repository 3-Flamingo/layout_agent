"""Random train/test split for human preference pairs.

Test is a uniform sample of real modified pairs, not the high-delta subset.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence

SPLIT_SEED = 202609182
SPLIT_N_TEST = 170
SPLIT_TEST_RATIO = 0.20
SPLIT_METHOD = "random"


def n_test_from_all(
    n_real: int,
    n_other: int = 0,
    *,
    ratio: float = SPLIT_TEST_RATIO,
) -> int:
    """Test size is a fraction of *all* pairs, but the test rows themselves are real.

    ``n_other`` is typically the synthetic count. The returned size never exceeds
    ``n_real``.
    """
    n_real = max(0, int(n_real))
    n_all = n_real + max(0, int(n_other))
    n_test = int(round(float(ratio) * n_all))
    return max(0, min(n_real, n_test))


def random_test_keys(
    view_keys: Iterable[str],
    *,
    n_test: int = SPLIT_N_TEST,
    seed: int = SPLIT_SEED,
) -> List[str]:
    """Deterministic shuffle, then take the first ``n_test`` unique keys."""
    uniq = sorted({str(key) for key in view_keys if key})
    if not uniq:
        return []
    rng = random.Random(seed)
    rng.shuffle(uniq)
    return uniq[: min(int(n_test), len(uniq))]


def assign_splits(
    rows: Sequence[dict],
    test_keys: Iterable[str],
) -> tuple[List[dict], List[dict]]:
    frozen = {str(key) for key in test_keys if key}
    test_rows: List[dict] = []
    train_rows: List[dict] = []
    for rec in rows:
        item = dict(rec)
        key = str(item.get("view_key") or "")
        if key in frozen:
            item["split"] = "test"
            test_rows.append(item)
        else:
            item["split"] = "train"
            train_rows.append(item)
    test_rows.sort(key=lambda r: str(r.get("pair_id") or ""))
    train_rows.sort(key=lambda r: str(r.get("pair_id") or ""))
    return test_rows, train_rows
