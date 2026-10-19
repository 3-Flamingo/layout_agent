#!/usr/bin/env python3
"""Merge reviewed CMU synth pairs into a second train file.

Does not change train.jsonl (real-only) or test.jsonl.

  train.jsonl              real pairs
  train_with_synth.jsonl   real + reviewed CMU perturbations
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset.train_variants import write_train_variants  # noqa: E402


def main() -> None:
    stats = write_train_variants(ROOT)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
