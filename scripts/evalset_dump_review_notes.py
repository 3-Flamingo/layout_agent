#!/usr/bin/env python3
"""Dump review.html localStorage notes from Chrome Local Storage.

Writes data/evalsys/human_pairs/review_notes_relayout.json
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evalsys" / "human_pairs" / "review_notes_relayout.json"
NEEDLE = "notes_review_human_pairs"


def _profile_leveldb_dirs() -> list[Path]:
    home = Path.home()
    bases = [
        home / "AppData/Local/Google/Chrome/User Data",
        home / "AppData/Local/Microsoft/Edge/User Data",
        home / "AppData/Local/Chromium/User Data",
    ]
    found: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for p in base.glob("*/Local Storage/leveldb"):
            if p.is_dir():
                found.append(p)
    return found


def _copy_unlocked(src: Path) -> Path:
    dst = Path(tempfile.mkdtemp(prefix="chrome_ls_"))
    for p in src.iterdir():
        if p.name.upper() == "LOCK":
            continue
        try:
            shutil.copy2(p, dst / p.name)
        except OSError:
            continue
    return dst


def extract_notes(leveldb_dir: Path) -> Optional[Tuple[Dict[str, Any], int]]:
    try:
        from chromium_reader.localstorage import LocalStorageReader
    except ImportError:
        return None
    best: Optional[Tuple[int, str]] = None
    with LocalStorageReader(leveldb_dir) as reader:
        for rec in reader.records():
            if rec.script_key != NEEDLE:
                continue
            seq = int(rec.leveldb_seq_number or 0)
            val = rec.value or ""
            if not val.startswith("{"):
                continue
            if best is None or seq >= best[0]:
                best = (seq, val)
    if not best:
        return None
    notes = json.loads(best[1])
    if not isinstance(notes, dict):
        return None
    return notes, best[0]


def main() -> None:
    dirs = _profile_leveldb_dirs()
    if not dirs:
        raise SystemExit("no Chrome/Edge Local Storage dirs found")
    best_notes: Optional[Dict[str, Any]] = None
    best_meta = (-1, -1, "")
    tmp_dirs: list[Path] = []
    try:
        for folder in dirs:
            copied = _copy_unlocked(folder)
            tmp_dirs.append(copied)
            got = extract_notes(copied)
            if not got:
                continue
            notes, seq = got
            n_relayout = sum(1 for k in notes if str(k).startswith("relayout_"))
            score = (n_relayout, seq)
            if score > best_meta[:2]:
                best_notes = notes
                best_meta = (n_relayout, seq, str(folder))
    finally:
        for folder in tmp_dirs:
            shutil.rmtree(folder, ignore_errors=True)
    if not best_notes:
        raise SystemExit(
            "could not parse notes_review_human_pairs from Chrome/Edge; "
            "pip install chromium-reader and reopen review.html"
        )
    n_relayout, seq, src = best_meta
    if n_relayout <= 0:
        raise SystemExit("parsed notes but found no relayout_* ids")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(best_notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} keys={len(best_notes)} relayout={n_relayout} seq={seq} from {src}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        raise
