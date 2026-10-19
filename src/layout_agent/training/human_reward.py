"""Human-gated learned reward delta.

The learned term is disabled by default and is only valid when a frozen
80-pair T2 / 15-item T1 gate proves that it beats the formula baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from layout_agent.models import EvalReport, LayoutState
from layout_agent.training.eval import CheckpointScorer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(report: Dict[str, Any], *keys: str) -> float:
    value: Any = report
    for key in keys:
        value = value[key]
    return float(value)


def build_human_reward_gate(
    model_report: Dict[str, Any],
    formula_report: Dict[str, Any],
    *,
    reward_checkpoint: Path,
    gold_manifest: Path,
    min_t1: int = 15,
    min_t2: int = 80,
    min_t1_gain: float = 0.02,
    min_t2_gain: float = 0.02,
) -> Dict[str, Any]:
    if min_t1 < 15 or min_t2 < 80:
        raise ValueError("human reward gate cannot weaken the 15-T1 / 80-T2 minimum")
    if min_t1_gain < 0 or min_t2_gain < 0:
        raise ValueError("human reward gains must be non-negative")
    checkpoint_hash = _sha256(reward_checkpoint)
    report_checkpoint = str(model_report.get("ckpt") or "")
    report_checkpoint_hash = str(model_report.get("checkpoint_sha256") or "")
    checkpoint_matches = (
        bool(report_checkpoint)
        and Path(report_checkpoint).resolve() == reward_checkpoint.resolve()
        and bool(report_checkpoint_hash)
        and report_checkpoint_hash == checkpoint_hash
    )
    manifest = json.loads(gold_manifest.read_text(encoding="utf-8"))
    labels = manifest.get("labels") or {}
    repo_root = (
        gold_manifest.resolve().parents[3]
        if len(gold_manifest.resolve().parents) >= 4
        else gold_manifest.resolve().parent
    )

    def label_hash(kind: str) -> str:
        raw = str(labels.get(kind) or "")
        if not raw:
            return ""
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        return _sha256(path) if path.is_file() else ""

    def input_hash(report: Dict[str, Any], kind: str) -> str:
        return str(
            (report.get("inputs", {}).get(kind, {}) or {}).get("sha256") or ""
        )

    expected_t1_hash = label_hash("t1")
    expected_t2_hash = label_hash("t2")
    model_t1_hash = input_hash(model_report, "t1")
    model_t2_hash = input_hash(model_report, "t2")
    formula_t1_hash = input_hash(formula_report, "t1")
    formula_t2_hash = input_hash(formula_report, "t2")
    model_t1_n = int(model_report.get("t1", {}).get("n", 0))
    formula_t1_n = int(formula_report.get("t1", {}).get("n", 0))
    model_t2_n = int(
        model_report.get("t2", {}).get("overall", {}).get("n", 0)
    )
    formula_t2_n = int(
        formula_report.get("t2", {}).get("overall", {}).get("n", 0)
    )
    model_t1 = _metric(model_report, "t1", "spearman")
    formula_t1 = _metric(formula_report, "t1", "spearman")
    model_t2 = _metric(model_report, "t2", "overall", "strict_accuracy")
    formula_t2 = _metric(formula_report, "t2", "overall", "strict_accuracy")
    checks = {
        "model_checkpoint_matches": checkpoint_matches,
        "gold_manifest_frozen": (
            manifest.get("status") == "human_labels_frozen"
            and bool(expected_t1_hash)
            and bool(expected_t2_hash)
        ),
        "same_t1_labels": (
            bool(expected_t1_hash)
            and model_t1_hash == formula_t1_hash == expected_t1_hash
        ),
        "same_t2_labels": (
            bool(expected_t2_hash)
            and model_t2_hash == formula_t2_hash == expected_t2_hash
        ),
        "t1_count": model_t1_n >= min_t1 and formula_t1_n >= min_t1,
        "t2_count": model_t2_n >= min_t2 and formula_t2_n >= min_t2,
        "t1_beats_formula": model_t1 >= formula_t1 + min_t1_gain,
        "t2_beats_formula": model_t2 >= formula_t2 + min_t2_gain,
    }
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_t1": min_t1,
            "min_t2": min_t2,
            "min_t1_gain": min_t1_gain,
            "min_t2_gain": min_t2_gain,
        },
        "metrics": {
            "model": {
                "t1_n": model_t1_n,
                "t1_spearman": model_t1,
                "t2_n": model_t2_n,
                "t2_strict_accuracy": model_t2,
            },
            "formula": {
                "t1_n": formula_t1_n,
                "t1_spearman": formula_t1,
                "t2_n": formula_t2_n,
                "t2_strict_accuracy": formula_t2,
            },
        },
        "reward_checkpoint": str(reward_checkpoint.resolve()),
        "reward_checkpoint_sha256": checkpoint_hash,
        "gold_manifest": str(gold_manifest.resolve()),
        "gold_manifest_sha256": _sha256(gold_manifest),
        "label_hashes": {
            "t1": expected_t1_hash,
            "t2": expected_t2_hash,
        },
    }


def require_human_reward_gate(
    gate_path: Path,
    checkpoint: Path,
) -> Dict[str, Any]:
    """Fail closed unless a passed, hash-matched gate exists for this checkpoint."""
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise ValueError(f"human reward gate did not pass: {gate_path}")
    expected_path = str(gate.get("reward_checkpoint") or "")
    if not expected_path or Path(expected_path).resolve() != Path(checkpoint).resolve():
        raise ValueError("human reward gate checkpoint path mismatch")
    expected_hash = str(gate.get("reward_checkpoint_sha256") or "")
    if not expected_hash or expected_hash != _sha256(checkpoint):
        raise ValueError("human reward checkpoint hash mismatch")
    if not gate.get("gold_manifest"):
        raise ValueError("human reward gate lacks frozen gold manifest")
    return gate


@dataclass
class GatedHumanReward:
    scorer: CheckpointScorer
    beta: float
    gate: Dict[str, Any]

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        gate_path: Path,
        *,
        beta: float,
        device: str | None = None,
    ) -> "GatedHumanReward":
        if beta <= 0:
            raise ValueError("human reward beta must be positive")
        gate = require_human_reward_gate(gate_path, checkpoint)
        manifest_raw = str(gate.get("gold_manifest") or "")
        if not manifest_raw:
            raise ValueError("human reward gate lacks frozen gold manifest")
        manifest_path = Path(manifest_raw)
        manifest_hash = str(gate.get("gold_manifest_sha256") or "")
        if not manifest_path.is_file() or _sha256(manifest_path) != manifest_hash:
            raise ValueError("human gold manifest changed after reward gating")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "human_labels_frozen":
            raise ValueError("human gold labels are not frozen")
        root = manifest_path.resolve().parents[3]
        label_hashes = gate.get("label_hashes") or {}
        if set(label_hashes) != {"t1", "t2"}:
            raise ValueError("human reward gate lacks frozen T1/T2 label hashes")
        for kind, expected in label_hashes.items():
            raw = str((manifest.get("labels") or {}).get(kind) or "")
            label_path = Path(raw)
            if not label_path.is_absolute():
                label_path = root / label_path
            if not expected or not label_path.is_file() or _sha256(label_path) != expected:
                raise ValueError(f"frozen human {kind} labels changed after gating")
        return cls(
            scorer=CheckpointScorer(checkpoint, device),
            beta=float(beta),
            gate=gate,
        )

    def delta(
        self,
        before: LayoutState,
        after: LayoutState,
        before_report: EvalReport,
        after_report: EvalReport,
    ) -> Tuple[float, Dict[str, float | bool]]:
        # Aesthetic quality can never compensate a hard violation.
        if before_report.hard_fail or after_report.hard_fail:
            return 0.0, {
                "human_reward_applied": False,
                "human_quality_before": 0.0,
                "human_quality_after": 0.0,
                "human_quality_delta": 0.0,
            }
        quality_before = self.scorer.score_state(before) / 100.0
        quality_after = self.scorer.score_state(after) / 100.0
        delta = quality_after - quality_before
        return self.beta * delta, {
            "human_reward_applied": True,
            "human_quality_before": quality_before,
            "human_quality_after": quality_after,
            "human_quality_delta": delta,
        }
