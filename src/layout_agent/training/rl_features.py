"""Tensor features for variable-length layout action candidates."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from layout_agent.geometry_parts import collect_parts
from layout_agent.models import Action, ActionType, EvalReport, LayoutState


HARD_CODES: Tuple[str, ...] = (
    "H_TT",
    "H_TD",
    "H_DD",
    "H_TE",
    "H_TL",
    "H_OOB",
    "H_TIN",
    "H_TOB",
)
POLICY_ACTION_TYPES: Tuple[ActionType, ...] = (
    ActionType.ACCEPT,
    ActionType.BALANCE_LINEAR_SIDES,
    ActionType.SET_LINEAR_LAYER_ORDER,
    ActionType.SET_LINEAR_SIDE,
    ActionType.SET_RADIAL_SECTOR,
)
CATEGORIES = ("linear", "diameter", "radial", "chamfer")
SIDES = ("L", "R", "U", "D")

GLOBAL_FEATURE_DIM = 21
ACTION_FEATURE_DIM = 36


def action_key(action: Action) -> str:
    side = action.side or ""
    sector = "" if action.sector is None else str(int(action.sector))
    order = ",".join(action.order or [])
    return "|".join(
        (
            action.type.value,
            action.target_id or "",
            action.group_id or "",
            side,
            sector,
            order,
        )
    )


def dedupe_actions(actions: Iterable[Action]) -> List[Action]:
    out: List[Action] = []
    seen: set[str] = set()
    for action in actions:
        key = action_key(action)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def legal_bbox(state: LayoutState) -> Tuple[float, float, float, float]:
    bbox = state.view.get("legal_bbox") or state.view.get("bbox") or {}
    bmin = bbox.get("min") or {"x": 0.0, "y": 0.0}
    bmax = bbox.get("max") or {"x": 100.0, "y": 80.0}
    return (
        float(bmin.get("x", 0.0)),
        float(bmin.get("y", 0.0)),
        float(bmax.get("x", 100.0)),
        float(bmax.get("y", 80.0)),
    )


def ensure_legal_bbox(state: LayoutState) -> None:
    if state.view.get("legal_bbox"):
        return
    bbox = state.view.get("bbox") or {
        "min": {"x": 0.0, "y": 0.0},
        "max": {"x": 100.0, "y": 80.0},
    }
    state.view = {
        **state.view,
        "legal_bbox": json.loads(json.dumps(bbox)),
    }


def _squash(value: float, scale: float) -> float:
    return math.tanh(float(value) / max(float(scale), 1e-9))


def global_features(
    state: LayoutState,
    report: EvalReport,
    step: int,
    max_steps: int,
) -> np.ndarray:
    violation_counts: Dict[str, int] = {code: 0 for code in HARD_CODES}
    for violation in report.hard_violations:
        if violation.code in violation_counts:
            violation_counts[violation.code] += 1
    densities = report.side_density or {}
    frac = min(max(float(step) / max(int(max_steps), 1), 0.0), 1.0)
    values = [
        _squash(report.metrics.get("P", 0.0), 1000.0),
        _squash(report.metrics.get("V_hard", 0.0), 10.0),
        _squash(report.metrics.get("S_total", 0.0), 100.0),
        *[_squash(violation_counts[code], 5.0) for code in HARD_CODES],
        *[_squash(densities.get(side, 0.0), 10.0) for side in SIDES],
        _squash(len(state.linears), 30.0),
        _squash(len(state.diameters) + len(state.radials), 30.0),
        frac,
        1.0 - frac,
        1.0 if report.hard_fail else 0.0,
        1.0,
    ]
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (GLOBAL_FEATURE_DIM,):
        raise AssertionError(f"global feature shape {arr.shape}, expected {(GLOBAL_FEATURE_DIM,)}")
    return arr


def _one_hot(value: object, choices: Sequence[object]) -> List[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _current_side(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "L" if dx < 0 else "R"
    return "D" if dy < 0 else "U"


def action_features(
    state: LayoutState,
    report: EvalReport,
    actions: Sequence[Action],
) -> np.ndarray:
    xmin, ymin, xmax, ymax = legal_bbox(state)
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    parts = collect_parts(state)
    linears = {item.id: item for item in state.linears}
    radials = {item.id: item for item in list(state.diameters) + list(state.radials)}
    conflicts: Dict[str, Dict[str, int]] = {}
    for violation in report.hard_violations:
        for ident in (violation.a_id, violation.b_id):
            if ident in parts:
                per_code = conflicts.setdefault(ident, {code: 0 for code in HARD_CODES})
                if violation.code in per_code:
                    per_code[violation.code] += 1

    rows: List[List[float]] = []
    for action in actions:
        target_id = action.target_id or ""
        linear = linears.get(target_id)
        radial = radials.get(target_id)
        target = linear or radial
        category = target.category if target is not None else ""
        part = parts.get(target_id)
        if part is None:
            cx = cy = width = height = 0.0
        else:
            footprint = part.footprint()
            center = footprint.center
            cx = 2.0 * ((center.x - xmin) / span_x) - 1.0
            cy = 2.0 * ((center.y - ymin) / span_y) - 1.0
            width = min((footprint.xmax - footprint.xmin) / span_x, 2.0)
            height = min((footprint.ymax - footprint.ymin) / span_y, 2.0)
        if linear is not None:
            direction = (float(linear.dir.x), float(linear.dir.y))
            current_side = _current_side(*direction)
        elif radial is not None:
            end = radial.path[-1] if radial.path else radial.anchor
            dx, dy = end.x - radial.anchor.x, end.y - radial.anchor.y
            norm = math.hypot(dx, dy) or 1.0
            direction = (dx / norm, dy / norm)
            current_side = _current_side(*direction)
        else:
            direction = (0.0, 0.0)
            current_side = ""
        if action.sector is None:
            sector_features = [0.0, 0.0, 0.0]
        else:
            angle = 2.0 * math.pi * (int(action.sector) % 8) / 8.0
            sector_features = [math.sin(angle), math.cos(angle), (int(action.sector) % 8) / 7.0]
        per_code = conflicts.get(target_id, {})
        row = [
            *_one_hot(action.type, POLICY_ACTION_TYPES),
            *_one_hot(category, CATEGORIES),
            *_one_hot(action.side or "", SIDES),
            *sector_features,
            cx,
            cy,
            width,
            height,
            direction[0],
            direction[1],
            *_one_hot(current_side, SIDES),
            *[_squash(per_code.get(code, 0), 3.0) for code in HARD_CODES],
            1.0 if target is not None else 0.0,
            1.0 if action.type in {
                ActionType.ACCEPT,
                ActionType.BALANCE_LINEAR_SIDES,
                ActionType.SET_LINEAR_LAYER_ORDER,
            } else 0.0,
        ]
        if len(row) != ACTION_FEATURE_DIM:
            raise AssertionError(f"action feature width {len(row)}, expected {ACTION_FEATURE_DIM}")
        rows.append(row)
    if not rows:
        return np.zeros((0, ACTION_FEATURE_DIM), dtype=np.float32)
    return np.clip(np.asarray(rows, dtype=np.float32), -1.0, 1.0)


def pad_action_features(
    features: np.ndarray,
    max_actions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(features) > max_actions:
        raise ValueError(f"{len(features)} legal actions exceed max_actions={max_actions}")
    padded = np.zeros((max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((max_actions,), dtype=np.int8)
    if len(features):
        padded[: len(features)] = features
        mask[: len(features)] = 1
    return padded, mask


def geometry_fingerprint(state: LayoutState) -> str:
    """Hash action-relevant geometry; ignore raster paths and timestamps."""
    payload = {
        "legal_bbox": state.view.get("legal_bbox") or state.view.get("bbox"),
        "linears": [item.model_dump(mode="json") for item in state.linears],
        "diameters": [item.model_dump(mode="json") for item in state.diameters],
        "radials": [item.model_dump(mode="json") for item in state.radials],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
