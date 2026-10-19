from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


Category = Literal["linear", "diameter", "radial", "chamfer"]
Side = Literal["L", "R", "U", "D"]


class Point(BaseModel):
    x: float
    y: float


class Box(BaseModel):
    min: Point
    max: Point


class ColorLegendEntry(BaseModel):
    id: str
    category: Category


class ColorLegend(BaseModel):
    color_space: str = "srgb"
    key_format: str = "hex_rrggbb"
    background: str = "000000"
    obstacle_reserved: List[str] = Field(default_factory=lambda: ["ffffff"])
    entries: Dict[str, ColorLegendEntry] = Field(default_factory=dict)


class RasterMeta(BaseModel):
    instance_image: str
    obstacle_image: Optional[str] = None
    width: int
    height: int
    origin: Point
    pixel_size: float
    y_down_image: bool = True
    anti_alias: bool = False


class LinearAnno(BaseModel):
    id: str
    category: Literal["linear"] = "linear"
    color: str
    p1: Point
    p2: Point
    dir: Point
    point: Point
    value: float
    arrow_width: float = 2.5
    text: Dict[str, Any] = Field(default_factory=dict)
    # CAD dimAngle (degrees). None / near-cardinal → axis-aligned placement.
    dim_angle: Optional[float] = None
    # CAD textPosition in view_local; when set, evaluation/render use it as text center.
    text_point: Optional[Point] = None
    dim_scale: float = 1.0
    # CAD text orientation flags (from override.text)
    dimtih: bool = False  # text inside horizontal
    dimtoh: bool = False  # text outside horizontal


class RadialAnno(BaseModel):
    id: str
    category: Literal["diameter", "radial", "chamfer"]
    color: str
    anchor: Point
    path: List[Point]
    text: Dict[str, Any] = Field(default_factory=dict)
    allow_jog: bool = True
    dim_scale: float = 1.0


class RoughnessAnno(BaseModel):
    """CadUserBlock name=roughness_down. Not a layout-movable instance."""

    id: str
    insert: Point
    rotation: float = 0.0
    scale: float = 1.0
    roughness_scale: float = 1.0
    value: str = ""
    style: int = 1
    text_height: float = 2.5
    triangle_side: float = 2.89
    prefix: str = "Rz"


class Obstacle(BaseModel):
    id: str
    kind: str = "polyline"
    raw_type: str = "projection"
    polyline: Optional[List[Point]] = None
    aabb: Optional[Box] = None
    fill_role: str = "loop"


class LayoutState(BaseModel):
    schema_version: str = "agent1.1.0"
    state_id: str
    created_at: str = ""
    source: Dict[str, str] = Field(default_factory=dict)
    units: str = "mm"
    coords: Dict[str, Any] = Field(default_factory=lambda: {"space": "view_local", "y_up": True})
    meta: Dict[str, Any] = Field(default_factory=dict)
    view: Dict[str, Any] = Field(default_factory=dict)
    raster: RasterMeta
    color_legend: ColorLegend
    obstacles: List[Obstacle] = Field(default_factory=list)
    linears: List[LinearAnno] = Field(default_factory=list)
    diameters: List[RadialAnno] = Field(default_factory=list)
    radials: List[RadialAnno] = Field(default_factory=list)
    roughness: List[RoughnessAnno] = Field(default_factory=list)
    linear_groups: List[Dict[str, Any]] = Field(default_factory=list)
    relations: List[Dict[str, Any]] = Field(default_factory=list)


class HardViolation(BaseModel):
    code: str
    a_id: str
    b_id: str
    detail: str = ""


class EvalReport(BaseModel):
    schema_version: str = "agent1.1.0"
    state_id: str
    hard_fail: bool = False
    hard_violations: List[HardViolation] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    side_density: Dict[str, float] = Field(default_factory=dict)
    time_ms: float = 0.0
    conflict_pairs: List[Tuple[str, str]] = Field(default_factory=list)


class ActionType(str, Enum):
    SET_LINEAR_SIDE = "SetLinearSide"
    SET_LINEAR_LAYER_ORDER = "SetLinearLayerOrder"
    BALANCE_LINEAR_SIDES = "BalanceLinearSides"
    SET_RADIAL_SECTOR = "SetRadialSector"
    REORDER_RADIAL = "ReorderRadial"
    ACCEPT = "Accept"
    REVERT = "Revert"


class Action(BaseModel):
    type: ActionType
    group_id: Optional[str] = None
    side: Optional[Side] = None
    target_id: Optional[str] = None
    sector: Optional[int] = None
    order: Optional[List[str]] = None
    reason: str = ""


class ScheduleDecision(str, Enum):
    """Brain / scheduler verdict after one optimize round."""

    CONTINUE = "Continue"  # enter next Decide round
    STOP_ACCEPT = "StopAccept"  # good enough
    STOP_NO_IMPROVE = "StopNoImprove"  # plateau
    STOP_REGRESS = "StopRegress"  # quality got worse
    STOP_MAX = "StopMaxSteps"


class ScheduleVerdict(BaseModel):
    decision: ScheduleDecision
    reason: str = ""
    loop_step: int = 0
    p_before: float = 0.0
    p_after: float = 0.0
    hard_fail_before: bool = False
    hard_fail_after: bool = False
    source: str = "rules"  # rules | llm


class AgentResult(BaseModel):
    improved: bool
    steps: int
    time_ms: float
    report_before: EvalReport
    report_after: Optional[EvalReport] = None
    actions_applied: List[List[Action]] = Field(default_factory=list)
    schedule_verdicts: List[ScheduleVerdict] = Field(default_factory=list)
    reason: str = ""
    state_before_id: str = ""
    state_after_id: str = ""
