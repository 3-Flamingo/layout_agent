#!/usr/bin/env python3
"""Generate stage-review artifacts from existing evalsys data (no new labels).

Outputs under data/evalsys/showcase/:
  index.html     items 1+2+3+5 (perturb wall, model scores, agent before/after, curves)
  rollout.html   item 4 (RL env greedy rollout GIF)
  gradcam.html   item 6 (Grad-CAM overlays)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "service_ai_layout_agent" / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from layout_agent.config import load_config
from layout_agent.decide import DecisionEngine
from layout_agent.evaluate import Evaluator
from layout_agent.evalset import render_preview
from layout_agent.execute import Executor
from layout_agent.models import Action, ActionType, LayoutState
from layout_agent.perceive import Perceiver
from layout_agent.training.data import StateEncoder, encode_channels, _obstacle_for
from layout_agent.training.env import LayoutEnv, enumerate_actions
from layout_agent.training.eval import CheckpointScorer
from layout_agent.training.model import RewardModel
from score_adapt import build_score_card

OUT = ROOT / "data" / "evalsys" / "showcase"
PAIRS = ROOT / "data" / "evalsys" / "pairs_perturb"
CKPT = ROOT / "runs" / "reward_v0" / "best.pt"
HISTORY = ROOT / "runs" / "reward_v0" / "history.json"
EVAL_JSON = ROOT / "runs" / "reward_v0" / "eval_construct.json"

BG = (22, 24, 28)
FG = (232, 232, 232)
MUTED = (150, 154, 162)
ACCENT = (86, 160, 255)
GOOD = (90, 200, 130)
BAD = (232, 92, 92)


def _font(px: int) -> ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        windir / "msyh.ttc",
        windir / "msyh.ttf",
        windir / "msyhbd.ttc",
        windir / "simhei.ttf",
        windir / "simsun.ttc",
        Path("arial.ttf"),
        Path("segoeui.ttf"),
        Path("DejaVuSans.ttf"),
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(str(p), px)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = im.size
    s = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if s < 1:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


def _caption_bar(w: int, lines: List[Tuple[str, Tuple[int, int, int]]], h: int = 36) -> Image.Image:
    bar = Image.new("RGB", (w, h * len(lines)), BG)
    dr = ImageDraw.Draw(bar)
    f = _font(16)
    for i, (txt, col) in enumerate(lines):
        dr.text((10, i * h + h // 2), txt, fill=col, font=f, anchor="lm")
    return bar


def _side_by_side(
    left: Image.Image,
    right: Image.Image,
    cap_l: str,
    cap_r: str,
    *,
    right_worse: bool = True,
) -> Image.Image:
    left, right = _fit(left, 720, 420), _fit(right, 720, 420)
    h = max(left.height, right.height)
    gap, pad = 16, 12
    canvas = Image.new("RGB", (left.width + right.width + gap + pad * 2, h + 44 + pad * 2), BG)
    canvas.paste(left, (pad, pad + 36))
    canvas.paste(right, (pad + left.width + gap, pad + 36))
    dr = ImageDraw.Draw(canvas)
    f = _font(18)
    dr.text((pad + left.width // 2, pad + 16), cap_l, fill=GOOD if right_worse else MUTED, font=f, anchor="mm")
    dr.text(
        (pad + left.width + gap + right.width // 2, pad + 16),
        cap_r,
        fill=BAD if right_worse else GOOD,
        font=f,
        anchor="mm",
    )
    return canvas


def _load_pairs() -> List[Dict[str, Any]]:
    recs = []
    for line in PAIRS.joinpath("manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def _compact(rec: Dict[str, Any], max_px: int = 2200) -> bool:
    ra = PAIRS / rec["render_a"]
    if not ra.exists():
        return False
    with Image.open(ra) as im:
        return max(im.size) <= max_px


def pick_wall(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """3 pairs per level with largest formula ΔP among compact renders."""
    chosen: List[Dict[str, Any]] = []
    used_views = set()
    for lv in ("light", "medium", "heavy"):
        cands = [
            r
            for r in recs
            if r.get("level") == lv and _compact(r)
            and r["view_key"] not in used_views
        ]
        cands.sort(key=lambda r: float(r["meta"]["P_b"]) - float(r["meta"]["P_a"]), reverse=True)
        for r in cands[:3]:
            chosen.append(r)
            used_views.add(r["view_key"])
    return chosen


def overlay_score(im: Image.Image, title: str, formula: str, model: Optional[float]) -> Image.Image:
    im = _fit(im, 800, 480)
    bar_h = 52
    out = Image.new("RGB", (im.width, im.height + bar_h), BG)
    out.paste(im, (0, bar_h))
    dr = ImageDraw.Draw(out)
    dr.rectangle([0, 0, im.width, bar_h], fill=(32, 36, 42))
    dr.text((12, 14), title, fill=FG, font=_font(16))
    extra = formula if model is None else f"{formula}   模型 {model:.1f}/100"
    dr.text((12, 34), extra, fill=MUTED, font=_font(13))
    return out


# ---------------------------------------------------------------------------
# 1 + 2  perturbation wall + model scores
# ---------------------------------------------------------------------------

def build_wall_and_scores(scorer: CheckpointScorer, recs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    wall_dir = OUT / "wall"
    score_dir = OUT / "model_scores"
    wall_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    wall_recs = pick_wall(recs)
    wall_out = []
    for rec in wall_recs:
        a = Image.open(PAIRS / rec["render_a"]).convert("RGB")
        b = Image.open(PAIRS / rec["render_b"]).convert("RGB")
        sa = scorer.score_state_file(PAIRS / rec["state_a"])
        sb = scorer.score_state_file(PAIRS / rec["state_b"])
        pa, pb = rec["meta"]["P_a"], rec["meta"]["P_b"]
        va, vb = rec["meta"]["V_hard_a"], rec["meta"]["V_hard_b"]
        left = overlay_score(a, "原图（优）", f"P={pa:.0f}  硬冲突={va:.0f}", sa)
        right = overlay_score(b, "扰动（劣）", f"P={pb:.0f}  硬冲突={vb:.0f}", sb)
        combo = _side_by_side(left, right, "原图", f"扰动 · {rec['level']}")
        fn = f"{rec['pair_id']}.png"
        combo.save(wall_dir / fn)
        wall_out.append(
            {
                "file": f"wall/{fn}",
                "pair_id": rec["pair_id"],
                "level": rec["level"],
                "ops": rec.get("ops") or [],
                "P_a": pa,
                "P_b": pb,
                "score_a": sa,
                "score_b": sb,
                "agree": sa is not None and sb is not None and sa > sb,
            }
        )

    # model-score demo: pick 4 compact pairs where the model clearly prefers A
    scored = []
    for rec in recs:
        if rec.get("level") != "heavy" or not _compact(rec, 1800):
            continue
        sa = scorer.score_state_file(PAIRS / rec["state_a"])
        sb = scorer.score_state_file(PAIRS / rec["state_b"])
        if sa is None or sb is None:
            continue
        scored.append((sa - sb, rec, sa, sb))
    scored.sort(key=lambda t: t[0], reverse=True)
    demo = []
    seen = set()
    for d, rec, sa, sb in scored:
        if rec["view_key"] in seen or d < 0.3:
            continue
        seen.add(rec["view_key"])
        a = Image.open(PAIRS / rec["render_a"]).convert("RGB")
        b = Image.open(PAIRS / rec["render_b"]).convert("RGB")
        left = overlay_score(a, "A  原图", f"公式 P={rec['meta']['P_a']:.0f}", sa)
        right = overlay_score(b, "B  扰动", f"公式 P={rec['meta']['P_b']:.0f}", sb)
        combo = _side_by_side(left, right, f"模型 {sa:.1f}", f"模型 {sb:.1f}")
        fn = f"{rec['pair_id']}.png"
        combo.save(score_dir / fn)
        demo.append(
            {
                "file": f"model_scores/{fn}",
                "pair_id": rec["pair_id"],
                "score_a": sa,
                "score_b": sb,
                "delta": sa - sb,
                "ops": rec.get("ops") or [],
            }
        )
        if len(demo) >= 4:
            break
    return wall_out, demo


# ---------------------------------------------------------------------------
# 3  agent before / after
# ---------------------------------------------------------------------------

def _improves(cand, cur) -> bool:
    mc, mb = cand.metrics, cur.metrics
    vh_c, vh_b = float(mc.get("V_hard", 0)), float(mb.get("V_hard", 0))
    if vh_c != vh_b:
        return vh_c < vh_b
    return float(mc.get("P", 0)) < float(mb.get("P", 0)) - 1e-6


def run_agent(state: LayoutState, work: Path) -> Tuple[LayoutState, Any, Any, List[str]]:
    cfg = load_config(ROOT / "config" / "default.yaml")
    cfg.setdefault("decide", {})
    cfg["decide"]["use_llm"] = False
    cfg["decide"]["global_plan"] = True
    cfg["decide"]["max_plan_actions"] = 16
    perceiver = Perceiver(cfg)
    evaluator = Evaluator(cfg)
    decision = DecisionEngine(cfg)
    executor = Executor(cfg, perceiver)
    work.mkdir(parents=True, exist_ok=True)
    inst = Path(state.raster.instance_image or "")
    if inst and not inst.is_absolute():
        for base in (ROOT / "data" / "datasets" / "demo", ROOT / "data" / "datasets"):
            cand = base / inst
            if cand.exists():
                state.raster.instance_image = str(cand)
                obs = Path(state.raster.obstacle_image or "")
                if obs and not obs.is_absolute() and (base / obs).exists():
                    state.raster.obstacle_image = str(base / obs)
                inst = cand
                break
    if not inst.exists():
        state = perceiver.render_icm(
            state, work / f"{state.state_id}.instance.png", work / f"{state.state_id}.obstacle.png"
        )
    before = executor.snapshot(state)
    report0 = evaluator.evaluate(before)
    after, report = before, report0
    notes: List[str] = []
    for rnd in range(3):
        acts = decision.decide(after, report)
        geo = [a for a in acts if a.type not in (ActionType.ACCEPT, ActionType.REVERT)]
        kept = 0
        for act in geo:
            cand, note = executor.apply(after, [act], work)
            crep = evaluator.evaluate(cand)
            if _improves(crep, report):
                after, report = cand, crep
                notes.append(note)
                kept += 1
        if kept == 0:
            break
    return after, report0, report, notes


def stacked_state() -> LayoutState:
    def lin(i, x, on_edge=False):
        d = {
            "id": f"dim{i}",
            "color": f"{0x10 + i:02x}0000",
            "p1": {"x": 60, "y": 40},
            "p2": {"x": 60, "y": 110},
            "dir": {"x": 0, "y": 1},
            "point": {"x": x, "y": 75},
            "value": 70,
            "text": {"width": 10, "height": 4, "content": str(70 - i)},
        }
        if on_edge:
            d["text_point"] = {"x": 60, "y": 75}
        return d

    raw = {
        "schema_version": "agent1.1.0",
        "state_id": "showcase_stack",
        "units": "mm",
        "view": {"bbox": {"min": {"x": 0, "y": 0}, "max": {"x": 200, "y": 150}}},
        "raster": {
            "instance_image": "x.png",
            "obstacle_image": "y.png",
            "width": 400,
            "height": 300,
            "origin": {"x": 0, "y": 0},
            "pixel_size": 0.5,
            "y_down_image": True,
        },
        "color_legend": {"entries": {}},
        "obstacles": [
            {
                "id": "part1",
                "kind": "polyline",
                "raw_type": "projection",
                "fill_role": "loop",
                "polyline": [
                    {"x": 60, "y": 40},
                    {"x": 140, "y": 40},
                    {"x": 140, "y": 110},
                    {"x": 60, "y": 110},
                ],
            }
        ],
        "linears": [lin(1, 40, True), lin(2, 40.5), lin(3, 41), lin(4, 41.5), lin(5, 42)],
        "diameters": [],
        "radials": [],
    }
    return LayoutState.model_validate(raw)


def build_agent_cases() -> List[Dict[str, Any]]:
    adir = OUT / "agent"
    adir.mkdir(parents=True, exist_ok=True)
    cases = []
    demo_conflict = ROOT / "data" / "datasets" / "demo" / "snapshots" / "case_conflict.state.json"
    demo_crowd = ROOT / "data" / "datasets" / "demo" / "snapshots" / "case_crowded.state.json"
    sources = [
        ("5 条尺寸堆叠（合成）", stacked_state()),
        ("demo · 冲突", LayoutState.model_validate(json.loads(demo_conflict.read_text(encoding="utf-8")))),
        ("demo · 过密", LayoutState.model_validate(json.loads(demo_crowd.read_text(encoding="utf-8")))),
    ]
    for title, st in sources:
        work = adir / (st.state_id or "v")
        after, r0, r1, notes = run_agent(st, work)
        c0, c1 = build_score_card(r0), build_score_card(r1)
        before_state = st
        pb = work / "before.png"
        pa = work / "after.png"
        render_preview(before_state, pb)
        render_preview(after, pa)
        left = overlay_score(
            Image.open(pb).convert("RGB"),
            "布局前",
            f"硬冲突 {int(r0.metrics.get('V_hard', 0))}  raw_P={r0.metrics.get('P', 0):.0f}",
            c0["score"],
        )
        right = overlay_score(
            Image.open(pa).convert("RGB"),
            "布局后（守护式全局规划）",
            f"硬冲突 {int(r1.metrics.get('V_hard', 0))}  raw_P={r1.metrics.get('P', 0):.0f}",
            c1["score"],
        )
        combo = _side_by_side(
            left, right, f"{c0['score']:.1f}/100", f"{c1['score']:.1f}/100", right_worse=False
        )
        fn = f"{st.state_id}.png"
        combo.save(adir / fn)
        cases.append(
            {
                "title": title,
                "file": f"agent/{fn}",
                "score_before": c0["score"],
                "score_after": c1["score"],
                "v_before": r0.metrics.get("V_hard", 0),
                "v_after": r1.metrics.get("V_hard", 0),
                "p_before": r0.metrics.get("P", 0),
                "p_after": r1.metrics.get("P", 0),
                "notes": notes[:8],
            }
        )
        print(f"  agent {title}: {c0['score']:.1f} → {c1['score']:.1f}  notes={notes}")
    return cases


# ---------------------------------------------------------------------------
# 5  training curves
# ---------------------------------------------------------------------------

def build_curves() -> Dict[str, Any]:
    hist = json.loads(HISTORY.read_text(encoding="utf-8"))
    eval_rep = json.loads(EVAL_JSON.read_text(encoding="utf-8")) if EVAL_JSON.exists() else {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return _curves_pil(hist, eval_rep)

    epochs = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), facecolor="#16181c")
    for ax in axes:
        ax.set_facecolor("#1c1f26")
        ax.tick_params(colors="#c8c8c8")
        for sp in ax.spines.values():
            sp.set_color("#3a3f4a")
        ax.yaxis.label.set_color("#c8c8c8")
        ax.xaxis.label.set_color("#c8c8c8")
        ax.title.set_color("#e8e8e8")

    axes[0].plot(epochs, [h["loss"] for h in hist], color="#56a0ff", marker="o", ms=3)
    axes[0].set_title("Training loss (Bradley-Terry)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")

    axes[1].plot(epochs, [h["val"]["all"] for h in hist], color="#e8e8e8", label="all", lw=2)
    for lv, col in (("light", "#e85c5c"), ("medium", "#ffd644"), ("heavy", "#5ac882")):
        axes[1].plot(epochs, [h["val"].get(lv, 0) for h in hist], color=col, label=lv)
    axes[1].axhline(0.5, color="#666", ls="--", lw=0.8, label="random")
    axes[1].set_title("Val pairwise agreement by perturbation level")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("agreement")
    axes[1].set_ylim(0.2, 1.0)
    axes[1].legend(facecolor="#1c1f26", edgecolor="#3a3f4a", labelcolor="#e8e8e8", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

    # bar of best-epoch per level
    best = max(hist, key=lambda h: h["val"]["all"])
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor="#16181c")
    ax.set_facecolor("#1c1f26")
    ax.tick_params(colors="#c8c8c8")
    for sp in ax.spines.values():
        sp.set_color("#3a3f4a")
    labels = ["all", "light", "medium", "heavy"]
    vals = [best["val"][k] for k in labels]
    colors = ["#e8e8e8", "#e85c5c", "#ffd644", "#5ac882"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0.5, color="#666", ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("agreement")
    ax.set_title(f"Best checkpoint (epoch {best['epoch']}) val agreement")
    ax.title.set_color("#e8e8e8")
    ax.yaxis.label.set_color("#c8c8c8")
    fig.tight_layout()
    fig.savefig(OUT / "curves_bar.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"history": hist, "best": best, "eval": eval_rep}


def _curves_pil(hist, eval_rep) -> Dict[str, Any]:
    # fallback unused if matplotlib present
    best = max(hist, key=lambda h: h["val"]["all"])
    return {"history": hist, "best": best, "eval": eval_rep}


# ---------------------------------------------------------------------------
# 4  RL rollout GIF
# ---------------------------------------------------------------------------

def _greedy_action(env: LayoutEnv) -> Optional[int]:
    """Try each legal action on a snapshot; pick the one that most lowers P."""
    best_i, best_p = None, float(env.report.metrics.get("P", 0.0))
    acts = env.legal_actions
    # skip Accept unless nothing else helps
    for i, act in enumerate(acts):
        if act.type == ActionType.ACCEPT:
            continue
        try:
            _, report, _, _ = env.simulate_action(i)
            p = float(report.metrics.get("P", 0.0))
        except Exception:
            continue
        if p < best_p - 1e-6:
            best_p, best_i = p, i
    if best_i is None:
        for i, act in enumerate(acts):
            if act.type == ActionType.ACCEPT:
                return i
    return best_i


def build_rollout() -> Dict[str, Any]:
    rdir = OUT / "rollout"
    rdir.mkdir(parents=True, exist_ok=True)
    st = stacked_state()
    tmp = rdir / "work"
    tmp.mkdir(exist_ok=True)
    env = LayoutEnv([ROOT / "data" / "datasets" / "demo" / "snapshots" / "case_conflict.state.json"], work_dir=tmp)
    # override: use stacked synthetic for a clearer story
    env.reset(options={"state": st, "index": 0})

    frames: List[Image.Image] = []
    log: List[Dict[str, Any]] = []
    for step in range(8):
        preview = rdir / f"step_{step:02d}.png"
        render_preview(env.state, preview)
        p_now = float(env.report.metrics.get("P", 0.0))
        v_hard = float(env.report.metrics.get("V_hard", 0.0))
        im = overlay_score(
            Image.open(preview).convert("RGB"),
            f"step {step}   P={p_now:.1f}",
            "greedy rollout（每步选使 P 下降最多的合法动作）",
            None,
        )
        frames.append(_fit(im, 900, 560))
        log.append({"step": step, "P": p_now, "v_hard": v_hard})
        idx = _greedy_action(env)
        if idx is None:
            break
        act = env.legal_actions[idx]
        if act.type == ActionType.ACCEPT:
            log[-1]["action"] = "Accept"
            break
        obs, r, terminated, truncated, info = env.step(idx)
        log[-1]["action"] = info.get("action", {}).get("type")
        log[-1]["reward"] = r
        if terminated or truncated:
            break

    # last frame after last step
    preview = rdir / f"step_{len(log):02d}.png"
    render_preview(env.state, preview)
    p_end = float(env.report.metrics.get("P", 0.0))
    frames.append(
        overlay_score(
            Image.open(preview).convert("RGB"),
            f"end   P={p_end:.1f}",
            "rollout finished",
            None,
        )
    )
    gif = rdir / "rollout.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=900, loop=0)
    # also a strip
    w = sum(f.width for f in frames) + 8 * (len(frames) - 1)
    h = max(f.height for f in frames)
    strip = Image.new("RGB", (w, h), BG)
    x = 0
    for f in frames:
        strip.paste(_fit(f, f.width, h), (x, 0))
        x += f.width + 8
    strip.save(rdir / "strip.png")
    return {"gif": "rollout/rollout.gif", "strip": "rollout/strip.png", "log": log}


# ---------------------------------------------------------------------------
# 6  Grad-CAM
# ---------------------------------------------------------------------------

def _gradcam(model: RewardModel, x, device) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    acts: List = []
    grads: List = []

    def fwd(_m, _i, o):
        acts.append(o)

    def bwd(_m, _gi, go):
        grads.append(go[0])

    h1 = model.net.layer4.register_forward_hook(fwd)
    h2 = model.net.layer4.register_full_backward_hook(bwd)
    model.zero_grad(set_to_none=True)
    logit = model(x)
    logit.sum().backward()
    h1.remove()
    h2.remove()
    A = acts[0].detach()  # [1,C,h,w]
    G = grads[0].detach()
    w = G.mean(dim=(2, 3), keepdim=True)
    cam = (w * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def _heatmap_overlay(base: Image.Image, cam: np.ndarray) -> Image.Image:
    base = _fit(base.convert("RGB"), 640, 480)
    heat = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize(base.size, Image.BILINEAR)
    # red-yellow colormap
    arr = np.asarray(heat).astype(np.float32) / 255.0
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(arr * 255 * 1.4, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((arr - 0.35) * 255 * 1.6, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((0.25 - arr) * 180, 0, 80).astype(np.uint8)
    heat_rgb = Image.fromarray(rgb, "RGB")
    return Image.blend(base, heat_rgb, 0.45)


def build_gradcam(scorer: CheckpointScorer, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import torch

    gdir = OUT / "gradcam"
    gdir.mkdir(parents=True, exist_ok=True)
    model = scorer.model
    model.eval()
    device = scorer.device
    out = []
    # use the same 4 scored demos + a couple of wall heavies
    cands = [r for r in recs if r.get("level") == "heavy" and _compact(r, 1800)]
    cands.sort(key=lambda r: float(r["meta"]["P_b"]) - float(r["meta"]["P_a"]), reverse=True)
    seen = set()
    picked = []
    for r in cands:
        if r["view_key"] in seen:
            continue
        seen.add(r["view_key"])
        picked.append(r)
        if len(picked) >= 4:
            break
    for rec in picked:
        state = LayoutState.model_validate(json.loads((PAIRS / rec["state_b"]).read_text(encoding="utf-8")))
        arr = scorer.encoder.encode(state, base_dir=PAIRS)
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        x.requires_grad_(True)
        cam = _gradcam(model, x, device)
        preview = Image.open(PAIRS / rec["render_b"]).convert("RGB")
        overlay = _heatmap_overlay(preview, cam)
        orig = _fit(preview, 640, 480)
        combo = _side_by_side(orig, overlay, "扰动后布局", "Grad-CAM（模型关注区域）")
        fn = f"{rec['pair_id']}.png"
        combo.save(gdir / fn)
        score = scorer.score_state_file(PAIRS / rec["state_b"])
        out.append({"file": f"gradcam/{fn}", "pair_id": rec["pair_id"], "ops": rec.get("ops") or [], "score": score})
        print(f"  gradcam {rec['pair_id']}")
    return out


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
body{margin:0;background:#16181c;color:#e8e8e8;font-family:"Segoe UI",sans-serif;line-height:1.55}
main{max-width:1100px;margin:0 auto;padding:28px 24px 80px}
h1{font-size:26px;font-weight:600;margin:0 0 8px}
h2{font-size:20px;margin:40px 0 10px;border-bottom:1px solid #2a2e36;padding-bottom:8px}
h3{font-size:15px;color:#bbb;font-weight:500}
p,li{color:#c8c8c8;font-size:14px}
.kicker{color:#8ab4f8;font-size:12px;letter-spacing:.08em;text-transform:uppercase}
.card{background:#1c1f26;border:1px solid #2a2e36;border-radius:8px;padding:16px;margin:14px 0}
img{max-width:100%;border-radius:6px;border:1px solid #2a2e36;background:#000}
.meta{font-size:12px;color:#888}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #2a2e36;padding:6px 8px;text-align:left}
th{color:#aaa;font-weight:500}
.ok{color:#5ac882} .bad{color:#e85c5c} .warn{color:#ffd644}
.note{background:#22262e;padding:12px 14px;border-left:3px solid #56a0ff;margin:12px 0;font-size:13px;color:#bbb}
"""


def write_index(wall, demos, agents, curves) -> None:
    best = curves["best"]
    ev = curves.get("eval") or {}
    t2 = (ev.get("t2") or {}).get("overall") or {}
    rows = ""
    for w in wall:
        agree = '<span class="ok">模型同意原图更好</span>' if w["agree"] else '<span class="warn">模型未拉开 / 方向相反</span>'
        sa = f'{w["score_a"]:.1f}' if w["score_a"] is not None else "—"
        sb = f'{w["score_b"]:.1f}' if w["score_b"] is not None else "—"
        rows += (
            f'<div class="card"><h3>{w["level"]} · {w["pair_id"]}</h3>'
            f'<p class="meta">扰动：{", ".join(w["ops"]) or "—"} · 公式 P {w["P_a"]:.0f}→{w["P_b"]:.0f} · '
            f'模型 {sa} vs {sb} · {agree}</p>'
            f'<img src="{w["file"]}" alt="{w["pair_id"]}"></div>'
        )
    demo_html = ""
    for d in demos:
        demo_html += (
            f'<div class="card"><h3>{d["pair_id"]} · Δ={d["delta"]:.1f}</h3>'
            f'<p class="meta">扰动：{", ".join(d["ops"])}</p>'
            f'<img src="{d["file"]}"></div>'
        )
    agent_html = ""
    for a in agents:
        notes = "、".join(a["notes"]) or "无动作被保留"
        agent_html += (
            f'<div class="card"><h3>{a["title"]}</h3>'
            f'<p>综合分 <b class="bad">{a["score_before"]:.1f}</b> → '
            f'<b class="ok">{a["score_after"]:.1f}</b>　'
            f'硬冲突 {a["v_before"]:.0f} → {a["v_after"]:.0f}　'
            f'raw_P {a["p_before"]:.0f} → {a["p_after"]:.0f}</p>'
            f'<p class="meta">保留动作：{notes}</p>'
            f'<img src="{a["file"]}"></div>'
        )
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>评估系统阶段性成果</title><style>{CSS}</style></head><body><main>
<p class="kicker">layout_agent · 2026-08-26 · 无新标注数据</p>
<h1>评估系统阶段性成果</h1>
<p>在生产 dump 管道与人工 T1/T2 标签到位之前，用现有 82 个去重视图、366 个扰动偏好对、v0 奖励模型和已上线的守护式规划，把能看见的进展摊开。数字只证明管线通，<b>不是最终效果结论</b>。</p>

<div class="note">数据基础：扰动对 train 299 / val 67；v0 为 ResNet18、256²、冻结前两段、CPU 训练 14 epoch 早停。
公式评估器在构造标签上一致率 100%（训练对已按公式过滤）；学习模型的真实对比要等人工标签。</div>

<h2>1. 扰动对比墙 — 数据管线已经能量产带标签样本</h2>
<p>同一视图：左=现网/原布局，右=分级随机扰动。扰动算子包括文字点抖动、尺寸线滑移、同轴分层互换、径向引线端点抖动。生成时丢弃了公式判定「扰动反而更好」的 124 对歧义样本。</p>
{rows}

<h2>2. 模型打分演示 — learned_quality 已经能输出 0–100 分</h2>
<p>v0 奖励模型对「原图 vs 重度扰动」打分。分数来自 sigmoid(logit)×100。挑的是模型分差较大的紧凑视图，用来展示接口形态，不是泛化证明。</p>
{demo_html}

<h2>3. Agent 优化前后 — 守护式全局规划已上线</h2>
<p>一次性建议改为「全局动作批 + 逐动作复评只保留变好的 + 最多 3 轮细化」。布局后综合分不会低于布局前。下面三例在本机用当前内核当场跑出。</p>
{agent_html}

<h2>5. 训练曲线 — 重度扰动已学会，轻度是瓶颈</h2>
<p>best checkpoint：epoch {best['epoch']}，val 一致率 all={best['val']['all']:.2f}，
light={best['val']['light']:.2f}，medium={best['val']['medium']:.2f}，heavy={best['val']['heavy']:.2f}。
loss 继续下降而 val 不再升，是 299 对喂 ResNet18 的过拟合，等 R1/R2 千对级数据后再放开分辨率和冻结。</p>
<div class="card"><img src="curves.png" alt="training curves"></div>
<div class="card"><img src="curves_bar.png" alt="agreement bar"></div>
<table>
<tr><th>集合</th><th>一致率</th><th>说明</th></tr>
<tr><td>val（训练时，epoch {best['epoch']}）</td><td>{best['val']['all']:.2f}</td><td>early-stop 依据</td></tr>
<tr><td>全部 366 对构造标签（含训练集）</td><td>{(t2.get('agreement_rate') or 0):.2f}</td>
<td>agree {t2.get('agree','—')} / disagree {t2.get('disagree','—')} / model_tie {t2.get('model_tie','—')}；含训练集，不能当效果</td></tr>
</table>
<p class="meta">Source: runs/reward_v0/history.json · eval_construct.json · 2026-08-24 训练</p>
</main></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def write_rollout(info: Dict[str, Any]) -> None:
    rows = "".join(
        f"<tr><td>{x.get('step')}</td><td>{x.get('action','')}</td>"
        f"<td>{x.get('P',0):.1f}</td><td>{x.get('v_hard',0):.0f}</td>"
        f"<td>{x.get('reward','')}</td></tr>"
        for x in info["log"]
    )
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>RL 环境 rollout</title><style>{CSS}</style></head><body><main>
<p class="kicker">item 4 · 单独产出</p>
<h1>RL 仿真环境 greedy rollout</h1>
<p>环境是 <code>LayoutEnv</code>（Executor + Evaluator）。每步枚举合法离散动作，选使公式代价 P 下降最多的一步（不是已训练的策略）。用来说明：决策侧强化学习的仿真器已经闭环，剩下的是采样与 PPO。</p>
<div class="note">奖励 = ΔP/100 − 0.05 步数惩罚；Accept 且硬约束清零时 +2 终局奖励。learned_quality 未接入奖励，避免 hack 未校准的学习维度。</div>
<div class="card"><h3>动画</h3><img src="{info['gif']}" alt="rollout gif"></div>
<div class="card"><h3>分帧条带</h3><img src="{info['strip']}" alt="strip"></div>
<table><tr><th>step</th><th>动作</th><th>P</th><th>V_hard</th><th>reward</th></tr>{rows}</table>
</main></body></html>"""
    (OUT / "rollout.html").write_text(html, encoding="utf-8")


def write_gradcam(items: List[Dict[str, Any]]) -> None:
    cards = ""
    for it in items:
        sc = f'{it["score"]:.1f}' if it.get("score") is not None else "—"
        cards += (
            f'<div class="card"><h3>{it["pair_id"]} · 模型分 {sc}</h3>'
            f'<p class="meta">扰动：{", ".join(it["ops"])}</p>'
            f'<img src="{it["file"]}"></div>'
        )
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Grad-CAM</title><style>{CSS}</style></head><body><main>
<p class="kicker">item 6 · 单独产出</p>
<h1>奖励模型 Grad-CAM</h1>
<p>对 v0 ResNet18 的 <code>layer4</code> 做 Grad-CAM，叠在扰动后的布局预览上。热区表示「拉高质量 logit 的空间位置」。若落在文字穿线、重叠、引线拉飞处，说明模型在看布局问题而不是背景噪声。</p>
<div class="note">v0 只在 366 对构造标签上训练，热图仅作可解释性示意，不能外推到生产终稿。</div>
{cards}
</main></body></html>"""
    (OUT / "gradcam.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading scorer…")
    scorer = CheckpointScorer(CKPT, "cpu")
    recs = _load_pairs()
    print("1+2 wall + model scores…")
    wall, demos = build_wall_and_scores(scorer, recs)
    print("3 agent cases…")
    agents = build_agent_cases()
    print("5 curves…")
    curves = build_curves()
    write_index(wall, demos, agents, curves)
    print("wrote", OUT / "index.html")
    print("4 rollout…")
    roll = build_rollout()
    write_rollout(roll)
    print("wrote", OUT / "rollout.html")
    print("6 gradcam…")
    gc = build_gradcam(scorer, recs)
    write_gradcam(gc)
    print("wrote", OUT / "gradcam.html")
    print("done.")


if __name__ == "__main__":
    main()
