"""Architecture figures for docs/布局优化工作总报告.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimSun"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
    }
)

INK = "#1F2933"
MUTED = "#52606D"
ARROW = "#334155"


def _box(ax, x, y, w, h, text, fc, fontsize=9.5, ec=INK, lw=1.15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, linespacing=1.25)
    return patch


def _diamond(ax, cx, cy, w, h, text, fc):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=INK, linewidth=1.15, joinstyle="round"))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9, color=INK, linespacing=1.2)


def _arrow(ax, x1, y1, x2, y2, text="", text_offset=(0, 0.12), color=ARROW):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=color,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
        )
    )
    if text:
        ax.text((x1 + x2) / 2 + text_offset[0], (y1 + y2) / 2 + text_offset[1], text, ha="center", va="center", fontsize=8, color=MUTED)


def _elbow(ax, points, color=ARROW):
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        last = i == len(points) - 2
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>" if last else "-",
                mutation_scale=11,
                linewidth=1.15,
                color=color,
                shrinkA=0,
                shrinkB=0,
            )
        )


def system_loop() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0.2, 6.6)
    ax.axis("off")

    blue, green, amber = "#E4EEF8", "#E5F4EA", "#FFF3DE"
    stop_c, purple = "#F8E6E6", "#F1E9F8"

    _box(ax, 0.15, 5.15, 1.6, 0.82, "视图状态", blue)
    _box(ax, 2.05, 5.15, 1.75, 0.82, "感知\n画出实例图", blue)
    _box(ax, 4.1, 5.02, 2.15, 1.08, "评价\n硬违规、软代价、公式代价", green, fontsize=9)
    _arrow(ax, 1.75, 5.56, 2.05, 5.56)
    _arrow(ax, 3.8, 5.56, 4.1, 5.56)

    _diamond(ax, 7.55, 5.56, 2.0, 1.22, "已满足\n图纸需求？", "#FFF8D6")
    _arrow(ax, 6.25, 5.56, 6.55, 5.56)
    _box(ax, 9.35, 5.12, 2.15, 0.88, "停止\n输出当前布局", stop_c)
    _arrow(ax, 8.55, 5.56, 9.35, 5.56)
    ax.text(8.95, 5.78, "是", fontsize=9, color=MUTED, ha="center")

    _box(ax, 6.45, 3.15, 2.2, 0.88, "决策\n选择一个摆放动作", amber)
    _arrow(ax, 7.55, 4.95, 7.55, 4.03)
    ax.text(7.72, 4.45, "否", fontsize=9, color=MUTED)
    ax.text(5.15, 4.55, "无硬违规，且高权重软约束已清掉", fontsize=8, color=MUTED, ha="center")

    _box(ax, 3.75, 3.15, 2.15, 0.88, "执行\n改坐标并重画", blue)
    _arrow(ax, 6.45, 3.59, 5.9, 3.59)
    _box(ax, 0.9, 3.08, 2.3, 1.02, "再评价\n本步之后的约束和代价", green, fontsize=9)
    _arrow(ax, 3.75, 3.59, 3.2, 3.59)

    _diamond(ax, 2.05, 1.35, 2.3, 1.25, "调度\n还要再决策吗？", purple)
    _elbow(ax, [(2.05, 3.08), (2.05, 1.98)])
    ax.text(2.18, 2.45, "本步前后的评价", fontsize=8, color=MUTED, ha="left")

    _elbow(ax, [(3.2, 1.7), (6.9, 1.7), (6.9, 3.15)], color="#7C3AED")
    ax.text(5.0, 1.88, "继续：再进一轮决策", fontsize=8, color="#6D28D9", ha="center")

    _box(ax, 4.55, 0.28, 2.45, 0.82, "停止\n保留目前最好的布局", stop_c, fontsize=9)
    _arrow(ax, 2.55, 0.72, 4.55, 0.72)
    ax.text(3.5, 0.95, "变差、无改进或步数用尽", fontsize=8, color=MUTED, ha="center")

    _elbow(ax, [(2.7, 1.85), (11.85, 1.85), (11.85, 5.56), (11.5, 5.56)], color="#B45309")
    ax.text(8.3, 2.05, "已满足，或硬违规刚刚消除", fontsize=8, color="#9A3412", ha="center")

    fig.savefig(OUT / "system_loop.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stage(ax, x, y, w, h, title, shape, fc):
    _box(ax, x, y, w, h, "", fc, ec="#94A3B8", lw=1.0)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(x + w / 2, y + h * 0.28, shape, ha="center", va="center", fontsize=7.5, color=MUTED)


def residual_arch() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.3))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    img_c, feat_c, head_c, add_c = "#E4EEF8", "#FFF3DE", "#F3EEF8", "#E5F4EA"
    ax.text(0.15, 5.95, "图像支路  ·  ResNet18，ImageNet 预训练；每个残差块是两个 3×3 卷积加跳连", fontsize=9, color=MUTED)
    ax.text(0.15, 2.85, "特征支路", fontsize=9, color=MUTED)

    stages = [
        (0.15, 1.55, "输入", "5 × 256 × 256"),
        (1.9, 1.55, "7×7 卷积, s2\n3×3 最大池化", "64 × 64 × 64"),
        (3.65, 1.55, "残差块 ×2", "64 × 64 × 64"),
        (5.4, 1.55, "残差块 ×2\n步长 2", "128 × 32 × 32"),
        (7.15, 1.55, "残差块 ×2\n步长 2", "256 × 16 × 16"),
        (8.9, 1.55, "残差块 ×2\n步长 2", "512 × 8 × 8"),
        (10.65, 1.55, "全局平均池化", "512"),
    ]
    y_img, h = 4.15, 1.45
    for i, (x, w, title, shape) in enumerate(stages):
        _stage(ax, x, y_img, w, h, title, shape, img_c)
        if i:
            prev_x, prev_w = stages[i - 1][0], stages[i - 1][1]
            _arrow(ax, prev_x + prev_w, y_img + h / 2, x, y_img + h / 2)

    _stage(ax, 0.15, 0.85, 2.15, 1.15, "16 维公式特征", "硬违规、代价、密度、散布", feat_c)
    _stage(ax, 2.7, 0.85, 1.7, 1.15, "全连接 + ReLU", "32", feat_c)
    _stage(ax, 4.8, 0.85, 1.7, 1.15, "全连接 + ReLU", "32", feat_c)
    _arrow(ax, 2.3, 1.42, 2.7, 1.42)
    _arrow(ax, 4.4, 1.42, 4.8, 1.42)

    _stage(ax, 7.15, 1.55, 1.55, 1.15, "拼接", "544", head_c)
    _elbow(ax, [(11.42, 4.15), (11.42, 3.45), (7.92, 3.45), (7.92, 2.7)])
    ax.text(9.6, 3.62, "512", fontsize=8, color=MUTED, ha="center")
    _elbow(ax, [(6.5, 1.42), (7.9, 1.42), (7.9, 1.55)])
    ax.text(7.15, 1.22, "32", fontsize=8, color=MUTED, ha="left")

    _stage(ax, 9.05, 1.55, 1.85, 1.15, "全连接，初始化为 0", "残差项  1", head_c)
    _arrow(ax, 8.7, 2.12, 9.05, 2.12)

    cx, cy, r = 11.55, 2.12, 0.28
    ax.add_patch(plt.Circle((cx, cy), r, facecolor=add_c, edgecolor=INK, linewidth=1.15, zorder=3))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=14, color=INK, zorder=4)
    _arrow(ax, 10.9, 2.12, cx - r, cy)

    _stage(ax, 10.85, 0.35, 1.4, 0.85, "公式分", "标量", "#FDE7D4")
    _arrow(ax, 11.55, 1.2, 11.55, cy - r, color="#C2410C")
    ax.text(12.15, 2.12, "排序分", fontsize=11, color=INK, ha="left", va="center")
    _arrow(ax, cx + r, cy, 12.05, cy)

    fig.savefig(OUT / "residual_arch.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    system_loop()
    residual_arch()
    print(OUT)
