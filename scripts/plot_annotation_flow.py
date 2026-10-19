"""Flow figures for docs/标注流程.md."""

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


def _box(ax, x, y, w, h, text, fc, fontsize=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.08",
            facecolor=fc,
            edgecolor=INK,
            linewidth=1.15,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.35,
    )


def _diamond(ax, cx, cy, w, h, text, fc):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=INK, linewidth=1.15))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2, text="", dx=0.0, dy=0.08):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=ARROW,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", va="center", fontsize=8.5, color=MUTED)


def flow() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 72)
    ax.axis("off")
    ax.set_title("标注流程", fontsize=14, color=INK, pad=8)

    _box(ax, 36, 63, 40, 6.2, "陈震发放当月清单和视图 JSON", "#E8F1FB")
    _arrow(ax, 56, 63, 56, 57.2)
    _box(ax, 36, 50.2, 40, 6.2, "庚池打开工程图预览\n对照同一份视图 JSON", "#F3F7F4")
    _arrow(ax, 56, 50.2, 56, 44.4)
    _diamond(ax, 56, 38.2, 28, 11.2, "本月这条是哪种记录", "#FFF6E8")

    _box(ax, 4, 22.5, 22, 8.2, "问题标签\ntags.jsonl", "#E7F5EE")
    _box(ax, 31, 22.5, 22, 8.2, "偏好对\npairs.jsonl", "#E7F5EE")
    _box(ax, 58, 22.5, 22, 8.2, "验收图\nacceptance.jsonl", "#E7F0FA")
    _box(ax, 85, 22.5, 23, 8.2, "复核或复标\nreviews / relabel", "#FDECEC")

    _arrow(ax, 46, 34.2, 15, 30.9, "10–12 月")
    _arrow(ax, 52, 33.4, 42, 30.9, "10–12 月")
    _arrow(ax, 62, 33.4, 69, 30.9, "仅 12 月")
    _arrow(ax, 66, 34.2, 96, 30.9, "仅 1 月")

    _arrow(ax, 56, 22.5, 56, 16.6)
    _box(ax, 34, 9.6, 44, 6.4, "陈震抽检，并用评价器读入视图 JSON", "#E8F1FB")
    _arrow(ax, 56, 9.6, 56, 4.8)
    _diamond(ax, 56, 0.2, 18, 8.4, "合格？", "#FFF6E8")
    # diamond centered too low; redraw layout with more room
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.4, 8.4))
    ax.set_xlim(0, 114)
    ax.set_ylim(0, 84)
    ax.axis("off")

    _box(ax, 34, 75.2, 46, 6.2, "1  陈震发放当月清单和视图 JSON", "#E8F1FB")
    _arrow(ax, 57, 75.2, 57, 69.6)
    _box(ax, 34, 62.6, 46, 6.4, "2  庚池看工程图预览\n并核对视图 JSON 是同一布局", "#F3F7F4")
    _arrow(ax, 57, 62.6, 57, 56.8)
    _diamond(ax, 57, 50.2, 34, 12, "3  本月这条记录写进哪个文件", "#FFF6E8")

    _box(ax, 2, 34.6, 24, 8, "问题标签\n10–12 月\ntags.jsonl", "#E7F5EE", fontsize=9.5)
    _box(ax, 30, 34.6, 24, 8, "偏好对\n10–12 月\npairs.jsonl", "#E7F5EE", fontsize=9.5)
    _box(ax, 58, 34.6, 24, 8, "验收图\n仅 12 月\nacceptance.jsonl", "#E7F0FA", fontsize=9.5)
    _box(ax, 86, 34.6, 26, 8, "复核 / 复标\n仅 1 月\nreviews、relabel", "#FDECEC", fontsize=9.5)

    _arrow(ax, 44, 45.4, 14, 42.8)
    _arrow(ax, 52, 44.8, 42, 42.8)
    _arrow(ax, 64, 44.8, 70, 42.8)
    _arrow(ax, 72, 45.4, 99, 42.8)

    _arrow(ax, 57, 34.6, 57, 28.2)
    _box(ax, 32, 21.2, 50, 6.4, "4  陈震抽检，评价器读入视图 JSON", "#E8F1FB")
    _arrow(ax, 57, 21.2, 57, 15.4)
    _diamond(ax, 57, 10.2, 22, 9.6, "5  合格？", "#FFF6E8")
    _arrow(ax, 68, 10.2, 92, 10.2, "是")
    _box(ax, 92, 7.2, 18, 6, "计入当月数量", "#E7F5EE", fontsize=9.5)
    _arrow(ax, 46, 10.2, 22, 10.2, "否")
    _box(ax, 2, 7.2, 20, 6, "退回，庚池修改后再交", "#FDECEC", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "annotation_flow.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tag_decision() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 9.2))
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 92)
    ax.axis("off")

    _box(ax, 31, 83.5, 46, 6, "打开一张工程图预览", "#F3F7F4")
    _arrow(ax, 54, 83.5, 54, 77.6)
    _diamond(ax, 54, 70.2, 36, 13, "有没有可见硬冲突", "#FFF6E8")
    _arrow(ax, 36, 70.2, 20, 70.2, "有")
    _box(ax, 1, 66, 18, 8.4, "acceptable = false\n勾上对应代码", "#FDECEC", fontsize=9)
    _arrow(ax, 72, 70.2, 88, 70.2, "没有")
    _box(ax, 88, 66, 18, 8.4, "acceptable = true", "#E7F5EE", fontsize=9.5)

    _arrow(ax, 10, 66, 10, 58.5)
    _arrow(ax, 97, 66, 97, 58.5)
    _arrow(ax, 10, 58.5, 54, 58.5)
    _arrow(ax, 97, 58.5, 54, 58.5)
    _arrow(ax, 54, 58.5, 54, 54.2)

    _diamond(ax, 54, 47.2, 32, 12.4, "有没有软问题", "#FFF6E8")
    _arrow(ax, 38, 47.2, 20, 47.2, "有")
    _box(ax, 1, 43, 18, 8.4, "勾上 overcrowd\n或 radial_clash", "#FFF6E8", fontsize=8.5)
    _arrow(ax, 70, 47.2, 88, 47.2, "没有")
    _box(ax, 88, 43, 18, 8.4, "软问题留空", "#F3F7F4", fontsize=9.5)

    _arrow(ax, 10, 43, 10, 35.5)
    _arrow(ax, 97, 43, 97, 35.5)
    _arrow(ax, 10, 35.5, 54, 35.5)
    _arrow(ax, 97, 35.5, 54, 35.5)
    _arrow(ax, 54, 35.5, 54, 29.6)

    _box(ax, 18, 22.6, 72, 6.4, "尺寸线相交、蹭界线另记，不改变 acceptable", "#FFF6E8")
    _arrow(ax, 54, 22.6, 54, 16.2)
    _box(ax, 24, 8.6, 60, 7, "写入当月 tags.jsonl\n12 月验收图再填写 expected", "#E7F0FA")
    fig.tight_layout()
    fig.savefig(OUT / "annotation_tag_decision.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flow()
    tag_decision()
    print("wrote", OUT)
