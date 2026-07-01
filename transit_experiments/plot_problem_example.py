from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch


BLUE = "#1677A8"
RED = "#D84A5B"
GREEN = "#2D7D46"
YELLOW = "#E3B624"
GRAY = "#666666"


def _node(ax, x: float, y: float, label: str, kind: str) -> None:
    if kind == "origin":
        face, edge, radius = "#DCEFD8", GREEN, 0.18
    elif kind == "destination":
        face, edge, radius = "#F7DDDF", RED, 0.18
    elif kind == "inactive":
        face, edge, radius = "#E1E1E1", "#999999", 0.09
    else:
        face, edge, radius = "#F6DF69", "#333333", 0.09
    ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge, lw=1.2, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, zorder=5)


def _arrow(ax, start, end, color, width, dashed=False, zorder=2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color,
            linestyle=(0, (3, 3)) if dashed else "solid",
            shrinkA=5,
            shrinkB=5,
            zorder=zorder,
        )
    )


def _panel(ax, redesigned: bool) -> None:
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.15, 2.6)
    ax.axis("off")

    title = "(b) Redesign plan" if redesigned else "(a) Existing service"
    ax.text(2.25, 2.48, title, ha="center", va="top", fontsize=10.5, fontweight="bold")

    stops = [(1.0, 1.25, "s1"), (2.25, 1.25, "s2"), (3.5, 1.25, "s3")]
    _arrow(ax, (0.95, 1.25), (3.55, 1.25), BLUE, 2.0 if not redesigned else 2.8)
    for idx, (x, y, label) in enumerate(stops):
        inactive = redesigned and idx == 1
        _node(ax, x, y, label, "inactive" if inactive else "stop")
        if inactive:
            ax.plot([x - 0.09, x + 0.09], [y - 0.09, y + 0.09], color=RED, lw=1.5, zorder=6)
            ax.plot([x - 0.09, x + 0.09], [y + 0.09, y - 0.09], color=RED, lw=1.5, zorder=6)

    _node(ax, 0.25, 2.05, "o1", "origin")
    _node(ax, 4.2, 2.05, "d1", "destination")
    _node(ax, 1.95, 0.35, "o2", "origin")
    _node(ax, 4.2, 0.35, "d2", "destination")

    _arrow(ax, (0.42, 2.05), (4.02, 2.05), RED, 3.0 if not redesigned else 2.2)
    _arrow(ax, (2.12, 0.35), (4.02, 0.35), RED, 2.7 if not redesigned else 2.0)
    _arrow(ax, (0.37, 1.92), (0.9, 1.35), GRAY, 1.0, dashed=True)
    _arrow(ax, (3.6, 1.35), (4.08, 1.92), GRAY, 1.0, dashed=True)
    if redesigned:
        _arrow(ax, (1.88, 0.52), (1.12, 1.14), RED, 1.1, dashed=True)
    else:
        _arrow(ax, (2.0, 0.52), (2.22, 1.14), GRAY, 1.0, dashed=True)
    _arrow(ax, (3.58, 1.15), (4.08, 0.48), GRAY, 1.0, dashed=True)

    if redesigned:
        ax.text(2.25, -0.02, "Transit: 100   |   Private: 100", ha="center", fontsize=9.3, fontweight="bold")
        ax.text(2.25, -0.14, "Lower congestion", ha="center", fontsize=8.8, color=GREEN)
    else:
        ax.text(2.25, -0.02, "Transit: 75   |   Private: 125", ha="center", fontsize=9.3, fontweight="bold")
        ax.text(2.25, -0.14, "High congestion", ha="center", fontsize=8.8, color=RED)


def main() -> None:
    output = Path(__file__).resolve().parent / "outputs" / "figures" / "problem_feedback_example.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), constrained_layout=False)
    _panel(axes[0], redesigned=False)
    _panel(axes[1], redesigned=True)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.96, bottom=0.22, wspace=0.08)

    legend = [
        Line2D([0], [0], color=BLUE, lw=2.5, label="Transit flow"),
        Line2D([0], [0], color=RED, lw=2.5, label="Private flow"),
        Line2D([0], [0], color=GRAY, lw=1.1, ls=(0, (3, 3)), label="Walk"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
