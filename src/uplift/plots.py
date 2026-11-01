"""Shared chart styling and saving."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from uplift.config import FIGURES, SMD_MATERIAL, SMD_MINOR

# Okabe-Ito colorblind-safe palette.
COLORS = {
    "treatment": "#0072B2",
    "control": "#E69F00",
    "model": "#009E73",
    "random": "#999999",
    "flag": "#D55E00",
}


def save_fig(fig: plt.Figure, name: str, figures_dir: Path = FIGURES) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(figures_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    return figures_dir / f"{name}.png"


def love_plot(smd: pd.DataFrame, title: str, subtitle: str) -> plt.Figure:
    """Dot plot of |SMD| per feature with the minor and material thresholds."""
    data = smd.sort_values("abs_smd")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [COLORS["flag"] if v > SMD_MATERIAL else COLORS["treatment"] for v in data["abs_smd"]]
    ax.scatter(data["abs_smd"], data.index, color=colors, zorder=3)
    ax.axvline(SMD_MINOR, color=COLORS["random"], ls="--", lw=1, label=f"minor ({SMD_MINOR})")
    ax.axvline(
        SMD_MATERIAL, color=COLORS["flag"], ls="--", lw=1, label=f"material ({SMD_MATERIAL})"
    )
    ax.set_xlim(0, max(SMD_MATERIAL * 1.3, data["abs_smd"].max() * 1.1))
    ax.set_xlabel("|standardized mean difference| (treatment vs control)")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
