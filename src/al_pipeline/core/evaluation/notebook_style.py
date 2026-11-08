"""Optional, presentation-inspired Matplotlib styling for example notebooks.

This module changes *presentation only*: no acquisition policies, metrics,
model training, or datasets are imported. Nothing happens on import. Call
``use_notebook_style()`` in a notebook to opt in; comment that call out or
use ``plt.style.use('default')`` to return to Matplotlib defaults.

The muted blue/teal/sage/lavender/blush palette is adapted from the
AppleCiDEr training-school presentation. It uses common installed fonts,
so it works on laptops, remote Jupyter servers, and headless CI machines.
"""
from __future__ import annotations

from matplotlib import rcParams
from matplotlib.axes import Axes
from cycler import cycler


# Stable semantic colors: an acquisition policy retains its hue in a learning
# curve, bar plot, or later astronomy notebook, regardless of plotting order.
POLICY_COLORS = {
    "random": "#98B5A6",               # muted sage
    "entropy": "#83B6CD",              # dusty sky blue
    "margin": "#519C9A",               # muted teal
    "entropy_diversity": "#AE9DC7",    # lavender
    "balanced_exploration": "#DEA2AD", # blush
    "reference": "#B7A588",            # sand
}

PRESENTATION_CYCLE = tuple(POLICY_COLORS.values())

PRESENTATION_RC = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.prop_cycle": cycler(color=PRESENTATION_CYCLE),
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"],
    "font.size": 10,
    "text.color": "#344047",
    "axes.labelcolor": "#5D6B74",
    "axes.titlecolor": "#344047",
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "axes.titlepad": 12,
    "figure.titlesize": 13,
    "figure.titleweight": "medium",
    "xtick.color": "#6F7A82",
    "ytick.color": "#6F7A82",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.edgecolor": "#DDE6E9",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": "#E5EBEE",
    "grid.alpha": 0.75,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.0,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 115,
    "savefig.dpi": 150,
}


def use_notebook_style(enabled: bool = True) -> None:
    """Opt in to the reusable notebook visual theme, without altering data.

    Calling this updates Matplotlib defaults in the *current kernel* only.
    To opt out, comment the call before running the notebook or run
    ``import matplotlib.pyplot as plt; plt.style.use('default')``.
    """
    if enabled:
        rcParams.update(PRESENTATION_RC)


def policy_color(name: str) -> str:
    """Return the stable color for a known acquisition policy.

    Unrecognized names use an unobtrusive slate instead of failing.
    """
    canonical = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "entropy_+_diversity": "entropy_diversity",
        "diversity": "entropy_diversity",
        "balanced": "balanced_exploration",
        "balanced_explore": "balanced_exploration",
        "full_data_reference": "reference",
    }
    return POLICY_COLORS.get(aliases.get(canonical, canonical), "#77858D")


def polish_policy_axes(ax: Axes, *, show_legend: bool = True) -> Axes:
    """Polish an existing plot, coloring series by their policy labels.

    Works with the repository's existing ``plot_strategy_comparison`` helper;
    preserves every line's data, limits, tick positions, and annotations.
    """
    for line in ax.lines:
        label = line.get_label()
        if isinstance(label, str) and not label.startswith("_"):
            line.set_color(policy_color(label))
            line.set_marker("o")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E5EBEE", linewidth=0.7, alpha=0.8)
    ax.grid(False, axis="x")
    if show_legend and ax.get_legend() is not None:
        ax.legend(frameon=False, loc="best", fontsize=8, ncol=1)
    return ax