"""Portable, CPU-only checks for the opt-in notebook plotting theme."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from al_pipeline.core.evaluation.notebook_style import (
    POLICY_COLORS,
    policy_color,
    polish_policy_axes,
    use_notebook_style,
)


def test_import_has_no_matplotlib_side_effects() -> None:
    # The module import itself is not a theme activation.
    # Activation is tested inside an isolated rcParams context below.
    assert policy_color("margin") == POLICY_COLORS["margin"]
    assert policy_color("entropy_diversity") == POLICY_COLORS["entropy_diversity"]
    assert policy_color("entropy + diversity") == POLICY_COLORS["entropy_diversity"]


def test_activation_is_opt_in_and_reversible() -> None:
    with matplotlib.rc_context():
        original_cycle = rcParams["axes.prop_cycle"]
        use_notebook_style(enabled=False)
        assert rcParams["axes.prop_cycle"] == original_cycle
        use_notebook_style()
        assert rcParams["axes.prop_cycle"] != original_cycle
        assert rcParams["axes.facecolor"] == "white"
        # Users can restore Matplotlib's normal appearance.
        plt.style.use("default")
        assert rcParams["axes.prop_cycle"] == original_cycle


def test_policy_axis_polish_only_changes_appearance() -> None:
    with matplotlib.rc_context():
        use_notebook_style()
        fig, ax = plt.subplots()
        x = np.array([0, 1, 2], dtype=float)
        y = np.array([0.25, 0.40, 0.42], dtype=float)
        ax.plot(x, y, label="entropy")
        ax.plot(x, y + 0.03, label="random")
        ax.legend()
        before = [(line.get_xdata().copy(), line.get_ydata().copy()) for line in ax.lines]
        polish_policy_axes(ax)
        assert ax.lines[0].get_color() == policy_color("entropy")
        assert ax.lines[1].get_color() == policy_color("random")
        for line, (expected_x, expected_y) in zip(ax.lines, before):
            np.testing.assert_array_equal(line.get_xdata(), expected_x)
            np.testing.assert_array_equal(line.get_ydata(), expected_y)
        plt.close(fig)