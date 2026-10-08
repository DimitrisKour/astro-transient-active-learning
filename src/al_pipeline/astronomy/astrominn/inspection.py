"""Utilities for inspecting real-data active-learning candidates.

These helpers are intentionally diagnostic. They visualize already-prepared
candidate samples and caller-supplied ZTF alert metadata for human inspection. They do not affect candidate eligibility, model
inference, acquisition scoring, or pool state.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
    load_astrominn_vit_config,
    preprocess_astrominn_vit,
)


CHANNEL_NAMES = (
    "Science",
    "Template",
    "Difference",
)


def load_candidate_table(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate a candidate-inspection CSV."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Candidate table does not exist: {path}")

    frame = pd.read_csv(
        path,
        dtype={
            "obj_id": str,
            "source_reference": str,
        },
    )

    required = {
        "obj_id",
        "alert",
        "acquisition_score",
        "filepath",
        "source_reference",
    }

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError("Candidate table is missing required columns: " + ", ".join(missing))

    return frame


def _format_value(
    value: Any,
) -> str:
    if value is None:
        return "—"

    if isinstance(
        value,
        float,
    ):
        if np.isnan(value):
            return "—"

        if abs(value) >= 1e4:
            return f"{value:.3e}"

        return f"{value:.4g}"

    return str(value)


def _metadata_text(
    row: pd.Series,
    *,
    metadata_fields: Sequence[str],
) -> str:
    lines = [
        f"Object: {row['obj_id']}",
        f"Alert: {int(row['alert'])}",
    ]

    if "rank" in row.index:
        lines.append(f"Rank: {int(row['rank'])}")

    if "group" in row.index:
        lines.append(f"Group: {row['group']}")

    lines.append(f"Acquisition score: {float(row['acquisition_score']):.4f}")

    if "predicted_class" in row.index:
        lines.append(f"Prediction: {row['predicted_class']}")

    probability_columns = [column for column in row.index if column.startswith("p_")]

    if probability_columns:
        lines.append("")
        lines.append("Class probabilities")

        for column in probability_columns:
            lines.append(f"  {column[2:]}: {float(row[column]):.3f}")

    native_available = "candid" in row.index and pd.notna(row.get("candid"))

    if native_available:
        lines.append("")
        lines.append("Native ZTF")

        if "nalerthist" in row.index:
            lines.append("  nalerthist: " + _format_value(row.get("nalerthist")))

        for field in metadata_fields:
            if field in row.index:
                lines.append(f"  {field}: " + _format_value(row.get(field)))

    return "\n".join(lines)


def plot_candidate_table(
    candidates: pd.DataFrame,
    *,
    max_objects: int | None = None,
    metadata_fields: Sequence[str] = (),
    show: bool = True,
) -> list[plt.Figure]:
    """Plot raw/model-input cutouts plus acquisition and alert metadata.

    One figure is produced per candidate. No filtering or candidate scoring
    occurs here; this function is solely for post-selection inspection.
    """

    frame = candidates.copy()

    if max_objects is not None:
        if max_objects < 0:
            raise ValueError("max_objects must be >= 0.")

        frame = frame.head(max_objects)

    config = load_astrominn_vit_config()

    figures: list[plt.Figure] = []

    for _, row in frame.iterrows():
        sample_path = Path(row["filepath"])

        if not sample_path.exists():
            raise FileNotFoundError(f"Candidate sample does not exist: {sample_path}")

        sample = np.load(
            sample_path,
            allow_pickle=True,
        ).item()

        raw_image = np.asarray(
            sample["image"],
            dtype=np.float32,
        )

        _, processed_image = preprocess_astrominn_vit(
            metadata=np.asarray(sample["metadata"]),
            image=raw_image,
            config=config,
        )

        processed_image = processed_image.detach().cpu().numpy()

        figure = plt.figure(
            figsize=(13, 6),
            constrained_layout=True,
        )

        grid = figure.add_gridspec(
            2,
            4,
            width_ratios=[
                1,
                1,
                1,
                1.25,
            ],
        )

        for channel, name in enumerate(CHANNEL_NAMES):
            raw_ax = figure.add_subplot(
                grid[
                    0,
                    channel,
                ]
            )

            model_ax = figure.add_subplot(
                grid[
                    1,
                    channel,
                ]
            )

            raw_ax.imshow(
                raw_image[channel],
                origin="lower",
                cmap="gray",
            )

            raw_ax.set_title(f"Raw {name}")

            model_ax.imshow(
                processed_image[channel],
                origin="lower",
                cmap="gray",
            )

            model_ax.set_title(f"ViT input {name}")

            raw_ax.axis("off")

            model_ax.axis("off")

        text_ax = figure.add_subplot(
            grid[
                :,
                3,
            ]
        )

        text_ax.axis("off")

        text_ax.text(
            0.0,
            1.0,
            _metadata_text(
                row,
                metadata_fields=metadata_fields,
            ),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=text_ax.transAxes,
        )

        title = str(row["obj_id"])

        if "group" in row.index:
            title += f" — {row['group']}"

        figure.suptitle(title)

        figures.append(figure)

        if show:
            plt.show()
            plt.close(figure)

    return figures


def plot_candidate_csv(
    path: str | Path,
    *,
    max_objects: int | None = None,
    metadata_fields: Sequence[str] = (),
    show: bool = True,
) -> list[plt.Figure]:
    """Load a saved selection/control CSV and visualize its candidates."""

    candidates = load_candidate_table(path)

    return plot_candidate_table(
        candidates,
        max_objects=max_objects,
        metadata_fields=metadata_fields,
        show=show,
    )
