"""Matched-control sampling utilities for AL diagnostics.

These utilities are for evaluation only. They do not alter acquisition scores,
pool eligibility, or pool state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def sample_exact_matched_control(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    match_on: Sequence[str],
    rng: np.random.Generator,
    id_column: str = "obj_id",
) -> pd.DataFrame:
    """Sample a disjoint random control with exact discrete matching.

    For every stratum defined by ``match_on``, the returned control contains
    exactly as many rows as ``selected``. Selected object IDs are excluded
    before sampling.

    Parameters
    ----------
    candidates
        Complete candidate table from which controls may be sampled.
    selected
        Selected/acquired candidate table whose stratum distribution should
        be reproduced.
    match_on
        One or more discrete columns to match exactly, e.g. ``("observation_group",)``.
    rng
        NumPy random generator controlling reproducibility.
    id_column
        Stable object identifier used to prevent selected/control overlap.

    Returns
    -------
    pandas.DataFrame
        A control table containing the same number of rows as ``selected``
        and exactly matching its distribution over ``match_on``.

    Raises
    ------
    ValueError
        If required columns are missing, selected IDs are duplicated, or a
        selected stratum has too few eligible control candidates.
    """

    match_on = tuple(match_on)

    if not match_on:
        raise ValueError("match_on must contain at least one column.")

    required = {
        id_column,
        *match_on,
    }

    missing_candidates = sorted(required.difference(candidates.columns))

    if missing_candidates:
        raise ValueError(
            "Candidate table is missing required columns: " + ", ".join(missing_candidates)
        )

    missing_selected = sorted(required.difference(selected.columns))

    if missing_selected:
        raise ValueError(
            "Selected table is missing required columns: " + ", ".join(missing_selected)
        )

    if selected[id_column].duplicated().any():
        raise ValueError(f"Selected table contains duplicate {id_column!r} values.")

    selected_ids = set(selected[id_column].astype(str))

    eligible = candidates[~candidates[id_column].astype(str).isin(selected_ids)].copy()

    chosen_indices: list[int] = []

    grouped = selected.groupby(
        list(match_on),
        dropna=False,
        sort=True,
    )

    for raw_key, selected_group in grouped:
        key = (
            raw_key
            if isinstance(
                raw_key,
                tuple,
            )
            else (raw_key,)
        )

        mask = np.ones(
            len(eligible),
            dtype=bool,
        )

        for column, value in zip(
            match_on,
            key,
        ):
            if pd.isna(value):
                mask &= eligible[column].isna().to_numpy()
            else:
                mask &= eligible[column].eq(value).to_numpy()

        stratum_indices = eligible.index[mask].to_numpy()

        required_count = len(selected_group)

        if len(stratum_indices) < required_count:
            description = ", ".join(
                f"{column}={value!r}"
                for column, value in zip(
                    match_on,
                    key,
                )
            )

            raise ValueError(
                "Insufficient eligible controls for matched stratum "
                f"({description}): need {required_count}, "
                f"found {len(stratum_indices)}."
            )

        sampled = rng.choice(
            stratum_indices,
            size=required_count,
            replace=False,
        )

        chosen_indices.extend(int(index) for index in sampled)

    # Randomize presentation order while preserving the matched counts.
    chosen_indices = list(rng.permutation(chosen_indices))

    control = candidates.loc[chosen_indices].copy().reset_index(drop=True)

    if len(control) != len(selected):
        raise RuntimeError("Matched-control sampling returned an unexpected number of rows.")

    if set(control[id_column].astype(str)).intersection(selected_ids):
        raise RuntimeError("Matched control overlaps the selected object set.")

    selected_distribution = (
        selected.groupby(
            list(match_on),
            dropna=False,
        )
        .size()
        .sort_index()
    )

    control_distribution = (
        control.groupby(
            list(match_on),
            dropna=False,
        )
        .size()
        .sort_index()
    )

    if not selected_distribution.equals(control_distribution):
        raise RuntimeError(
            "Matched-control distribution does not exactly reproduce the selected distribution."
        )

    return control
