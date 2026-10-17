"""Internal numeric and candidate-alignment guards for batch acquisition."""

from __future__ import annotations

import numpy as np

from al_pipeline.core.pool import Pool
from al_pipeline.core.selection import TopKSelector


def feature_matrix(value: np.ndarray, *, name: str, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] < 1 or (array.shape[0] == 0 and not allow_empty):
        raise ValueError(f"{name} must have shape (n_objects, n_features), with positive feature width")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be real and numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=np.float64)


def eligible_candidates(
    ids: list[str] | np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
    pool: Pool,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_array = np.asarray(ids, dtype=object)
    scores_array = np.asarray(scores, dtype=float)
    if ids_array.ndim != 1 or ids_array.size != expected_rows:
        raise ValueError("Candidate IDs must align row-for-row with acquisition representations")
    TopKSelector._validate_inputs(ids=ids_array, scores=scores_array, k=k)
    eligible = np.flatnonzero([name in pool.unlabeled_ids for name in ids_array])
    if k > len(eligible):
        raise ValueError(f"Cannot select {k} objects: only {len(eligible)} eligible candidates")
    return ids_array, scores_array, eligible


def squared_distances(features: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    delta = features - anchor
    result = np.einsum("ij,ij->i", delta, delta)
    if not np.all(np.isfinite(result)):
        raise ValueError("Feature distances overflowed: scale the representations first")
    return result


def validate_scores(actual: np.ndarray, expected: np.ndarray, *, method: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-10):
        raise ValueError(
            f"{method} selector expects scores from its matching scorer on "
            "the same ordered candidate data"
        )


def validated_ids(values, *, expected_rows: int, name: str) -> tuple[str, ...]:
    ids = tuple(values)
    if len(ids) != expected_rows:
        raise ValueError(f"{name} must align with corresponding representation rows")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{name} must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return ids
