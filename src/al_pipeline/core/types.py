"""Model-independent prediction, selection, and label batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np


ClassificationOutputType: TypeAlias = Literal[
    "logits",
    "probabilities",
]


@dataclass(frozen=True, slots=True)
class ClassificationPredictions:
    """Classifier outputs together with their numerical interpretation.

    Parameters
    ----------
    values
        Two-dimensional array with shape
        ``(n_objects, n_classes)``.

    output_type
        Whether ``values`` contains raw logits or class probabilities.
    """

    values: np.ndarray
    output_type: ClassificationOutputType

    def __post_init__(self) -> None:
        """Validate and normalize the prediction array."""

        values = np.asarray(self.values)

        if values.ndim != 2:
            raise ValueError(
                "Classification predictions must be two-dimensional "
                "with shape (n_objects, n_classes)."
            )

        # TODO:
        # If adaptive learning is later extended to binary classifiers that expose
        # a single sigmoid output rather than two class scores, support that
        # representation through an explicit conversion step rather than
        # weakening the current classification-output contract.

        if values.shape[1] < 2:
            raise ValueError("Classification predictions must contain at least two classes.")

        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("Classification predictions must be numeric.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Classification predictions must contain only finite values.")

        object.__setattr__(self, "values", values)

    def as_probabilities(self) -> np.ndarray:
        """Return classifier outputs as validated class probabilities.

        Logits are converted with a numerically stable softmax.
        Probability inputs are validated and returned unchanged.

        Returns
        -------
        np.ndarray
            Probability array with shape ``(n_objects, n_classes)``.

        Raises
        ------
        ValueError
            If supplied probabilities fall outside the valid range or do
            not sum to one for each object.
        """

        values = self.values

        if self.output_type == "logits":
            shifted = values - np.max(
                values,
                axis=1,
                keepdims=True,
            )

            exponentials = np.exp(shifted)

            return exponentials / np.sum(
                exponentials,
                axis=1,
                keepdims=True,
            )

        if self.output_type == "probabilities":
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError("Probabilities must lie between 0 and 1.")

            row_sums = np.sum(values, axis=1)

            if not np.allclose(
                row_sums,
                1.0,
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError("Class probabilities must sum to 1 for each object.")

            return values

        raise ValueError(f"Unsupported classification output type: {self.output_type!r}")


@dataclass(slots=True)
class SelectionBatch:
    """Objects selected during one acquisition query.

    Parameters
    ----------
    ids
        Stable object identifiers for the selected objects.

    indices
        Positions of the selected objects in the candidate array that was
        scored by the acquisition strategy.

    scores
        Acquisition scores corresponding to the selected objects.
        Higher scores always indicate higher acquisition priority.
    """

    ids: list[str]
    indices: np.ndarray
    scores: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the stored selection arrays."""

        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.scores = np.asarray(self.scores, dtype=float)

        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional.")

        if self.scores.ndim != 1:
            raise ValueError("scores must be one-dimensional.")

        if not (len(self.ids) == len(self.indices) == len(self.scores)):
            raise ValueError("ids, indices, and scores must have the same length.")

    def __len__(self) -> int:
        """Return the number of selected objects."""

        return len(self.ids)


@dataclass(slots=True)
class LabeledBatch:
    """Labels obtained for a collection of queried objects.

    Parameters
    ----------
    ids
        Stable identifiers of objects for which labels were obtained.

    labels
        Labels corresponding to ``ids``. The first dimension must contain
        one entry per object. Additional dimensions are allowed so that the
        adaptive layer does not assume a particular target representation.
    """

    ids: list[str]
    labels: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the returned labels."""

        self.labels = np.asarray(self.labels)

        if len(self.ids) != len(set(self.ids)):
            raise ValueError("ids must not contain duplicates.")

        if not all(isinstance(object_id, str) for object_id in self.ids):
            raise TypeError("ids must contain only strings.")

        if self.labels.ndim == 0:
            raise ValueError("labels must have at least one dimension.")

        if self.labels.shape[0] != len(self.ids):
            raise ValueError("ids and labels must contain the same number of objects.")

    def __len__(self) -> int:
        """Return the number of labeled objects."""

        return len(self.ids)
