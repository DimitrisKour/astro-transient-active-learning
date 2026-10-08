"""Oracle backed by labels already available in a dataset."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from al_pipeline.core.oracle.base import LabelProvider
from al_pipeline.core.types import LabeledBatch, SelectionBatch


class ExistingLabelsOracle(LabelProvider):
    """Provide labels from a pre-existing hidden label mapping.

    This oracle is intended for controlled active-learning experiments in
    which ground-truth labels already exist but are hidden from the
    acquisition process.

    Parameters
    ----------
    labels
        Mapping from stable object IDs to their known labels.
    """

    def __init__(
        self,
        labels: Mapping[str, object],
    ) -> None:
        if not all(isinstance(object_id, str) for object_id in labels):
            raise TypeError("Label mapping keys must contain only strings.")

        self._labels = dict(labels)

    def query(
        self,
        selection: SelectionBatch,
    ) -> LabeledBatch:
        """Return known labels for the selected objects.

        Labels are returned in exactly the same object order as the input
        selection.

        Raises
        ------
        KeyError
            If any selected object does not have a known label.
        ValueError
            If the selected labels cannot be represented as a consistent
            NumPy array.
        """

        missing_ids = [object_id for object_id in selection.ids if object_id not in self._labels]

        if missing_ids:
            raise KeyError(f"No existing label is available for selected IDs: {missing_ids}")

        selected_labels = [self._labels[object_id] for object_id in selection.ids]

        try:
            labels_array = np.asarray(selected_labels)
        except ValueError as error:
            raise ValueError(
                "Selected labels could not be represented as a consistent NumPy array."
            ) from error

        return LabeledBatch(
            ids=list(selection.ids),
            labels=labels_array,
        )


# TODO:
# ExistingLabelsOracle currently assumes that every queried object should
# have a ground-truth label. This is appropriate for controlled benchmarking.
# If retrospective experiments later use only partially labeled reference
# datasets, decide whether to support partial returns here or introduce a
# separate label-provider implementation.
