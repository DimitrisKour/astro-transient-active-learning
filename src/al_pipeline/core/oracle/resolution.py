"""Result types for resolving real active-learning labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from al_pipeline.core.types import LabeledBatch


class LabelStatus(str, Enum):
    """Outcome of attempting to resolve one queried object's label.

    ``OUTSIDE_LABEL_SPACE`` describes the scientific resolution only. It does
    not prescribe whether the object should be discarded, retained for anomaly
    review, or promoted later; that lifecycle decision belongs to workflow
    policy.
    """

    LABELED = "labeled"
    UNRESOLVED = "unresolved"
    OUTSIDE_LABEL_SPACE = "outside_label_space"


@dataclass(frozen=True, slots=True)
class LabelResolution:
    """Label-resolution result for one queried object.

    Parameters
    ----------
    object_id
        Stable identifier of the queried object.

    status
        Outcome of the label-resolution attempt.

    label
        Label in the adaptive-learning target space. This must be present
        only when ``status`` is ``LABELED``.

    source
        External source that supplied the classification evidence, if known.

    native_label
        Classification reported by the external source before any mapping
        into the AppleCiDEr target label space.

    provenance
        Additional reproducibility metadata associated with the resolution.
        Its contents are intentionally provider-specific.
    """

    object_id: str
    status: LabelStatus
    label: object | None = None
    source: str | None = None
    native_label: object | None = None
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate one label-resolution result."""

        if not isinstance(self.object_id, str):
            raise TypeError("object_id must be a string.")

        if not self.object_id:
            raise ValueError("object_id must not be empty.")

        if not isinstance(self.status, LabelStatus):
            raise TypeError("status must be a LabelStatus.")

        if self.status is LabelStatus.LABELED:
            if self.label is None:
                raise ValueError("A labeled resolution must contain a target label.")

        elif self.label is not None:
            raise ValueError("Only labeled resolutions may contain a target label.")

        if self.source is not None:
            if not isinstance(self.source, str):
                raise TypeError("source must be a string or None.")

            if not self.source:
                raise ValueError("source must not be an empty string.")


@dataclass(frozen=True, slots=True)
class LabelResolutionBatch:
    """Resolution outcomes for a collection of queried objects."""

    resolutions: tuple[LabelResolution, ...]

    def __init__(
        self,
        resolutions: Sequence[LabelResolution],
    ) -> None:
        normalized = tuple(resolutions)

        if not all(isinstance(result, LabelResolution) for result in normalized):
            raise TypeError("resolutions must contain only LabelResolution objects.")

        object_ids = [result.object_id for result in normalized]

        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Label resolutions must not contain duplicate object IDs.")

        object.__setattr__(
            self,
            "resolutions",
            normalized,
        )

    def __len__(self) -> int:
        """Return the number of resolution outcomes."""

        return len(self.resolutions)

    def by_status(
        self,
        status: LabelStatus,
    ) -> tuple[LabelResolution, ...]:
        """Return all resolutions with one status."""

        return tuple(result for result in self.resolutions if result.status is status)

    def labeled_batch(self) -> LabeledBatch:
        """Return only successfully resolved labels for training.

        The order of successfully labeled objects is preserved from the
        resolution batch.
        """

        labeled = self.by_status(LabelStatus.LABELED)

        labels = [result.label for result in labeled]

        try:
            labels_array = np.asarray(labels)
        except ValueError as error:
            raise ValueError(
                "Resolved labels could not be represented as a consistent NumPy array."
            ) from error

        return LabeledBatch(
            ids=[result.object_id for result in labeled],
            labels=labels_array,
        )


# TODO:
# If real catalog experiments show that classification confidence or
# classification time affects whether a label is trustworthy, promote those
# fields to explicit attributes rather than hiding them in provenance.
