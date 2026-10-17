"""Prediction-level ensembles with explicit sample and class alignment.

This module never loads checkpoints or trains models. Domain adapters provide
member logits/probabilities in any row order; identities are mandatory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from al_pipeline.core.types import ClassificationPredictions
from al_pipeline.core.prediction.calibration import Label, _labels, _softmax_from_log_scores


Aggregation = Literal["probability_mean", "logit_mean"]


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(probabilities, dtype=float)
    valid = probabilities > 0
    terms[valid] = probabilities[valid] * np.log(probabilities[valid])
    return -np.sum(terms, axis=1)


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    """One model's predictions and identities, in arbitrary input row order."""

    member_id: str
    object_ids: tuple[str, ...]
    class_labels: tuple[Label, ...]
    predictions: ClassificationPredictions
    checkpoint_ref: str | Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.member_id, str) or not self.member_id.strip():
            raise ValueError("member_id must be a non-empty string.")
        ids = tuple(self.object_ids)
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("object_ids must be unique, non-empty strings.")
        if not isinstance(self.predictions, ClassificationPredictions):
            raise TypeError("predictions must be ClassificationPredictions.")
        shape = self.predictions.values.shape
        if len(ids) != shape[0]:
            raise ValueError("Number of object IDs differs from prediction rows.")
        object.__setattr__(self, "object_ids", ids)
        object.__setattr__(self, "class_labels", _labels(self.class_labels, shape[1]))
        # Reject invalid supplied probability vectors immediately.
        self.predictions.as_probabilities()
        if self.checkpoint_ref is not None:
            if not isinstance(self.checkpoint_ref, (str, Path)):
                raise TypeError("checkpoint_ref must be a string or Path.")
            ref = str(self.checkpoint_ref)
            if not ref.strip():
                raise ValueError("checkpoint_ref must be nonempty if provided.")
            object.__setattr__(self, "checkpoint_ref", ref)


class PredictionEnsemble:
    """Aligned weighted aggregation of independent classifier predictions.

    Members must predict the same candidate IDs and the same label set.
    Rows and class columns are reordered to match the first member, not
    silently assumed to align. Equal weights are used unless specified.
    """

    def __init__(
        self,
        members: Sequence[EnsembleMember],
        *,
        weights: Sequence[float] | None = None,
        aggregation: Aggregation = "probability_mean",
    ) -> None:
        self.members = tuple(members)
        if not self.members:
            raise ValueError("At least one ensemble member is required.")
        names = [member.member_id for member in self.members]
        if len(set(names)) != len(names):
            raise ValueError("Ensemble member IDs must be unique.")
        if aggregation not in {"probability_mean", "logit_mean"}:
            raise ValueError("Unsupported ensemble aggregation.")
        if aggregation == "logit_mean" and any(
            member.predictions.output_type != "logits" for member in self.members
        ):
            raise ValueError(
                "logit_mean requires raw (or explicitly scaled) logits from every member."
            )
        self.aggregation: Aggregation = aggregation
        self.object_ids = self.members[0].object_ids
        self.class_labels = self.members[0].class_labels

        if weights is None:
            w = np.ones(len(self.members), dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
        if (w.shape != (len(self.members),) or not np.all(np.isfinite(w))
                or np.any(w < 0) or not np.any(w > 0)):
            raise ValueError("weights must be finite, non-negative, and not all zero.")
        self.weights = tuple(float(v) for v in w / w.sum())

        self._aligned: tuple[ClassificationPredictions, ...] = tuple(
            self._align(member) for member in self.members
        )

    def _align(self, member: EnsembleMember) -> ClassificationPredictions:
        if set(member.object_ids) != set(self.object_ids):
            raise ValueError(f"Member {member.member_id!r} has different object IDs.")
        if set(member.class_labels) != set(self.class_labels):
            raise ValueError(f"Member {member.member_id!r} has different class labels.")
        rows = {name: i for i, name in enumerate(member.object_ids)}
        cols = {label: i for i, label in enumerate(member.class_labels)}
        aligned = np.asarray(member.predictions.values)[
            np.ix_([rows[name] for name in self.object_ids],
                   [cols[label] for label in self.class_labels])
        ]
        return ClassificationPredictions(aligned, member.predictions.output_type)

    def member_probabilities(self) -> tuple[np.ndarray, ...]:
        """One aligned probability array per ensemble member."""
        return tuple(model.as_probabilities() for model in self._aligned)

    def predict(self) -> ClassificationPredictions:
        """Aggregate probabilities, or softmax the mean logits if configured."""
        if self.aggregation == "probability_mean":
            probabilities = np.average(
                np.stack(self.member_probabilities()), axis=0, weights=self.weights
            )
        else:
            logits = np.average(
                np.stack([member.values for member in self._aligned]),
                axis=0, weights=self.weights,
            )
            probabilities = _softmax_from_log_scores(logits)
        return ClassificationPredictions(probabilities, "probabilities")

    def predictive_entropy(self) -> np.ndarray:
        """Entropy of configured ensemble predictions."""
        return _entropy(self.predict().values)

    def expected_member_entropy(self) -> np.ndarray:
        """Weighted expected categorical entropy across ensemble members."""
        return np.average(
            np.stack([_entropy(p) for p in self.member_probabilities()]),
            axis=0, weights=self.weights,
        )

    def disagreement(self) -> np.ndarray:
        """Jensen-Shannon / BALD-like ensemble disagreement per candidate.

        Uses entropy of the *mean probabilities*, regardless of aggregation
        mode, minus mean member entropy. It is not Bayesian epistemic
        uncertainty unless the ensemble supports that interpretation.
        """
        mean = np.average(
            np.stack(self.member_probabilities()), axis=0, weights=self.weights
        )
        values = _entropy(mean) - self.expected_member_entropy()
        return np.maximum(values, 0.0)  # suppress tiny floating-point negatives

    def evaluate(
        self,
        y_true: Sequence[Label] | np.ndarray,
        *,
        metrics: Sequence[str] = ("accuracy", "f1_macro", "log_loss", "brier", "ece"),
        calibration_bins: int = 15,
    ) -> dict[str, object]:
        """Compute ensemble and per-member metrics using existing core metrics.

        ``y_true`` must follow ``self.object_ids``. Evaluation is read-only;
        callers must enforce validation/test split isolation.
        """
        from al_pipeline.core.evaluation.metrics import compute_classification_metrics

        kwargs = {"metrics": tuple(metrics), "labels": self.class_labels,
                  "calibration_bins": calibration_bins}
        report = {
            "ensemble": compute_classification_metrics(
                np.asarray(y_true), self.predict(), **kwargs
            ),
            "members": {
                member.member_id: compute_classification_metrics(
                    np.asarray(y_true), aligned, **kwargs
                )
                for member, aligned in zip(self.members, self._aligned)
            },
        }
        return report

    def definition(self) -> dict[str, object]:
        """Serializable provenance *metadata*, not model weights/predictions."""
        return {
            "schema_version": 1,
            "aggregation": self.aggregation,
            "class_labels": list(self.class_labels),
            "object_count": len(self.object_ids),
            "ordered_object_ids_sha256": hashlib.sha256(
                json.dumps(
                    self.object_ids, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "members": [
                {"member_id": m.member_id, "checkpoint_ref": m.checkpoint_ref,
                 "weight": self.weights[i], "output_type": m.predictions.output_type}
                for i, m in enumerate(self.members)
            ],
        }

    def save_definition(self, path: str | Path) -> None:
        """Save ensemble metadata; reproducing it also requires model outputs."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(self.definition(), f, indent=2)
            f.write("\n")
        temporary.replace(destination)
