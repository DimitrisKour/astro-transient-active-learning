"""Model-independent scalar temperature scaling for multiclass classifiers.

Fitting is explicitly restricted to a held-out validation/calibration split.
Neither the model nor active-learning acquisition changes automatically.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from al_pipeline.core.types import ClassificationPredictions


Label = str | int
CalibrationSplit = Literal["validation", "calibration"]


def _labels(values: Sequence[Label], n_classes: int) -> tuple[Label, ...]:
    resolved = tuple(v.item() if isinstance(v, np.generic) else v for v in values)
    if len(resolved) != n_classes or len(set(resolved)) != n_classes:
        raise ValueError("class_labels must identify every output column exactly once.")
    if not all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in resolved):
        raise TypeError("class_labels must contain only string or integer labels.")
    return resolved


def _logits(values: ClassificationPredictions | np.ndarray) -> np.ndarray:
    if isinstance(values, ClassificationPredictions):
        if values.output_type != "logits":
            raise ValueError("Temperature fitting requires raw logits, not probabilities.")
        return np.asarray(values.values, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray(ClassificationPredictions(arr, "logits").values, dtype=np.float64)


def _nll(logits: np.ndarray, indices: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    # Subtract max before exponentiation to avoid overflows and underflows.
    maximum = np.max(scaled, axis=1)
    log_norm = maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))
    return float(np.mean(log_norm - scaled[np.arange(len(indices)), indices]))


def _softmax_from_log_scores(log_scores: np.ndarray) -> np.ndarray:
    maximum = np.max(log_scores, axis=1, keepdims=True)
    shifted = np.exp(log_scores - maximum)
    return shifted / shifted.sum(axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """One positive scalar temperature, tied to an explicit class order.

    ``transform`` always returns validated probabilities. For probability
    inputs it applies softmax(log(p) / T), which is *post-ensemble* temperature
    scaling rather than scaling each ensemble member's raw logits.

    ``fit`` accepts raw validation/calibration logits exclusively; test labels
    and training labels are never accepted as fitting data.
    """

    temperature: float
    class_labels: tuple[Label, ...]
    fit_split: str = "manual"
    calibration_samples: int = 0
    baseline_nll: float | None = None
    fitted_nll: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and strictly positive.")
        object.__setattr__(self, "class_labels", _labels(self.class_labels, len(self.class_labels)))
        if len(self.class_labels) < 2:
            raise ValueError("At least two classes are required.")
        if self.fit_split not in {"manual", "validation", "calibration"}:
            raise ValueError("fit_split must be 'manual', 'validation' or 'calibration'.")
        if not isinstance(self.calibration_samples, int) or self.calibration_samples < 0:
            raise ValueError("calibration_samples must be a non-negative integer.")
        for name in ("baseline_nll", "fitted_nll"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative if present.")

    @classmethod
    def fit(
        cls,
        logits: ClassificationPredictions | np.ndarray,
        y_true: Sequence[Label] | np.ndarray,
        *,
        class_labels: Sequence[Label],
        split: CalibrationSplit,
        min_temperature: float = 0.05,
        max_temperature: float = 20.0,
        iterations: int = 96,
    ) -> TemperatureScaler:
        """Minimize unweighted multiclass NLL on held-out logits.

        A deterministic one-dimensional bounded golden-section search runs in
        log-temperature, using NumPy only. The unscaled temperature (1.0) is
        retained if it is equally good within numerical tolerance.
        """
        if split not in ("validation", "calibration"):
            raise ValueError("Fit only on an explicitly held-out validation/calibration split.")
        z = _logits(logits)
        if z.shape[0] == 0:
            raise ValueError("Calibration requires at least one sample.")
        labels = _labels(class_labels, z.shape[1])
        truth = np.asarray(y_true)
        if truth.ndim != 1 or len(truth) != len(z):
            raise ValueError("y_true must contain one label per logit row.")
        lookup = {label: idx for idx, label in enumerate(labels)}
        try:
            encoded = np.asarray([lookup[v.item() if isinstance(v, np.generic) else v]
                                  for v in truth], dtype=np.int64)
        except (KeyError, TypeError) as exc:
            raise ValueError("y_true contains labels outside class_labels.") from exc

        if (not math.isfinite(min_temperature) or not math.isfinite(max_temperature)
                or min_temperature <= 0 or min_temperature > 1.0
                or max_temperature < 1.0 or max_temperature <= min_temperature):
            raise ValueError(
                "Require 0 < min_temperature <= 1 <= max_temperature "
                "with min_temperature < max_temperature, both finite."
            )
        if not isinstance(iterations, int) or iterations < 8:
            raise ValueError("iterations must be an integer >= 8.")

        low, high = math.log(min_temperature), math.log(max_temperature)
        golden = (math.sqrt(5) - 1) / 2
        c = high - golden * (high - low)
        d = low + golden * (high - low)
        fc, fd = _nll(z, encoded, math.exp(c)), _nll(z, encoded, math.exp(d))
        for _ in range(iterations):
            if fc < fd:
                high, d, fd = d, c, fc
                c = high - golden * (high - low)
                fc = _nll(z, encoded, math.exp(c))
            else:
                low, c, fc = c, d, fd
                d = low + golden * (high - low)
                fd = _nll(z, encoded, math.exp(d))

        trial = math.exp((low + high) / 2)
        baseline = _nll(z, encoded, 1.0)
        alternatives = [(baseline, 1.0), (_nll(z, encoded, trial), trial),
                        (_nll(z, encoded, min_temperature), min_temperature),
                        (_nll(z, encoded, max_temperature), max_temperature)]
        fitted_nll, chosen = min(alternatives, key=lambda item: item[0])
        if baseline - fitted_nll <= 1e-10:
            chosen, fitted_nll = 1.0, baseline
        return cls(chosen, labels, split, len(z), baseline, fitted_nll)

    def transform(
        self,
        predictions: ClassificationPredictions,
        *,
        class_labels: Sequence[Label],
    ) -> ClassificationPredictions:
        """Apply the fitted temperature and return class probabilities.

        Class labels must match the fitted *order*, not just the same set.
        Input predictions may be logits or probabilities. Exact zeros in
        probability inputs remain zero after temperature transformation.
        """
        order = _labels(class_labels, predictions.values.shape[1])
        if order != self.class_labels:
            raise ValueError("Prediction class order differs from calibration class order.")
        if predictions.output_type == "logits":
            scaled = np.asarray(predictions.values, dtype=float) / self.temperature
        elif predictions.output_type == "probabilities":
            probabilities = predictions.as_probabilities()
            with np.errstate(divide="ignore"):
                scaled = np.log(probabilities) / self.temperature
        else:
            raise ValueError("Unsupported prediction type.")
        return ClassificationPredictions(_softmax_from_log_scores(scaled), "probabilities")

    def scale_logits(
        self,
        predictions: ClassificationPredictions,
        *,
        class_labels: Sequence[Label],
    ) -> ClassificationPredictions:
        """Return scaled logits for pipelines that combine model logits."""
        if predictions.output_type != "logits":
            raise ValueError("scale_logits requires raw logits.")
        order = _labels(class_labels, predictions.values.shape[1])
        if order != self.class_labels:
            raise ValueError("Prediction class order differs from calibration class order.")
        return ClassificationPredictions(
            np.asarray(predictions.values, dtype=float) / self.temperature, "logits"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a versioned, JSON-safe description of this calibrator."""
        return {
            "schema_version": 1,
            "method": "scalar_temperature",
            "temperature": float(self.temperature),
            "class_labels": list(self.class_labels),
            "fit_split": self.fit_split,
            "calibration_samples": self.calibration_samples,
            "baseline_nll": self.baseline_nll,
            "fitted_nll": self.fitted_nll,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TemperatureScaler:
        """Reconstruct a calibrator, rejecting incompatible versions."""
        if payload.get("schema_version") != 1 or payload.get("method") != "scalar_temperature":
            raise ValueError("Unsupported temperature-calibration schema.")
        return cls(
            temperature=float(payload["temperature"]),
            class_labels=tuple(payload["class_labels"]),
            fit_split=str(payload["fit_split"]),
            calibration_samples=int(payload["calibration_samples"]),
            baseline_nll=(
                None if payload["baseline_nll"] is None else float(payload["baseline_nll"])
            ),
            fitted_nll=(None if payload["fitted_nll"] is None else float(payload["fitted_nll"])),
        )

    def save(self, path: str | Path) -> None:
        """Atomically write this calibrator's small JSON artifact."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
