"""Optional, model-independent multi-metric *within-cycle* checkpoint retention.

The backend supplies a callback that writes a checkpoint to a specified path.
No neural-network, CUDA, dataset, or serialization dependency is imposed by core.
The update manifest is authoritative; superseded artifacts are reclaimed only
AFTER the manifest has been atomically replaced.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping


Direction = Literal["min", "max"]


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{field} must be numeric, not a bool or string")
    try:
        result = float(value)  # Accept numpy scalar values without depending on numpy.
    except (ValueError, TypeError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True, slots=True)
class MetricCriterion:
    """Validation metric and improvement direction; strict improvements only."""

    name: str
    direction: Direction
    evaluation_level: Literal["sample", "object", "mixed"] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.name, "metric name")
        if self.direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")
        if self.evaluation_level not in {None, "sample", "object"}:
            raise ValueError("metric evaluation_level must be sample or object")

    def is_better(self, candidate: float, incumbent: float) -> bool:
        return candidate < incumbent if self.direction == "min" else candidate > incumbent


DEFAULT_CHECKPOINT_CRITERIA = (
    MetricCriterion("validation_loss", "min", "sample"),
    MetricCriterion("validation_macro_f1", "max", "object"),
)


@dataclass(frozen=True, slots=True)
class CheckpointTrainingIdentity:
    """Provenance fixed for a single training fit and its validation population."""

    run_id: str
    cycle: int
    seed: int
    labeled_set_version: str
    validation_identity: str
    evaluation_level: Literal["sample", "object", "mixed"]
    initial_checkpoint: str | None = None

    def __post_init__(self) -> None:
        for field in ("run_id", "labeled_set_version", "validation_identity"):
            _nonempty(getattr(self, field), field)
        if type(self.cycle) is not int or self.cycle < 0:
            raise ValueError("cycle must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.evaluation_level not in {"sample", "object", "mixed"}:
            raise ValueError("evaluation_level must be sample, object or mixed")
        if self.initial_checkpoint is not None:
            _nonempty(self.initial_checkpoint, "initial_checkpoint")


@dataclass(frozen=True, slots=True)
class SelectedCheckpoint:
    """One criterion's selected epoch, score and durable artifact reference."""

    criterion: str
    epoch: int
    value: float
    path: str


@dataclass(frozen=True, slots=True)
class SelectionUpdate:
    """Result from observing one epoch (including whether weights were saved)."""

    epoch: int
    improved: tuple[str, ...]
    checkpoint_path: Path | None


class WithinCycleCheckpointKeeper:
    """Keep separate loss-best/F1-best references with bounded artifact storage.

    Every epoch with at least one improvement causes exactly one backend save.
    The two criteria can point to the same artifact if they improve together.
    The manifest is atomically written before superseded artifacts are deleted.
    On startup, ``resume=True`` verifies fit identity, criteria, and artifacts.

    The caller MUST pass validation-only metrics and a backend save callback
    that writes the model state to the supplied *temporary* file path. The
    callback must not delete or rename that file. Do not use test metrics.
    This helper is single-writer; concurrent jobs require external locking.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_dir: str | Path,
        *,
        identity: CheckpointTrainingIdentity,
        criteria: tuple[MetricCriterion, ...] = DEFAULT_CHECKPOINT_CRITERIA,
        resume: bool = False,
        suffix: str = ".pt",
    ) -> None:
        if not criteria or len({c.name for c in criteria}) != len(criteria):
            raise ValueError("criteria must contain unique metric names")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("suffix must be a simple extension such as '.pt'")
        if identity.evaluation_level != "mixed":
            if any(c.evaluation_level not in {None, identity.evaluation_level} for c in criteria):
                raise ValueError("Metric evaluation level conflicts with fit provenance")
        self.output_dir = Path(output_dir).resolve()
        self.artifact_dir = self.output_dir / "artifacts"
        self.manifest_path = self.output_dir / "checkpoint_selection.json"
        self.identity = identity
        self.criteria = tuple(criteria)
        self.suffix = suffix
        self._best: dict[str, SelectedCheckpoint] = {}
        self._last_epoch = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self._load_existing()
        elif self.manifest_path.exists() or self.artifact_dir.exists():
            raise FileExistsError("Checkpoint state exists; specify resume=True after validating fit")

    @property
    def best(self) -> dict[str, SelectedCheckpoint]:
        return dict(self._best)

    @property
    def last_epoch(self) -> int:
        return self._last_epoch

    def selected(self, metric_name: str) -> SelectedCheckpoint:
        return self._best[metric_name]

    def consider(
        self,
        *,
        epoch: int,
        validation_metrics: Mapping[str, float],
        save_checkpoint: Callable[[Path], None],
    ) -> SelectionUpdate:
        """Select independent best metrics and persist a shared epoch snapshot.

        A failed checkpoint callback does not advance the state; temporary
        files are discarded, existing selected weights remain untouched.
        """
        if type(epoch) is not int or epoch <= self._last_epoch:
            raise ValueError("epoch must be an integer strictly after the previous epoch")
        scores = {
            criterion.name: _finite_number(validation_metrics[criterion.name], criterion.name)
            for criterion in self.criteria
        }
        improving = tuple(
            criterion.name for criterion in self.criteria
            if criterion.name not in self._best
            or criterion.is_better(scores[criterion.name], self._best[criterion.name].value)
        )
        next_best = dict(self._best)
        checkpoint_path: Path | None = None
        if improving:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.artifact_dir / f"epoch_{epoch:06d}{self.suffix}"
            if checkpoint_path.exists():
                raise FileExistsError(f"Untracked checkpoint artifact exists: {checkpoint_path}")
            fd, temporary = tempfile.mkstemp(
                prefix=f".epoch_{epoch:06d}_", suffix=f"{self.suffix}.pending",
                dir=self.artifact_dir,
            )
            os.close(fd)
            try:
                save_checkpoint(Path(temporary))
                if not Path(temporary).is_file() or Path(temporary).stat().st_size <= 0:
                    raise ValueError("Checkpoint callback did not write a nonempty artifact")
                os.replace(temporary, checkpoint_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            for metric in improving:
                next_best[metric] = SelectedCheckpoint(
                    criterion=metric, epoch=epoch, value=scores[metric],
                    path=str(checkpoint_path),
                )
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "identity": asdict(self.identity),
            "criteria": [asdict(c) for c in self.criteria],
            "suffix": self.suffix,
            "last_epoch": epoch,
            "best": {name: asdict(item) for name, item in next_best.items()},
        }
        # If persisting fails, previous manifest still references old intact
        # artifacts. The new artifact is an orphan, preserved for inspection.
        _write_json_atomic(self.manifest_path, payload)
        previous_paths = {item.path for item in self._best.values()}
        current_paths = {item.path for item in next_best.values()}
        self._best = next_best
        self._last_epoch = epoch
        for obsolete in previous_paths - current_paths:
            path = Path(obsolete)
            if path.parent == self.artifact_dir:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # Garbage collection is best-effort; published decisions
                    # remain valid if a Lustre or permissions error occurs.
                    pass
        return SelectionUpdate(epoch=epoch, improved=improving, checkpoint_path=checkpoint_path)

    def _load_existing(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without manifest: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint manifest schema version")
        if payload.get("identity") != asdict(self.identity):
            raise ValueError("Checkpoint provenance differs from requested training fit")
        if payload.get("criteria") != [asdict(c) for c in self.criteria]:
            raise ValueError("Checkpoint criteria differ from stored state")
        if payload.get("suffix") != self.suffix:
            raise ValueError("Checkpoint artifact extension differs")
        last = payload.get("last_epoch")
        if type(last) is not int or last < 1:
            raise ValueError("Invalid recorded epoch")
        best = {name: SelectedCheckpoint(**item) for name, item in payload["best"].items()}
        if set(best) != {c.name for c in self.criteria}:
            raise ValueError("Missing or unknown checkpoint criterion")
        for metric, record in best.items():
            artifact = Path(record.path)
            if record.criterion != metric or not 1 <= record.epoch <= last:
                raise ValueError("Inconsistent checkpoint manifest")
            if artifact.parent != self.artifact_dir or not artifact.is_file():
                raise FileNotFoundError(f"Missing/misplaced selected checkpoint: {artifact}")
            _finite_number(record.value, f"saved {metric}")
        self._best = best
        self._last_epoch = last


__all__ = [
    "MetricCriterion", "DEFAULT_CHECKPOINT_CRITERIA", "CheckpointTrainingIdentity",
    "SelectedCheckpoint", "SelectionUpdate", "WithinCycleCheckpointKeeper",
]
