"""Optional cross-cycle checkpoint decisions and durable decision provenance.

The policy never mutates the labeled pool, training artifacts, or model weights.
A previously accepted model may remain in use while the dataset continues to
advance. All comparison metrics MUST come from the same held-out validation
population, aggregation level, and ordered label space (never the test split).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

from al_pipeline.core.training.checkpoint_selection import (
    MetricCriterion,
    _finite_number,
    _nonempty,
    _write_json_atomic,
)


AcceptanceMode = Literal[
    "always_latest", "keep_best", "tolerate_regression", "require_improvement"
]


@dataclass(frozen=True, slots=True)
class CheckpointAcceptancePolicy:
    """Specify how a newly trained cycle's candidate replaces the incumbent.

    ``keep_best`` requires a strict improvement (ties retain incumbent).
    ``tolerate_regression`` accepts if regression <= tolerance (ties accepted).
    ``require_improvement`` requires improvement > tolerance.
    ``always_latest`` preserves current experimental behavior.
    """

    mode: AcceptanceMode = "always_latest"
    criterion: MetricCriterion = MetricCriterion("validation_loss", "min")
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "always_latest", "keep_best", "tolerate_regression", "require_improvement"
        }:
            raise ValueError("Unknown checkpoint acceptance policy")
        tolerance = _finite_number(self.tolerance, "tolerance")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        if self.mode in {"always_latest", "keep_best"} and tolerance != 0:
            raise ValueError(f"{self.mode} does not use a nonzero tolerance")
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class EvaluatedCheckpoint:
    """Checkpoint provenance and explicit fixed-validation evaluation.

    ``labeled_set_version`` is independent of ``trained_cycle``: retaining an
    older checkpoint must never undo an acquisition or change the labeled set.
    ``validation_identity`` should be a stable fingerprint including dataset
    version / label mapping and evaluation protocol, not a filename alone.
    """

    checkpoint_path: str
    run_id: str
    seed: int
    trained_cycle: int
    labeled_set_version: str
    validation_identity: str
    evaluation_level: Literal["sample", "object"]
    class_order: tuple[str, ...]
    validation_metrics: Mapping[str, float]
    selected_by: str
    initial_checkpoint: str | None = None
    evaluation_split: str = "validation"

    def __post_init__(self) -> None:
        for field in (
            "checkpoint_path", "run_id", "labeled_set_version", "validation_identity", "selected_by"
        ):
            _nonempty(getattr(self, field), field)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.trained_cycle) is not int or self.trained_cycle < 0:
            raise ValueError("trained_cycle must be a nonnegative integer")
        if self.evaluation_split != "validation":
            raise ValueError("Checkpoint acceptance must never select on test metrics")
        if self.evaluation_level not in {"sample", "object"}:
            raise ValueError("evaluation_level must be sample or object")
        labels = tuple(self.class_order)
        if len(labels) < 2 or any(not isinstance(x, str) or not x for x in labels):
            raise ValueError("class_order must provide at least two nonempty class labels")
        if len(set(labels)) != len(labels):
            raise ValueError("class_order contains duplicate labels")
        object.__setattr__(self, "class_order", labels)
        if not self.validation_metrics:
            raise ValueError("At least one validation metric is required")
        normalized = {
            str(name): _finite_number(value, f"validation metric {name}")
            for name, value in self.validation_metrics.items()
        }
        if any(not name.strip() for name in normalized):
            raise ValueError("Metric names must be nonempty")
        if any(name.lower().startswith(("test_", "test/")) for name in normalized):
            raise ValueError("Do not supply test metrics to checkpoint acceptance")
        object.__setattr__(self, "validation_metrics", normalized)
        if self.initial_checkpoint is not None:
            _nonempty(self.initial_checkpoint, "initial_checkpoint")

    def score(self, criterion: MetricCriterion) -> float:
        try:
            return self.validation_metrics[criterion.name]
        except KeyError as exc:
            raise ValueError(f"Missing selection metric {criterion.name!r}") from exc


def _check_comparable(candidate: EvaluatedCheckpoint, incumbent: EvaluatedCheckpoint) -> None:
    for field in (
        "run_id", "seed", "validation_identity", "evaluation_level", "class_order",
        "evaluation_split",
    ):
        if getattr(candidate, field) != getattr(incumbent, field):
            raise ValueError(f"Candidate and incumbent have incompatible {field}")
    if candidate.trained_cycle <= incumbent.trained_cycle:
        raise ValueError("Candidate must come from a later training cycle")
    if Path(candidate.checkpoint_path) == Path(incumbent.checkpoint_path):
        raise ValueError("Each cycle must keep immutable checkpoint artifacts at distinct paths")


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    """An explicit, inspectable choice of model for subsequent inference."""

    candidate: EvaluatedCheckpoint
    incumbent: EvaluatedCheckpoint | None
    accepted: EvaluatedCheckpoint
    accepted_candidate: bool
    policy: CheckpointAcceptancePolicy
    reason: str
    signed_improvement: float | None


def decide_checkpoint_acceptance(
    *,
    candidate: EvaluatedCheckpoint,
    incumbent: EvaluatedCheckpoint | None,
    policy: CheckpointAcceptancePolicy = CheckpointAcceptancePolicy(),
) -> AcceptanceDecision:
    """Choose an accepted checkpoint without changing any filesystem or AL state.

    Signed improvement is positive when the candidate wins under the metric's
    direction. Ties under ``keep_best`` retain the incumbent.
    """
    if policy.criterion.evaluation_level is not None and (
        candidate.evaluation_level != policy.criterion.evaluation_level
    ):
        raise ValueError("Candidate metric evaluation level differs from acceptance policy")
    candidate_score = candidate.score(policy.criterion)
    if incumbent is None:
        return AcceptanceDecision(
            candidate, None, candidate, True, policy, "initial_model", None,
        )
    _check_comparable(candidate, incumbent)
    incumbent_score = incumbent.score(policy.criterion)
    improvement = (
        incumbent_score - candidate_score
        if policy.criterion.direction == "min"
        else candidate_score - incumbent_score
    )
    if policy.mode == "always_latest":
        accepted, reason = True, "latest_model"
    elif policy.mode == "keep_best":
        accepted = improvement > 0
        reason = "strict_improvement" if accepted else "incumbent_not_outperformed"
    elif policy.mode == "require_improvement":
        accepted = improvement > policy.tolerance
        reason = "required_improvement_met" if accepted else "required_improvement_not_met"
    else:
        accepted = improvement >= -policy.tolerance or math.isclose(
            improvement, -policy.tolerance, rel_tol=1e-12, abs_tol=1e-12
        )
        reason = "within_allowed_regression" if accepted else "regression_exceeds_tolerance"
    return AcceptanceDecision(
        candidate=candidate,
        incumbent=incumbent,
        accepted=candidate if accepted else incumbent,
        accepted_candidate=accepted,
        policy=policy,
        reason=reason,
        signed_improvement=float(improvement),
    )


def _checkpoint_to_dict(item: EvaluatedCheckpoint | None) -> dict | None:
    if item is None:
        return None
    result = asdict(item)
    result["class_order"] = list(item.class_order)
    return result


def _checkpoint_from_dict(item: dict | None) -> EvaluatedCheckpoint | None:
    if item is None:
        return None
    return EvaluatedCheckpoint(**item)


def _policy_to_dict(policy: CheckpointAcceptancePolicy) -> dict:
    return asdict(policy)


def _decision_to_dict(decision: AcceptanceDecision) -> dict:
    return {
        "candidate": _checkpoint_to_dict(decision.candidate),
        "incumbent": _checkpoint_to_dict(decision.incumbent),
        "accepted": _checkpoint_to_dict(decision.accepted),
        "accepted_candidate": decision.accepted_candidate,
        "reason": decision.reason,
        "signed_improvement": decision.signed_improvement,
    }


class CheckpointAcceptanceLedger:
    """Atomic, append-only per-run decision ledger (single-writer only).

    One candidate is recorded per acquisition cycle, with strict +1 cycle
    advancement and idempotent retries for exactly identical candidates.
    A first record registers an existing L0/baseline model with no incumbent.
    This module NEVER writes a model checkpoint or changes RunState/Pool.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        policy: CheckpointAcceptancePolicy = CheckpointAcceptancePolicy(),
    ) -> None:
        self.path = Path(path)
        self.policy = policy
        self._decisions: list[AcceptanceDecision] = []
        if self.path.exists():
            self._load()

    @property
    def decisions(self) -> tuple[AcceptanceDecision, ...]:
        return tuple(self._decisions)

    @property
    def accepted(self) -> EvaluatedCheckpoint | None:
        return self._decisions[-1].accepted if self._decisions else None

    def record(self, candidate: EvaluatedCheckpoint) -> AcceptanceDecision:
        """Record one candidate and return the accepted model for next inference."""
        if self._decisions:
            prior = self._decisions[-1]
            if candidate.trained_cycle == prior.candidate.trained_cycle:
                if candidate == prior.candidate:
                    return prior  # Identical retry: never append twice.
                raise ValueError("Different candidate already recorded for this cycle")
            if candidate.trained_cycle != prior.candidate.trained_cycle + 1:
                raise ValueError("Checkpoint decisions must advance one acquisition cycle")
        incumbent = self.accepted
        decision = decide_checkpoint_acceptance(
            candidate=candidate, incumbent=incumbent, policy=self.policy,
        )
        records = self._decisions + [decision]
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "policy": _policy_to_dict(self.policy),
            "decisions": [_decision_to_dict(record) for record in records],
        }
        _write_json_atomic(self.path, payload)
        self._decisions.append(decision)
        return decision

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported acceptance ledger version")
        if data.get("policy") != _policy_to_dict(self.policy):
            raise ValueError("Cannot resume ledger using a different acceptance policy")
        decisions: list[AcceptanceDecision] = []
        for record in data["decisions"]:
            candidate = _checkpoint_from_dict(record["candidate"])
            if candidate is None:
                raise ValueError("Missing recorded candidate")
            incumbent = decisions[-1].accepted if decisions else None
            if decisions and candidate.trained_cycle != decisions[-1].candidate.trained_cycle + 1:
                raise ValueError("Nonconsecutive acceptance ledger cycles")
            expected = decide_checkpoint_acceptance(
                candidate=candidate, incumbent=incumbent, policy=self.policy,
            )
            if _decision_to_dict(expected) != record:
                raise ValueError("Recorded acceptance decision does not match policy")
            decisions.append(expected)
        self._decisions = decisions


__all__ = [
    "CheckpointAcceptancePolicy", "EvaluatedCheckpoint", "AcceptanceDecision",
    "decide_checkpoint_acceptance", "CheckpointAcceptanceLedger",
]
