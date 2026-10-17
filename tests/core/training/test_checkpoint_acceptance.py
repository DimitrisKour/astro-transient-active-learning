"""Pure cross-cycle acceptance + ledger tests; no neural network or datasets."""

import json

import pytest

from al_pipeline.core.training.checkpoint_selection import MetricCriterion
from al_pipeline.core.training.checkpoint_acceptance import (
    CheckpointAcceptanceLedger,
    CheckpointAcceptancePolicy,
    EvaluatedCheckpoint,
    decide_checkpoint_acceptance,
)


def checkpoint(cycle=0, loss=0.5, f1=0.6, **changes):
    args = dict(
        checkpoint_path=f"/runs/c{cycle}/best.pt", run_id="runA", seed=135,
        trained_cycle=cycle, labeled_set_version=f"L{cycle}",
        validation_identity="sha256:fixed_validation", evaluation_level="object",
        class_order=("AGN", "CV", "SN I", "SN II", "TDE"),
        validation_metrics={"validation_loss": loss, "validation_macro_f1": f1},
        selected_by="validation_loss", initial_checkpoint=None,
    )
    args.update(changes)
    return EvaluatedCheckpoint(**args)


def rule(mode="always_latest", metric="validation_loss", direction="min", tolerance=0.0):
    return CheckpointAcceptancePolicy(mode=mode, criterion=MetricCriterion(metric, direction),
                                      tolerance=tolerance)


def decide(old, new, policy):
    return decide_checkpoint_acceptance(candidate=new, incumbent=old, policy=policy)


def test_default_accept_latest_even_when_worse():
    old, new = checkpoint(), checkpoint(1, loss=0.9)
    outcome = decide(old, new, CheckpointAcceptancePolicy())
    assert outcome.accepted_candidate and outcome.accepted == new
    assert outcome.signed_improvement == pytest.approx(-0.4)


def test_keep_best_loss_strict_comparison_and_tie():
    old = checkpoint()
    p = rule("keep_best")
    assert decide(old, checkpoint(1, loss=0.4), p).accepted_candidate
    assert not decide(old, checkpoint(1, loss=0.7), p).accepted_candidate
    assert not decide(old, checkpoint(1, loss=0.5), p).accepted_candidate


def test_keep_best_macro_f1_is_maximized():
    old = checkpoint()
    p = rule("keep_best", "validation_macro_f1", "max")
    assert decide(old, checkpoint(1, f1=0.7), p).accepted_candidate
    assert not decide(old, checkpoint(1, f1=0.4), p).accepted_candidate


def test_tolerate_regression_including_boundary():
    old = checkpoint(loss=0.5)
    p = rule("tolerate_regression", tolerance=0.05)
    assert decide(old, checkpoint(1, loss=0.54), p).accepted_candidate
    assert not decide(old, checkpoint(1, loss=0.551), p).accepted_candidate
    assert decide(old, checkpoint(1, loss=0.5), p).accepted_candidate


def test_require_improvement_strict_threshold():
    old = checkpoint(f1=0.4)
    p = rule("require_improvement", "validation_macro_f1", "max", tolerance=0.05)
    assert not decide(old, checkpoint(1, f1=0.44), p).accepted_candidate
    assert decide(old, checkpoint(1, f1=0.46), p).accepted_candidate


def test_initial_model_registration():
    d = decide(None, checkpoint(), rule("keep_best"))
    assert d.reason == "initial_model"
    assert d.accepted_candidate
    assert d.incumbent is None


@pytest.mark.parametrize("changes,field", [
    ({"seed": 246}, "seed"),
    ({"run_id": "other"}, "run_id"),
    ({"validation_identity": "different"}, "validation_identity"),
    ({"evaluation_level": "sample"}, "evaluation_level"),
    ({"class_order": ("AGN", "CV")}, "class_order"),
])
def test_reject_incomparable_checkpoint_validation(changes, field):
    with pytest.raises(ValueError, match=field):
        decide(checkpoint(), checkpoint(1, **changes), rule("keep_best"))


def test_reject_same_or_earlier_cycle():
    with pytest.raises(ValueError, match="later"):
        decide(checkpoint(2), checkpoint(1), rule("keep_best"))
    with pytest.raises(ValueError, match="later"):
        decide(checkpoint(2), checkpoint(2), rule("keep_best"))


def test_missing_metric_rejected_even_if_always_latest():
    with pytest.raises(ValueError, match="Missing"):
        decide(checkpoint(), checkpoint(1), rule("always_latest", "not_a_metric", "min"))


def test_nonfinite_metric_and_test_split_rejected():
    with pytest.raises(ValueError, match="finite"):
        checkpoint(validation_metrics={"validation_loss": float("nan")})
    with pytest.raises(ValueError, match="test"):
        checkpoint(evaluation_split="test")


def test_policy_validation():
    with pytest.raises(ValueError, match="nonnegative"):
        rule("require_improvement", tolerance=-0.1)
    with pytest.raises(ValueError, match="does not use"):
        rule("always_latest", tolerance=0.1)


def test_ledger_retention_not_rollback_labeled_data(tmp_path):
    ledger = CheckpointAcceptanceLedger(tmp_path / "acceptance.json", policy=rule("keep_best"))
    start = checkpoint(0, loss=0.5)
    bad = checkpoint(1, loss=0.9)
    better = checkpoint(2, loss=0.4)
    ledger.record(start)
    discarded = ledger.record(bad)
    assert discarded.accepted == start
    assert discarded.candidate.labeled_set_version == "L1"
    assert discarded.accepted.labeled_set_version == "L0"
    accepted = ledger.record(better)
    assert accepted.accepted == better
    assert len(ledger.decisions) == 3


def test_ledger_restart_and_idempotent_retry(tmp_path):
    path = tmp_path / "acc.json"
    first = CheckpointAcceptanceLedger(path, policy=rule("keep_best"))
    c0 = checkpoint()
    c1 = checkpoint(1, loss=0.7)
    first.record(c0)
    first.record(c1)
    second = CheckpointAcceptanceLedger(path, policy=rule("keep_best"))
    assert len(second.decisions) == 2
    assert second.accepted == c0
    assert second.record(c1) == first.decisions[-1]
    assert len(second.decisions) == 2
    second.record(checkpoint(2, loss=0.3))
    assert second.accepted.trained_cycle == 2


def test_ledger_cannot_change_policy_or_candidate(tmp_path):
    path = tmp_path / "acc.json"
    ledger = CheckpointAcceptanceLedger(path, policy=rule("keep_best"))
    ledger.record(checkpoint())
    with pytest.raises(ValueError, match="different acceptance policy"):
        CheckpointAcceptanceLedger(path, policy=rule("always_latest"))
    ledger.record(checkpoint(1))
    with pytest.raises(ValueError, match="Different candidate"):
        ledger.record(checkpoint(1, loss=0.9))


def test_ledger_requires_consecutive_cycles(tmp_path):
    ledger = CheckpointAcceptanceLedger(tmp_path / "acc.json")
    ledger.record(checkpoint(0))
    with pytest.raises(ValueError, match="one acquisition cycle"):
        ledger.record(checkpoint(2))


def test_ledger_detects_tampered_acceptance(tmp_path):
    path = tmp_path / "acc.json"
    ledger = CheckpointAcceptanceLedger(path, policy=rule("keep_best"))
    ledger.record(checkpoint(0))
    ledger.record(checkpoint(1, loss=0.9))
    contents = json.loads(path.read_text())
    contents["decisions"][-1]["accepted_candidate"] = True
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="does not match"):
        CheckpointAcceptanceLedger(path, policy=rule("keep_best"))


def test_ledger_independent_of_run_state(tmp_path):
    ledger = CheckpointAcceptanceLedger(tmp_path / "acceptance.json", policy=rule("keep_best"))
    ledger.record(checkpoint(0, loss=0.5))
    ledger.record(checkpoint(1, loss=0.8))
    assert set(tmp_path.iterdir()) == {tmp_path / "acceptance.json"}


def test_ledger_evaluation_identity_rejects_changed_split(tmp_path):
    ledger = CheckpointAcceptanceLedger(tmp_path / "acceptance.json")
    ledger.record(checkpoint(0))
    with pytest.raises(ValueError, match="validation_identity"):
        ledger.record(checkpoint(1, validation_identity="other_validation"))


def test_reject_acceptance_metric_wrong_granularity():
    with pytest.raises(ValueError, match="evaluation level differs"):
        decide(checkpoint(), checkpoint(1), CheckpointAcceptancePolicy(
            mode="keep_best", criterion=MetricCriterion("validation_loss", "min", "sample")))


def test_checkpoint_path_must_not_be_overwritten_across_cycles():
    with pytest.raises(ValueError, match="immutable checkpoint"):
        decide(checkpoint(), checkpoint(1, checkpoint_path="/runs/c0/best.pt"), rule("keep_best"))


def test_tolerance_boundary_is_inclusive():
    assert decide(checkpoint(loss=0.5), checkpoint(1, loss=0.55),
                  rule("tolerate_regression", tolerance=0.05)).accepted_candidate


def test_named_test_metrics_rejected():
    with pytest.raises(ValueError, match="test metrics"):
        checkpoint(validation_metrics={"test_accuracy": 0.9, "validation_loss": 0.5})


def test_string_scores_rejected():
    with pytest.raises(TypeError, match="numeric"):
        checkpoint(validation_metrics={"validation_loss": "0.5"})
