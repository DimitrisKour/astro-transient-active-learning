"""Model-agnostic ensemble regression tests with intentionally shuffled IDs."""

import json

import numpy as np
import pytest

from al_pipeline.core.prediction import EnsembleMember, PredictionEnsemble, TemperatureScaler
from al_pipeline.core.types import ClassificationPredictions


LABELS = ("SN I", "SN II", "AGN")
IDS = ("ztf-a", "ztf-b", "ztf-c")


def member(name, p, ids=IDS, labels=LABELS, output="probabilities"):
    return EnsembleMember(name, ids, labels, ClassificationPredictions(np.asarray(p), output),
                          checkpoint_ref=f"cycle0/{name}.pt")


def test_aligns_both_rows_and_columns():
    one = member("one", [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    # Second member uses order c,a,b and labels AGN,SN I,SN II.
    two = member("two", [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                 ids=("ztf-c", "ztf-a", "ztf-b"), labels=("AGN", "SN I", "SN II"))
    ensemble = PredictionEnsemble([one, two])
    np.testing.assert_allclose(ensemble.predict().values, one.predictions.values)
    assert ensemble.object_ids == IDS
    assert ensemble.class_labels == LABELS
    np.testing.assert_allclose(ensemble.disagreement(), 0, atol=1e-12)


def test_weighted_probability_mean():
    a = member("a", [[0.8, 0.1, 0.1]] * 3)
    b = member("b", [[0.2, 0.7, 0.1]] * 3)
    e = PredictionEnsemble([a, b], weights=(3, 1))
    np.testing.assert_allclose(e.predict().values, [[0.65, 0.25, 0.1]] * 3)
    assert np.isclose(e.weights[0], 0.75)


def test_entropy_and_disagreement_cases():
    high_entropy_equal = member("a", [[0.5, 0.5, 0.0]] * 3)
    high_entropy_equal2 = member("b", [[0.5, 0.5, 0.0]] * 3)
    agreeing = PredictionEnsemble([high_entropy_equal, high_entropy_equal2])
    assert np.all(agreeing.predictive_entropy() > 0.68)
    np.testing.assert_allclose(agreeing.disagreement(), 0, atol=1e-12)

    disagree = PredictionEnsemble([
        member("left", [[0.99, 0.01, 0.0]] * 3),
        member("right", [[0.01, 0.99, 0.0]] * 3),
    ])
    assert np.all(disagree.disagreement() > 0.60)
    assert np.all(disagree.disagreement() <= np.log(3))


def test_logit_mean_not_probability_mean():
    a = member("a", [[9, 0, 0]] * 3, output="logits")
    b = member("b", [[0, 0, 0]] * 3, output="logits")
    logits = PredictionEnsemble([a, b], aggregation="logit_mean")
    probs = PredictionEnsemble([a, b], aggregation="probability_mean")
    assert logits.predict().values[0, 0] > probs.predict().values[0, 0]
    # Disagreement definition is the same regardless of output aggregation.
    np.testing.assert_allclose(logits.disagreement(), probs.disagreement())


def test_calibration_of_members_and_post_ensemble():
    x = member("x", [[3, 1, 0]] * 3, output="logits")
    y = member("y", [[1, 2, 0]] * 3, output="logits")
    t = TemperatureScaler(2.0, LABELS)
    calibrated_members = PredictionEnsemble([
        member("x", t.scale_logits(x.predictions, class_labels=LABELS).values, output="logits"),
        member("y", t.scale_logits(y.predictions, class_labels=LABELS).values, output="logits"),
    ])
    ensemble = PredictionEnsemble([x, y])
    post_calibrated = t.transform(ensemble.predict(), class_labels=LABELS)
    assert not np.allclose(calibrated_members.predict().values, post_calibrated.values)


def test_per_member_and_ensemble_reports(tmp_path):
    one = member("a", [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    two = member("b", [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7]])
    ensemble = PredictionEnsemble([one, two])
    report = ensemble.evaluate(["SN I", "SN II", "AGN"])
    assert report["ensemble"]["accuracy"] == 1
    assert set(report["members"]) == {"a", "b"}
    assert "ece" in report["ensemble"]
    path = tmp_path / "definition.json"
    ensemble.save_definition(path)
    definition = json.loads(path.read_text())
    assert definition["members"][0]["checkpoint_ref"] == "cycle0/a.pt"
    assert definition["object_count"] == 3
    assert len(definition["ordered_object_ids_sha256"]) == 64
    assert "predictions" not in path.read_text()


def test_rejects_unaligned_objects_classes_and_duplicates():
    a = member("a", [[1, 0, 0]] * 3)
    wrong_ids = member("b", [[1, 0, 0]] * 3, ids=("x", "ztf-b", "ztf-c"))
    with pytest.raises(ValueError, match="object IDs"):
        PredictionEnsemble([a, wrong_ids])
    wrong_labels = member("b", [[1, 0, 0]] * 3, labels=("SN I", "SN II", "TDE"))
    with pytest.raises(ValueError, match="class labels"):
        PredictionEnsemble([a, wrong_labels])
    with pytest.raises(ValueError, match="unique"):
        PredictionEnsemble([a, a])
    with pytest.raises(ValueError, match="unique"):
        member("bad", [[1, 0, 0]] * 3, ids=("ztf-a", "ztf-a", "ztf-c"))


def test_rejects_bad_weights_and_logits_policy():
    a = member("a", [[1, 0, 0]] * 3)
    b = member("b", [[0, 1, 0]] * 3)
    for weights in ((0, 0), (1, -1), (float("nan"), 1), (1,), (float("inf"), 1)):
        with pytest.raises(ValueError, match="weights"):
            PredictionEnsemble([a, b], weights=weights)
    with pytest.raises(ValueError, match="logit_mean requires"):
        PredictionEnsemble([a, b], aggregation="logit_mean")
    with pytest.raises(ValueError, match="aggregation"):
        PredictionEnsemble([a, b], aggregation="some_other_rule")


def test_empty_and_bad_probabilities():
    with pytest.raises(ValueError, match="At least one"):
        PredictionEnsemble([])
    with pytest.raises(ValueError, match="sum to 1"):
        member("invalid", [[0.2, 0.3, 0.4]] * 3)
