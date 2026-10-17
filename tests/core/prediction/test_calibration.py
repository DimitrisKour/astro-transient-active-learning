"""Calibration tests need only NumPy and the existing prediction contract."""

import json

import numpy as np
import pytest

from al_pipeline.core.prediction import TemperatureScaler
from al_pipeline.core.types import ClassificationPredictions


LABELS = ("SN I", "SN II", "AGN")


def preds(z):
    return ClassificationPredictions(np.asarray(z, dtype=float), "logits")


def test_temperature_recovers_softening_and_improves_nll():
    # Construct exactly miscalibrated logits by multiplying true log-probabilities.
    rng = np.random.default_rng(123)
    n = 1800
    z = rng.normal(size=(n, 3))
    p = preds(z).as_probabilities()
    truth = np.array(LABELS)[[rng.choice(3, p=row) for row in p]]
    scaled = TemperatureScaler.fit(preds(z * 3.5), truth, class_labels=LABELS, split="validation")
    assert 2.5 < scaled.temperature < 4.5
    assert scaled.fitted_nll < scaled.baseline_nll
    assert scaled.transform(preds(z * 3.5), class_labels=LABELS).values.shape == (n, 3)


def test_no_improvement_retains_identity():
    data = preds(np.zeros((5, 3)))
    scaler = TemperatureScaler.fit(data, ["SN I"] * 5, class_labels=LABELS, split="calibration")
    assert scaler.temperature == 1.0
    np.testing.assert_allclose(scaler.transform(data, class_labels=LABELS).values, 1 / 3)


def test_zero_probability_preserved():
    scaler = TemperatureScaler(2.0, LABELS)
    data = ClassificationPredictions(np.array([[1.0, 0.0, 0.0], [0.0, 0.2, 0.8]]),
                                     "probabilities")
    actual = scaler.transform(data, class_labels=LABELS).values
    assert actual[0].tolist() == [1.0, 0.0, 0.0]
    assert actual[1, 0] == 0
    assert np.isclose(actual.sum(axis=1), 1).all()


def test_scaling_logits_and_probabilities_agree():
    z = preds([[2, 1, 0], [4, 0, -4]])
    scaler = TemperatureScaler(1.8, LABELS)
    from_logits = scaler.transform(z, class_labels=LABELS).values
    from_probs = scaler.transform(ClassificationPredictions(z.as_probabilities(), "probabilities"),
                                  class_labels=LABELS).values
    np.testing.assert_allclose(from_logits, from_probs)
    np.testing.assert_allclose(scaler.scale_logits(z, class_labels=LABELS).values, z.values / 1.8)


def test_serialization_roundtrip(tmp_path):
    scaler = TemperatureScaler.fit(preds([[3, 1, 0], [2, 0, -2], [0, 2, 1]]),
                                   ["SN I", "SN II", "AGN"],
                                   class_labels=LABELS, split="validation")
    path = tmp_path / "calibrator.json"
    scaler.save(path)
    reconstructed = TemperatureScaler.load(path)
    assert reconstructed == scaler
    assert json.loads(path.read_text())["fit_split"] == "validation"
    np.testing.assert_allclose(
        reconstructed.transform(preds([[1, 2, 3]]), class_labels=LABELS).values,
        scaler.transform(preds([[1, 2, 3]]), class_labels=LABELS).values,
    )


@pytest.mark.parametrize("bad_split", ["test", "train", "unlabeled", ""])
def test_fitting_rejects_non_heldout_splits(bad_split):
    with pytest.raises(ValueError, match="held-out"):
        TemperatureScaler.fit(preds([[1, 2, 3]]), ["SN I"],
                              class_labels=LABELS, split=bad_split)


def test_fitting_rejects_probabilities_and_bad_labels():
    data = ClassificationPredictions(np.array([[0.2, 0.3, 0.5]]), "probabilities")
    with pytest.raises(ValueError, match="requires raw logits"):
        TemperatureScaler.fit(data, ["AGN"], class_labels=LABELS, split="validation")
    with pytest.raises(ValueError, match="outside class_labels"):
        TemperatureScaler.fit(preds([[1, 2, 3]]), ["TDE"],
                              class_labels=LABELS, split="validation")
    with pytest.raises(ValueError, match="exactly once"):
        TemperatureScaler.fit(preds([[1, 2, 3]]), ["SN I"],
                              class_labels=("SN I", "SN I", "AGN"), split="validation")


def test_fitting_validates_shape_and_bounds():
    with pytest.raises(ValueError, match="one label per"):
        TemperatureScaler.fit(preds([[1, 2, 3]]), [], class_labels=LABELS,
                              split="validation")
    with pytest.raises(ValueError, match="temperature"):
        TemperatureScaler.fit(preds([[1, 2, 3]]), ["SN I"], class_labels=LABELS,
                              split="validation", min_temperature=-1)


def test_transform_rejects_class_order_mismatch():
    scaler = TemperatureScaler(1.5, LABELS)
    with pytest.raises(ValueError, match="class order"):
        scaler.transform(preds([[1, 2, 3]]), class_labels=LABELS[::-1])
    with pytest.raises(ValueError, match="raw logits"):
        scaler.scale_logits(ClassificationPredictions([[0.2, 0.3, 0.5]], "probabilities"),
                            class_labels=LABELS)


def test_invalid_scalers_and_schema():
    for t in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(t, LABELS)
    with pytest.raises(ValueError, match="schema"):
        TemperatureScaler.from_dict({"schema_version": 2, "method": "scalar_temperature"})
