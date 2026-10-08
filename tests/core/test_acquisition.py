"""Tests for active-learning acquisition strategies."""

import numpy as np
import pytest

from al_pipeline.core.acquisition.entropy import EntropyAcquisition
from al_pipeline.core.acquisition.margin import MarginAcquisition
from al_pipeline.core.acquisition.random import RandomAcquisition
from al_pipeline.core.types import ClassificationPredictions


def test_random_acquisition_is_reproducible():
    strategy_1 = RandomAcquisition(np.random.default_rng(42))
    strategy_2 = RandomAcquisition(np.random.default_rng(42))

    scores_1 = strategy_1.score(5)
    scores_2 = strategy_2.score(5)

    assert np.array_equal(scores_1, scores_2)
    assert scores_1.shape == (5,)


def test_random_acquisition_rejects_negative_candidate_count():
    strategy = RandomAcquisition(np.random.default_rng(42))

    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        strategy.score(-1)


def test_entropy_prioritizes_uncertain_predictions():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.70, 0.30],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    scores = EntropyAcquisition().score(predictions)

    assert scores.shape == (3,)
    assert scores[2] > scores[1] > scores[0]


def test_entropy_handles_zero_probabilities():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
        output_type="probabilities",
    )

    scores = EntropyAcquisition().score(predictions)

    assert np.all(np.isfinite(scores))
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] > scores[0]


def test_entropy_logits_and_probabilities_are_equivalent():
    logits = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.0],
        ]
    )

    logit_predictions = ClassificationPredictions(
        values=logits,
        output_type="logits",
    )

    probabilities = logit_predictions.as_probabilities()

    probability_predictions = ClassificationPredictions(
        values=probabilities,
        output_type="probabilities",
    )

    strategy = EntropyAcquisition()

    assert np.allclose(
        strategy.score(logit_predictions),
        strategy.score(probability_predictions),
    )


def test_margin_prioritizes_smallest_classification_margin():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.95, 0.05],
                [0.60, 0.40],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    scores = MarginAcquisition().score(predictions)

    assert scores.shape == (3,)
    assert scores[2] > scores[1] > scores[0]


def test_margin_supports_multiclass_predictions():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.80, 0.10, 0.10],
                [0.40, 0.35, 0.25],
            ]
        ),
        output_type="probabilities",
    )

    scores = MarginAcquisition().score(predictions)

    assert scores[1] > scores[0]
