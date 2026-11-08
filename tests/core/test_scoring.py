"""Tests for the canonical candidate-scoring API."""

import numpy as np
import pytest

from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.scoring.fisher import FisherScorer, FisherScoringInput
from al_pipeline.core.scoring.margin import MarginScorer
from al_pipeline.core.scoring.random import RandomScorer
from al_pipeline.core.types import ClassificationPredictions


def test_random_scorer_is_reproducible():
    scorer_1 = RandomScorer(np.random.default_rng(42))
    scorer_2 = RandomScorer(np.random.default_rng(42))

    scores_1 = scorer_1.score(5)
    scores_2 = scorer_2.score(5)

    assert np.array_equal(scores_1, scores_2)
    assert scores_1.shape == (5,)


def test_random_scorer_rejects_negative_candidate_count():
    scorer = RandomScorer(np.random.default_rng(42))

    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        scorer.score(-1)


def test_entropy_scorer_prioritizes_uncertain_predictions():
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

    scores = EntropyScorer().score(predictions)

    assert scores.shape == (3,)
    assert scores[2] > scores[1] > scores[0]


def test_margin_scorer_prioritizes_smallest_margin():
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

    scores = MarginScorer().score(predictions)

    assert scores.shape == (3,)
    assert scores[2] > scores[1] > scores[0]


def test_fisher_scorer_matches_closed_form_values():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
        output_type="probabilities",
    )
    embeddings = np.array(
        [
            [3.0, 4.0],
            [0.0, 2.0],
        ]
    )

    scores = FisherScorer(include_bias=True).score(
        FisherScoringInput(
            predictions=predictions,
            embeddings=embeddings,
        )
    )

    assert np.allclose(
        scores,
        np.array(
            [
                0.0,
                (4.0 + 1.0) * 0.5,
            ]
        ),
    )


def test_fisher_closed_form_matches_explicit_autograd_trace():
    torch = pytest.importorskip('torch', reason='Optional PyTorch autograd verification')
    from torch import nn

    torch.manual_seed(7)
    classifier = nn.Linear(3, 4, bias=True).double()
    embedding = torch.tensor(
        [0.4, -1.2, 0.7],
        dtype=torch.float64,
    )
    logits = classifier(embedding)
    probabilities = torch.softmax(logits, dim=0)
    log_probabilities = torch.log_softmax(logits, dim=0)
    explicit_trace = 0.0

    for target in range(4):
        classifier.zero_grad(set_to_none=True)
        log_probabilities[target].backward(retain_graph=True)
        gradient_norm_squared = sum(
            float(parameter.grad.square().sum()) for parameter in classifier.parameters()
        )
        explicit_trace += probabilities[target].detach().item() * gradient_norm_squared

    result = FisherScorer(include_bias=True).score(
        FisherScoringInput(
            predictions=ClassificationPredictions(
                values=logits.detach().numpy()[None, :],
                output_type="logits",
            ),
            embeddings=embedding.detach().numpy()[None, :],
        )
    )

    assert result[0] == pytest.approx(
        explicit_trace,
        rel=1e-12,
        abs=1e-12,
    )


def test_fisher_input_rejects_misaligned_embeddings():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.5, 0.5],
                [0.6, 0.4],
            ]
        ),
        output_type="probabilities",
    )

    with pytest.raises(ValueError, match="same number of objects"):
        FisherScoringInput(
            predictions=predictions,
            embeddings=np.ones((1, 3)),
        )
