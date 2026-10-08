"""Tests for shared adaptive-learning types."""

import numpy as np
import pytest

from al_pipeline.astronomy.astrominn.types import ClassificationInferenceResult
from al_pipeline.core.types import ClassificationPredictions, LabeledBatch, SelectionBatch


def test_inference_result_accepts_aligned_embeddings():
    predictions = ClassificationPredictions(
        values=np.array([[0.8, 0.2], [0.4, 0.6]]),
        output_type="probabilities",
    )
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = ClassificationInferenceResult(
        object_ids=("a", "b"),
        alerts=(1, 2),
        predictions=predictions,
        embeddings=embeddings,
    )

    assert np.array_equal(result.embeddings, embeddings)


def test_inference_result_rejects_misaligned_embeddings():
    predictions = ClassificationPredictions(
        values=np.array([[0.8, 0.2], [0.4, 0.6]]),
        output_type="probabilities",
    )

    with pytest.raises(ValueError, match="same rows"):
        ClassificationInferenceResult(
            object_ids=("a", "b"),
            alerts=(1, 2),
            predictions=predictions,
            embeddings=np.ones((1, 3)),
        )


def test_classification_predictions_accept_valid_probabilities():
    values = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.6],
        ]
    )

    predictions = ClassificationPredictions(
        values=values,
        output_type="probabilities",
    )

    assert np.array_equal(predictions.values, values)
    assert predictions.output_type == "probabilities"


def test_classification_predictions_return_valid_probabilities_unchanged():
    values = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.6],
        ]
    )

    predictions = ClassificationPredictions(
        values=values,
        output_type="probabilities",
    )

    result = predictions.as_probabilities()

    assert np.array_equal(result, values)


def test_classification_predictions_convert_logits_to_probabilities():
    logits = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )

    predictions = ClassificationPredictions(
        values=logits,
        output_type="logits",
    )

    probabilities = predictions.as_probabilities()

    assert probabilities.shape == logits.shape
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )
    assert np.allclose(
        probabilities[0],
        [0.5, 0.5],
    )


def test_classification_predictions_reject_invalid_probability_sum():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.4],
                [0.4, 0.6],
            ]
        ),
        output_type="probabilities",
    )

    with pytest.raises(
        ValueError,
        match="must sum to 1",
    ):
        predictions.as_probabilities()


def test_classification_predictions_reject_negative_probabilities():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [1.1, -0.1],
            ]
        ),
        output_type="probabilities",
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        predictions.as_probabilities()


def test_classification_predictions_handle_large_logits_stably():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [1000.0, 999.0],
                [-1000.0, -1001.0],
            ]
        ),
        output_type="logits",
    )

    probabilities = predictions.as_probabilities()

    assert np.all(np.isfinite(probabilities))
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_classification_predictions_require_two_dimensions():
    values = np.array([0.8, 0.2])

    with pytest.raises(
        ValueError,
        match="must be two-dimensional",
    ):
        ClassificationPredictions(
            values=values,
            output_type="probabilities",
        )


def test_classification_predictions_require_at_least_two_classes():
    values = np.array(
        [
            [0.8],
            [0.4],
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least two classes",
    ):
        ClassificationPredictions(
            values=values,
            output_type="probabilities",
        )


def test_classification_predictions_reject_nonfinite_values():
    values = np.array(
        [
            [0.8, 0.2],
            [np.nan, 0.6],
        ]
    )

    with pytest.raises(
        ValueError,
        match="only finite values",
    ):
        ClassificationPredictions(
            values=values,
            output_type="probabilities",
        )


def test_selection_batch_accepts_valid_selection():
    selection = SelectionBatch(
        ids=["a", "b"],
        indices=np.array([2, 5]),
        scores=np.array([0.9, 0.7]),
    )

    assert selection.ids == ["a", "b"]
    assert np.array_equal(
        selection.indices,
        np.array([2, 5]),
    )
    assert np.allclose(
        selection.scores,
        np.array([0.9, 0.7]),
    )
    assert len(selection) == 2


def test_selection_batch_requires_matching_lengths():
    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        SelectionBatch(
            ids=["a", "b"],
            indices=np.array([2, 5]),
            scores=np.array([0.9]),
        )


def test_selection_batch_requires_one_dimensional_indices():
    with pytest.raises(
        ValueError,
        match="indices must be one-dimensional",
    ):
        SelectionBatch(
            ids=["a", "b"],
            indices=np.array([[2, 5]]),
            scores=np.array([0.9, 0.7]),
        )


def test_selection_batch_requires_one_dimensional_scores():
    with pytest.raises(
        ValueError,
        match="scores must be one-dimensional",
    ):
        SelectionBatch(
            ids=["a", "b"],
            indices=np.array([2, 5]),
            scores=np.array([[0.9, 0.7]]),
        )


def test_labeled_batch_accepts_valid_labels():
    labeled = LabeledBatch(
        ids=["a", "b"],
        labels=np.array([2, 4]),
    )

    assert labeled.ids == ["a", "b"]
    assert np.array_equal(
        labeled.labels,
        np.array([2, 4]),
    )
    assert len(labeled) == 2


def test_labeled_batch_supports_multidimensional_labels():
    labeled = LabeledBatch(
        ids=["a", "b"],
        labels=np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ]
        ),
    )

    assert labeled.labels.shape == (2, 3)


def test_labeled_batch_requires_matching_lengths():
    with pytest.raises(
        ValueError,
        match="same number of objects",
    ):
        LabeledBatch(
            ids=["a", "b"],
            labels=np.array([1]),
        )


def test_labeled_batch_rejects_duplicate_ids():
    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        LabeledBatch(
            ids=["a", "a"],
            labels=np.array([1, 2]),
        )


def test_labeled_batch_rejects_non_string_ids():
    with pytest.raises(
        TypeError,
        match="must contain only strings",
    ):
        LabeledBatch(
            ids=["a", 2],
            labels=np.array([1, 2]),
        )


def test_labeled_batch_rejects_scalar_labels():
    with pytest.raises(
        ValueError,
        match="at least one dimension",
    ):
        LabeledBatch(
            ids=["a"],
            labels=np.array(1),
        )
