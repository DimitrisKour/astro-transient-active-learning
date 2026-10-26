"""Tests for reusable, model-agnostic checkpoint acceptance statistics."""
import numpy as np
import pytest

from al_pipeline.core.analysis.checkpoint_acceptance import (
    AcceptanceRule, acceptance_decision, classification_metrics,
    exploratory_rules, paired_comparison, stratified_paired_bootstrap,
)


def sample():
    y = np.repeat(np.arange(5), [6, 4, 8, 5, 2]).astype(int)
    incumbent = y.copy()
    candidate = y.copy()
    candidate[6] = 0  # lose one TDE
    candidate[14] = 3  # lose one SN I
    return y, incumbent, candidate


def test_perfect_metrics_and_paired_counts():
    y, old, new = sample()
    m = classification_metrics(y, old, 5)
    assert m['macro_f1'] == 1
    assert m['balanced_accuracy'] == 1
    c = paired_comparison(y, old, new)
    assert c['n_changed_predictions'] == 2
    assert c['n_lost_correct'] == 2
    assert c['n_gained_correct'] == 0
    assert c['per_class'][1]['support'] == 4
    assert c['per_class'][1]['delta_recall'] == -0.25
    assert c['per_class'][1]['net_lost_correct'] == 1
    assert c['delta_macro_f1'] < 0


def test_bootstrap_paired_and_stratified_is_reproducible():
    y, old, new = sample()
    a = stratified_paired_bootstrap(y, old, new, repetitions=60, seed=135)
    b = stratified_paired_bootstrap(y, old, new, repetitions=60, seed=135)
    assert a == b
    assert a['delta_macro_f1']['q975'] <= 0
    assert len(a['delta_recall']) == 5
    assert a['delta_recall'][1]['median'] < 0
    assert 0 <= a['delta_macro_f1']['fraction_negative'] <= 1


def test_identical_predictions_have_zero_bootstrap_delta():
    y, old, _ = sample()
    b = stratified_paired_bootstrap(y, old, old, repetitions=30, seed=246)
    assert b['delta_macro_f1']['q025'] == 0
    assert b['delta_balanced_accuracy']['q975'] == 0
    assert b['delta_recall'][1]['q025'] == 0


def test_acceptance_rule_panel_and_rare_counts():
    y, old, new = sample()
    comparison = paired_comparison(y, old, new)
    assert not acceptance_decision(comparison, AcceptanceRule('strict', 0))['accept']
    assert acceptance_decision(comparison, AcceptanceRule('lenient', -1.0))['accept']
    rare = AcceptanceRule('rare', -1.0, None, (None, 0, None, None, None))
    verdict = acceptance_decision(comparison, rare)
    assert verdict['violations'] == ['net_lost_correct_class_1']
    assert len(exploratory_rules()) >= 4


def test_invalid_and_partial_class_inputs_fail_closed():
    y, old, new = sample()
    with pytest.raises(ValueError):
        paired_comparison(y[:-1], old, new)
    with pytest.raises(ValueError):
        paired_comparison(y.astype(float), old, new)
    with pytest.raises(ValueError):
        paired_comparison(y[y != 4], old[y != 4], new[y != 4])
    with pytest.raises(ValueError):
        stratified_paired_bootstrap(y, old, new, repetitions=5)
    with pytest.raises(ValueError):
        AcceptanceRule('bad', 0, max_net_recall_losses=(-1,))
