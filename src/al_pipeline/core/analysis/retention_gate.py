"""Model-independent, deterministic object-level checkpoint retention decisions.

Only macro-F1 controls deployment in 8C.1. Balanced accuracy and class net
true-positive changes are reported as shadow rules, never deployment gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class RetentionRule:
    macro_tolerance: float = 0.01
    balanced_accuracy_shadow_tolerance: float = 0.03
    # Class indices are supplied by the astronomy adapter; no astronomy names in core.
    shadow_net_correct_loss_limits: tuple[int, ...] = (35, 3, 100, 30, 10)

    def __post_init__(self):
        if not 0 <= self.macro_tolerance < 1:
            raise ValueError('Invalid macro-F1 tolerance')
        if not 0 <= self.balanced_accuracy_shadow_tolerance < 1:
            raise ValueError('Invalid BA tolerance')
        if not self.shadow_net_correct_loss_limits or any(x <= 0 for x in self.shadow_net_correct_loss_limits):
            raise ValueError('Class limits must be positive')


@dataclass(frozen=True)
class ObjectPredictions:
    ids: tuple[str, ...]
    targets: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        ids = tuple(str(x) for x in self.ids)
        y = np.asarray(self.targets, dtype=np.int64)
        scores = np.asarray(self.values, dtype=np.float64)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Object IDs must be unique and nonempty')
        if y.ndim != 1 or scores.ndim != 2 or scores.shape[0] != len(ids) or len(y) != len(ids):
            raise ValueError('Invalid object prediction shapes')
        if scores.shape[1] < 2 or np.any(y < 0) or np.any(y >= scores.shape[1]):
            raise ValueError('Out-of-range label')
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite predictions')
        object.__setattr__(self, 'ids', ids)
        object.__setattr__(self, 'targets', y)
        object.__setattr__(self, 'values', scores)


def metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    n_classes = scores.shape[1]
    pred = np.argmax(scores, axis=1)
    recalls, f1s, supports = [], [], []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (y == c)))
        support = int(np.sum(y == c))
        predicted = int(np.sum(pred == c))
        if support == 0:
            raise ValueError(f'Cannot audit class {c} without positives')
        recall = tp / support
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    return {'macro_f1': float(np.mean(f1s)), 'balanced_accuracy': float(np.mean(recalls)),
            'accuracy': float(np.mean(pred == y)), 'recall': recalls,
            'f1_per_class': f1s, 'support': supports,
            'correct_per_class': [int(round(a*b)) for a,b in zip(recalls,supports)]}


def paired_retention(incumbent: ObjectPredictions, candidate: ObjectPredictions,
                     rule: RetentionRule, class_names: Sequence[str]) -> dict:
    if len(class_names) != candidate.values.shape[1] or len(rule.shadow_net_correct_loss_limits) != len(class_names):
        raise ValueError('Class name/limit count mismatch')
    # Align by object ID; never rely on file ordering.
    indices = {name: i for i, name in enumerate(candidate.ids)}
    if len(incumbent.ids) != len(candidate.ids) or set(incumbent.ids) != set(indices):
        raise ValueError('Incumbent/candidate validation object set mismatch')
    order = np.asarray([indices[oid] for oid in incumbent.ids], dtype=int)
    if not np.array_equal(incumbent.targets, candidate.targets[order]):
        raise ValueError('Incumbent/candidate validation targets mismatch')
    a = metrics(incumbent.values, incumbent.targets)
    b = metrics(candidate.values[order], incumbent.targets)
    macro_delta = b['macro_f1'] - a['macro_f1']
    ba_delta = b['balanced_accuracy'] - a['balanced_accuracy']
    # Inclusive >= tolerance; equality accepted to avoid unexpected rejection.
    macro_pass = macro_delta >= -rule.macro_tolerance - 1e-12
    shadow_ba_violation = ba_delta < -rule.balanced_accuracy_shadow_tolerance - 1e-12
    cls = {}
    shadow_recall_violations = []
    incumbent_pred = incumbent.values.argmax(axis=1)
    candidate_pred = candidate.values[order].argmax(axis=1)
    for k, name in enumerate(class_names):
        y = incumbent.targets == k
        lost = int(np.sum(y & (incumbent_pred == k) & (candidate_pred != k)))
        regained = int(np.sum(y & (incumbent_pred != k) & (candidate_pred == k)))
        net = lost - regained
        limit = rule.shadow_net_correct_loss_limits[k]
        if net >= limit:
            shadow_recall_violations.append(name)
        cls[name] = {'support': a['support'][k], 'incumbent_correct': a['correct_per_class'][k],
                     'candidate_correct': b['correct_per_class'][k],
                     'incumbent_recall': a['recall'][k], 'candidate_recall': b['recall'][k],
                     'delta_recall': b['recall'][k] - a['recall'][k],
                     'lost_correct_objects': lost, 'recovered_correct_objects': regained,
                     'net_correct_loss': net, 'shadow_loss_limit': limit}
    return {'accepted': bool(macro_pass), 'decision_rule': 'macro_f1_1pp',
            'macro_delta': macro_delta, 'balanced_accuracy_delta': ba_delta,
            'incumbent_metrics': a, 'candidate_metrics': b, 'classes': cls,
            'shadow_rule': {'would_accept': bool(macro_pass and not shadow_ba_violation and not shadow_recall_violations),
                            'macro_violation': not macro_pass, 'ba_violation': bool(shadow_ba_violation),
                            'class_recall_violations': shadow_recall_violations,
                            'controls_retention': False}}
