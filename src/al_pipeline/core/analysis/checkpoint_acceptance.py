"""Pure object-level paired checkpoint comparisons for active-learning audits.

No model-specific imports, filesystem I/O, or test-set access.
Bootstrap resamples *pairs* of predictions, stratified on their shared true class.
The bootstrap is descriptive; it does not correct adaptive validation reuse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcceptanceRule:
    name: str
    macro_floor: float
    balanced_floor: float | None = None
    max_net_recall_losses: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_net_recall_losses is not None and any(
            threshold is not None and threshold < 0 for threshold in self.max_net_recall_losses
        ):
            raise ValueError("Recall-loss counts cannot be negative")


# Exploratory rule panel. NOT an optimized policy or a recommended deployment gate.
def exploratory_rules(n_classes: int = 5) -> tuple[AcceptanceRule, ...]:
    if n_classes != 5:
        raise ValueError("Exploratory rare-class rules currently assume five classes")
    return (
        AcceptanceRule("strict_macro", 0.0),
        AcceptanceRule("macro_tolerance_1pp", -0.01),
        AcceptanceRule("macro_balanced_guard", -0.01, -0.015),
        # Class order is AGN, TDE, SN I, SN II, CV. Counts are examples,
        # not class-invariant percentage thresholds. 8A must scrutinize them.
        AcceptanceRule("macro_balanced_rare_guard", -0.01, -0.015,
                       (None, 1, None, 2, 2)),
    )


def _validate(truth: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray,
              n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = [np.asarray(x) for x in (truth, incumbent, candidate)]
    if any(x.ndim != 1 for x in arr) or not len(arr[0]) or any(len(x) != len(arr[0]) for x in arr):
        raise ValueError("truth and predictions must be nonempty aligned 1D arrays")
    if n_classes < 2:
        raise ValueError("At least two classes required")
    for x in arr:
        if x.dtype.kind not in 'iu' or np.any(x < 0) or np.any(x >= n_classes):
            raise ValueError("Invalid class-index array")
    if len(np.unique(arr[0])) != n_classes:
        raise ValueError("Every class must be represented in validation")
    return tuple(x.astype(np.int64, copy=False) for x in arr)


def classification_metrics(truth: np.ndarray, prediction: np.ndarray,
                           n_classes: int) -> dict:
    """Object-level macro F1, accuracy, balanced accuracy, per-class recall/F1."""
    y, p, _ = _validate(truth, prediction, prediction, n_classes)
    mat = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(mat)
    support = mat.sum(axis=1)
    predicted = mat.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes, dtype=float), where=denom > 0)
    return {"macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "accuracy": float(tp.sum() / support.sum()), "recall": recall,
            "class_f1": f1, "support": support}


def paired_comparison(truth: np.ndarray, incumbent: np.ndarray,
                      candidate: np.ndarray, n_classes: int = 5) -> dict:
    y, old, new = _validate(truth, incumbent, candidate, n_classes)
    a = classification_metrics(y, old, n_classes)
    b = classification_metrics(y, new, n_classes)
    lost = np.bincount(y[(old == y) & (new != y)], minlength=n_classes)
    gained = np.bincount(y[(old != y) & (new == y)], minlength=n_classes)
    return {
        "incumbent_macro_f1": a["macro_f1"], "candidate_macro_f1": b["macro_f1"],
        "delta_macro_f1": b["macro_f1"] - a["macro_f1"],
        "incumbent_balanced_accuracy": a["balanced_accuracy"],
        "candidate_balanced_accuracy": b["balanced_accuracy"],
        "delta_balanced_accuracy": b["balanced_accuracy"] - a["balanced_accuracy"],
        "delta_accuracy": b["accuracy"] - a["accuracy"],
        "per_class": [
            {"class_index": c, "support": int(a["support"][c]),
             "incumbent_recall": float(a["recall"][c]),
             "candidate_recall": float(b["recall"][c]),
             "delta_recall": float(b["recall"][c] - a["recall"][c]),
             "delta_f1": float(b["class_f1"][c] - a["class_f1"][c]),
             "lost_correct": int(lost[c]), "gained_correct": int(gained[c]),
             "net_lost_correct": int(lost[c] - gained[c])}
            for c in range(n_classes)],
        "n_changed_predictions": int(np.count_nonzero(old != new)),
        "n_lost_correct": int(lost.sum()), "n_gained_correct": int(gained.sum()),
    }


def stratified_paired_bootstrap(truth: np.ndarray, incumbent: np.ndarray,
                                candidate: np.ndarray, *, n_classes: int = 5,
                                repetitions: int = 600, seed: int = 42) -> dict:
    """Stratified paired resampling; percentile intervals are descriptive."""
    y, old, new = _validate(truth, incumbent, candidate, n_classes)
    if repetitions < 20:
        raise ValueError("repetitions must be >= 20")
    groups = [np.flatnonzero(y == c) for c in range(n_classes)]
    rng = np.random.default_rng(seed)
    macro, balanced = np.empty(repetitions), np.empty(repetitions)
    recalls = np.empty((repetitions, n_classes))
    for i in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        metrics = paired_comparison(y[indices], old[indices], new[indices], n_classes)
        macro[i] = metrics["delta_macro_f1"]
        balanced[i] = metrics["delta_balanced_accuracy"]
        recalls[i] = [row["delta_recall"] for row in metrics["per_class"]]

    def summarize(x: np.ndarray) -> dict:
        return {"q025": float(np.quantile(x, .025)),
                "median": float(np.median(x)), "q975": float(np.quantile(x, .975)),
                "fraction_negative": float(np.mean(x < 0)),
                "fraction_nonpositive": float(np.mean(x <= 0))}
    return {"n_bootstrap": repetitions, "seed": seed,
            "delta_macro_f1": summarize(macro),
            "delta_balanced_accuracy": summarize(balanced),
            "delta_recall": [summarize(recalls[:, c]) for c in range(n_classes)]}


def acceptance_decision(comparison: dict, rule: AcceptanceRule) -> dict:
    violations = []
    if comparison["delta_macro_f1"] < rule.macro_floor - 1e-12:
        violations.append("macro_f1")
    if rule.balanced_floor is not None and comparison["delta_balanced_accuracy"] < rule.balanced_floor - 1e-12:
        violations.append("balanced_accuracy")
    if rule.max_net_recall_losses is not None:
        if len(rule.max_net_recall_losses) != len(comparison["per_class"]):
            raise ValueError("Recall rule/class dimension mismatch")
        for c, (limit, value) in enumerate(zip(rule.max_net_recall_losses,
                                               comparison["per_class"], strict=True)):
            if limit is not None and value["net_lost_correct"] > limit:
                violations.append(f"net_lost_correct_class_{c}")
    return {"rule": rule.name, "accept": not violations, "violations": violations}
