#!/usr/bin/env python3
"""Task 8A: saved-validation, no-GPU retrospective checkpoint audit.

This tool NEVER accesses the test split, runs ViT inference, or alters source suites.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from al_pipeline.core.analysis.checkpoint_acceptance import (
    acceptance_decision, exploratory_rules, paired_comparison,
    stratified_paired_bootstrap,
)
from al_pipeline.astronomy.analysis.checkpoint_audit import (
    CLASS_NAMES, POLICIES, SEEDS, SUITES, acquisition_diagnostics,
    ensure_alignment, read_snapshot,
)

DEFAULT_PERSISTENT = Path("/projects/bcrv/dkourtesis/applecider_adaptive")
DEFAULT_DATA = Path("/work/hdd/bcrv/dkourtesis/applecider_adaptive")
DEFAULT_OUTPUT = DEFAULT_DATA / "analysis" / "task8a_retrospective_acceptance_v1"


def dump_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty audit table {path}")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*items: object) -> int:
    digest = hashlib.sha256("|".join(map(str, items)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def pair_record(inc, cand, *, kind: str, repetitions: int) -> tuple[dict, list[dict], dict]:
    ensure_alignment(inc, cand)
    metrics = paired_comparison(inc.truth, inc.pred, cand.pred)
    uncertainty = stratified_paired_bootstrap(
        inc.truth, inc.pred, cand.pred, repetitions=repetitions,
        seed=stable_seed(inc.suite, inc.policy, inc.seed, inc.cycle, cand.cycle, kind))
    base = {"suite_key": cand.suite_key, "policy": cand.policy, "seed": cand.seed,
            "incumbent_cycle": inc.cycle, "candidate_cycle": cand.cycle,
            "comparison": kind}
    row = {**base, **{k: v for k, v in metrics.items() if k != "per_class"},
           "macro_ci025": uncertainty["delta_macro_f1"]["q025"],
           "macro_ci975": uncertainty["delta_macro_f1"]["q975"],
           "balanced_ci025": uncertainty["delta_balanced_accuracy"]["q025"],
           "balanced_ci975": uncertainty["delta_balanced_accuracy"]["q975"],
           "bootstrap_fraction_negative_macro": uncertainty["delta_macro_f1"]["fraction_negative"],
           "bootstrap_repetitions": repetitions}
    class_rows = [{**base, "class": name, **class_result,
                   "recall_ci025": uncertainty["delta_recall"][index]["q025"],
                   "recall_ci975": uncertainty["delta_recall"][index]["q975"],
                   "bootstrap_fraction_negative_recall": uncertainty["delta_recall"][index]["fraction_negative"]}
                  for index, (name, class_result) in enumerate(zip(CLASS_NAMES, metrics["per_class"], strict=True))]
    return row, class_rows, metrics


def audit(*, persistent: Path, data: Path, output: Path, suites: tuple[str, ...],
          bootstrap: int, allow_incomplete_secondary: bool, write: bool = True) -> dict:
    if not suites:
        raise ValueError("At least one suite is required")
    if any(key not in SUITES for key in suites) or len(set(suites)) != len(suites):
        raise ValueError("Unknown or duplicate suite names")
    if output.resolve().is_relative_to(persistent.resolve() / "runs") or output.resolve().is_relative_to(data.resolve() / "runs"):
        raise ValueError("Audit output must not be inside a source run directory")
    comparisons, class_rows, prediction_changes = [], [], []
    acquisition_rows, rule_rows, first_rows, missing = [], [], [], []
    completed_count = {}
    for suite_key in suites:
        for policy in POLICIES:
            for seed in SEEDS:
                cache = {}
                for cycle in range(10):
                    try:
                        cache[cycle] = read_snapshot(persistent=persistent, data=data,
                                                     suite_key=suite_key, policy=policy,
                                                     seed=seed, cycle=cycle)
                    except FileNotFoundError as exc:
                        if suite_key == "task4" or not allow_incomplete_secondary:
                            raise
                        # A missing cycle is permitted only at the end, NEVER a hole.
                        if any((data / "runs" / SUITES[suite_key] / policy /
                                f"seed_{seed}" / f"cycle_{later:03d}" /
                                "evaluation" / "validation_predictions.npz").exists()
                               for later in range(cycle + 1, 10)):
                            raise ValueError(f"Missing interior cycle {cycle}") from exc
                        missing.append({"suite_key": suite_key, "policy": policy,
                                        "seed": seed, "first_missing_cycle": cycle,
                                        "reason": str(exc)})
                        break
                if 0 not in cache:
                    raise ValueError(f"Missing L0 for {suite_key} {policy} {seed}")
                completed_count[f"{suite_key}/{policy}/{seed}"] = max(cache)
                # All policies use the same target objects; ensure exact alignment.
                for cycle, snapshot in cache.items():
                    ensure_alignment(cache[0], snapshot)
                    if cycle:
                        acquisition_rows.append(acquisition_diagnostics(
                            persistent=persistent, suite_key=suite_key,
                            policy=policy, seed=seed, cycle=cycle))
                for cycle in sorted(cache):
                    if cycle == 0:
                        continue
                    inc, cand = cache[cycle - 1], cache[cycle]
                    row, byclass, metrics = pair_record(inc, cand, kind="adjacent_saved", repetitions=bootstrap)
                    comparisons.append(row)
                    class_rows.extend(byclass)
                    changed = inc.pred != cand.pred
                    for idx in changed.nonzero()[0]:
                        prediction_changes.append({"suite_key": suite_key, "policy": policy,
                                                   "seed": seed, "incumbent_cycle": inc.cycle,
                                                   "candidate_cycle": cycle,
                                                   "object_id": str(cand.object_ids[idx]),
                                                   "true_class": CLASS_NAMES[int(cand.truth[idx])],
                                                   "old_prediction": CLASS_NAMES[int(inc.pred[idx])],
                                                   "new_prediction": CLASS_NAMES[int(cand.pred[idx])],
                                                   "lost_correct": bool(inc.pred[idx] == cand.truth[idx]),
                                                   "gained_correct": bool(cand.pred[idx] == cand.truth[idx])})
                # Sequential *hypothetical* first rejection for each rule.
                # Once rejected, never label subsequent saved candidates as valid counterfactuals.
                for rule in exploratory_rules():
                    accepted_cycle, first_rejection = 0, None
                    for cycle in range(1, max(cache) + 1):
                        inc, cand = cache[accepted_cycle], cache[cycle]
                        comparison = paired_comparison(inc.truth, inc.pred, cand.pred)
                        decision = acceptance_decision(comparison, rule)
                        rule_rows.append({"suite_key": suite_key, "policy": policy,
                                          "seed": seed, "rule": rule.name,
                                          "last_accepted_cycle": accepted_cycle,
                                          "candidate_cycle": cycle,
                                          "accept": decision["accept"],
                                          "violations": ";".join(decision["violations"]),
                                          "delta_macro_f1": comparison["delta_macro_f1"],
                                          "delta_balanced_accuracy": comparison["delta_balanced_accuracy"],
                                          "stage": "valid_observed_lineage_until_first_rejection"})
                        if not decision["accept"]:
                            first_rejection = cycle
                            break
                        accepted_cycle = cycle
                    first_rows.append({"suite_key": suite_key, "policy": policy,
                                       "seed": seed, "rule": rule.name,
                                       "first_rejected_cycle": first_rejection,
                                       "last_accepted_cycle": accepted_cycle,
                                       "evaluated_through_cycle": max(cache),
                                       "fully_accepted_saved_trajectory": first_rejection is None,
                                       "warning": "No interpretation of later saved checkpoints after rejection"})
    # Unweighted per-trajectory descriptive counts; observations from repeated
    # cycles/seeds/policies are correlated and not independent trial evidence.
    trajectory_summary = []
    for suite_key in suites:
        for policy in POLICIES:
            for seed in SEEDS:
                items = [row for row in comparisons if row["suite_key"] == suite_key
                         and row["policy"] == policy and row["seed"] == seed]
                trajectory_summary.append({
                    "suite_key": suite_key, "policy": policy, "seed": seed,
                    "n_completed_transitions": len(items),
                    "n_macro_f1_regressions": sum(row["delta_macro_f1"] < -1e-12 for row in items),
                    "n_balanced_accuracy_regressions": sum(row["delta_balanced_accuracy"] < -1e-12 for row in items),
                    "mean_delta_macro_f1": sum(row["delta_macro_f1"] for row in items) / len(items) if items else None,
                    "worst_delta_macro_f1": min((row["delta_macro_f1"] for row in items), default=None),
                    "first_macro_f1_regression": next((row["candidate_cycle"] for row in items
                                                       if row["delta_macro_f1"] < -1e-12), None),
                })
    rule_summary = []
    for suite_key in suites:
        for rule in exploratory_rules():
            cases = [row for row in first_rows if row["suite_key"] == suite_key
                     and row["rule"] == rule.name]
            rule_summary.append({"suite_key": suite_key, "rule": rule.name,
                                 "trajectories": len(cases),
                                 "trajectories_with_rejection": sum(
                                     row["first_rejected_cycle"] is not None for row in cases),
                                 "first_rejection_cycles": ";".join(
                                     f"{row['seed']}:{row['policy']}:{row['first_rejected_cycle']}"
                                     for row in cases if row["first_rejected_cycle"] is not None)})
    summary = {"schema": "task8a_retrospective_acceptance_v1", "primary": "task4" if "task4" in suites else None,
               "suites": list(suites), "validation_only": True, "test_accessed": False,
               "training_or_inference_performed": False,
               "bootstrap_method": "paired object-level class-stratified, percentile descriptive intervals",
               "bootstrap_repetitions": bootstrap,
               "metric_notes": "Candidate macro-F1 is not necessarily independent of epoch-selection validation",
               "first_rejection_caveat": "Only first rejection is meaningful; saved future training lineage diverges afterward",
               "rule_caveat": "Exploratory thresholds only; no gate approved/optimized by these data",
               "completed_cycles": completed_count,
               "n_adjacent_comparisons": len(comparisons),
               "n_changed_object_records": len(prediction_changes),
               "n_first_rejection_cases": len(first_rows), "missing_secondary": missing,
               "trajectory_summary": trajectory_summary, "rule_summary": rule_summary,
               "test_prediction_files_read": False, "test_metrics_used": False,
               "evaluation_summary_may_contain_test_metadata": True}
    if write:
        output.mkdir(parents=True, exist_ok=True)
        dump_csv(output / "adjacent_comparisons.csv", comparisons)
        dump_csv(output / "per_class_changes.csv", class_rows)
        dump_csv(output / "prediction_flips.csv", prediction_changes)
        dump_csv(output / "acquisition_batches.csv", acquisition_rows)
        dump_csv(output / "exploratory_rules.csv", rule_rows)
        dump_csv(output / "first_rejections.csv", first_rows)
        dump_csv(output / "trajectory_summary.csv", trajectory_summary)
        dump_csv(output / "rule_summary.csv", rule_summary)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (output / "README.md").write_text(
            "# Task 8A saved-validation audit\n\n"
            "All rows use *object-level validation* and existing saved checkpoints only. "
            "No test artifacts are read. `adjacent_comparisons.csv` compares consecutive "
            "saved checkpoints (descriptive even after potential gate rejection). "
            "`first_rejections.csv` identifies the first rejection under each *illustrative* "
            "rule. `exploratory_rules.csv` stops at that point: never use later saved "
            "trajectory results as an acceptance-loop counterfactual. "
            "Bootstrap intervals are descriptive and do not undo validation-based "
            "epoch selection or repeated validation reuse. "
            "The report parses evaluation_summary.json for validation provenance; that JSON may also contain test-split metadata, but no test metrics are used and no test prediction files are opened. The `prediction_flips.csv` file contains object identifiers from the "
            "historical validation set and should be shared only with collaborators.\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--persistent", type=Path, default=DEFAULT_PERSISTENT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suites", nargs="+", choices=tuple(SUITES), default=("task4", "task6b", "task6c"))
    parser.add_argument("--bootstrap", type=int, default=600)
    parser.add_argument("--require-complete-secondary", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps({"mode": "read_only", "persistent": str(args.persistent),
                          "data": str(args.data), "output": str(args.output),
                          "suites": args.suites, "seeds": SEEDS, "policies": POLICIES,
                          "cycles": list(range(10)), "bootstrap": args.bootstrap,
                          "test": "never accessed", "training": "none"}, indent=2))
    else:
        print(json.dumps(audit(persistent=args.persistent, data=args.data,
                               output=args.output, suites=tuple(args.suites),
                               bootstrap=args.bootstrap,
                               allow_incomplete_secondary=not args.require_complete_secondary), indent=2))


if __name__ == "__main__":
    main()
