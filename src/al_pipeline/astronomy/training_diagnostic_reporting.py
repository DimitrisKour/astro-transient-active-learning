"""Paired reports for training diagnostics, including incomplete-task accounting."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from al_pipeline.astronomy.training_diagnostics import load_plan, sha256, write_json

METRICS = ("f1_macro", "balanced_accuracy", "accuracy")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _bootstrap_mean_interval(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = random.Random(173)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return means[124], means[4874]


def report_diagnostics(plan_path: Path, output_dir: Path, *, plots: bool = True) -> dict:
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_by_task = {}
    incomplete = []
    records = []
    per_class = []
    plan_digest = sha256(plan_path)
    # Resolve small task results relative to the downloaded plan, making the
    # report portable without downloading model checkpoints or historical arrays.
    for task in plan["tasks"]:
        directory = plan_path.parent / "tasks" / task["task_id"]
        path = directory / "result.json"
        if not path.exists():
            status_path = directory / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            incomplete.append({"task": task, "status": status.get("status", "not_started")})
            continue
        result = json.loads(path.read_text())
        if result.get("task") != task or result.get("plan_sha256") != plan_digest:
            raise ValueError(f"Result belongs to another task/plan: {path}")
        splits = {split["split"]: split for split in result["evaluation"]}
        if set(splits) != {"validation", "test"}:
            raise ValueError(f"Missing/duplicate validation or test evaluation: {path}")
        result_by_task[task["index"]] = result
        for split, evaluation in splits.items():
            for metric in METRICS:
                value = float(evaluation["object_metrics"][metric]) * 100
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite object metric: {path}, {split}, {metric}")
                records.append({
                    **task, "split": split, "metric": metric, "value_percent": value,
                    "best_epoch": result["training"]["best_epoch"],
                    "epochs_completed": result["training"]["epochs_completed"],
                    "training_seconds": result["training"]["elapsed_seconds"],
                    "best_validation_loss": result["training"]["best_validation_loss"],
                })
            for name, metrics in result.get("object_per_class", {}).get(split, {}).items():
                per_class.append({**task, "split": split, "class_name": name, **metrics})
    paired = []
    transition = defaultdict(dict)
    order = defaultdict(dict)
    initial_hashes_by_seed = defaultdict(set)
    for task in plan["tasks"]:
        if task["kind"] != "order" or task["index"] not in result_by_task:
            continue
        initialization = result_by_task[task["index"]].get("initialization")
        if initialization:
            initial_hashes_by_seed[task["training_seed"]].add(initialization["vit_state_sha256"])
    if any(len(hashes) > 1 for hashes in initial_hashes_by_seed.values()):
        raise ValueError("Full-data order variants used different initial ViT weights for the same seed.")
    for record in records:
        if record["kind"] == "transition":
            key = (record["training_seed"], record["anchor"], record["split"], record["metric"])
            transition[key][record["cycle"]] = record
        else:
            key = (record["training_seed"], record["split"], record["metric"])
            order[key][record["variant"]] = record
    for (seed, anchor, split, metric), cycles in sorted(transition.items()):
        if set(cycles) != {6, 7}:
            continue
        a, b = cycles[6]["value_percent"], cycles[7]["value_percent"]
        paired.append({
            "training_seed": seed, "anchor": anchor, "split": split, "metric": metric,
            "cycle_6_percent": a, "cycle_7_percent": b, "delta_pp": b - a,
        })
    transition_summaries = []
    for split in ("validation", "test"):
        for metric in METRICS:
            group = [r for r in paired if not r["anchor"] and r["split"] == split
                     and r["metric"] == metric]
            deltas = [r["delta_pp"] for r in group]
            low, high = _bootstrap_mean_interval(deltas)
            transition_summaries.append({
                "split": split, "metric": metric, "planned_pairs": len(plan["repeat_seeds"]),
                "completed_pairs": len(group),
                "cycle_6_mean_percent": statistics.mean(r["cycle_6_percent"] for r in group)
                                        if group else None,
                "cycle_7_mean_percent": statistics.mean(r["cycle_7_percent"] for r in group)
                                        if group else None,
                "delta_mean_pp": statistics.mean(deltas) if deltas else None,
                "delta_sd_pp": _sd(deltas), "delta_ci95_low_pp": low, "delta_ci95_high_pp": high,
                "negative_pairs": sum(d < 0 for d in deltas),
                "fraction_negative": sum(d < 0 for d in deltas) / len(deltas) if deltas else None,
                "all_pairs_complete": len(group) == len(plan["repeat_seeds"]),
            })
    differences = []
    for (seed, split, metric), variants in sorted(order.items()):
        if "canonical" not in variants:
            continue
        reference = variants["canonical"]["value_percent"]
        for variant, record in sorted(variants.items()):
            if variant == "canonical":
                continue
            differences.append({
                "training_seed": seed, "split": split, "metric": metric, "variant": variant,
                "original_order_percent": record["value_percent"],
                "canonical_percent": reference, "original_minus_canonical_pp":
                record["value_percent"] - reference,
            })
    order_summaries = []
    for split in ("validation", "test"):
        for metric in METRICS:
            for variant in ("canonical", *plan.get("order_strategies", (
                "random", "entropy", "entropy_diversity", "entropy_balanced_explore",
            ))):
                values = [r["value_percent"] for r in records if r["kind"] == "order"
                          and r["split"] == split and r["metric"] == metric
                          and r["variant"] == variant]
                order_summaries.append({
                    "split": split, "metric": metric, "variant": variant,
                    "completed_seeds": len(values), "planned_seeds": 3,
                    "mean_percent": statistics.mean(values) if values else None,
                    "sd_percent": _sd(values),
                })
    metric_fields = [
        "index", "task_id", "kind", "cycle", "training_seed", "loader_seed", "variant",
        "anchor", "objects", "samples", "split", "metric", "value_percent", "best_epoch",
        "epochs_completed", "training_seconds", "best_validation_loss",
    ]
    _write_csv(output_dir / "metrics.csv", records, metric_fields)
    _write_csv(output_dir / "paired_deltas.csv", paired, [
        "training_seed", "anchor", "split", "metric", "cycle_6_percent", "cycle_7_percent", "delta_pp",
    ])
    _write_csv(output_dir / "transition_summary.csv", transition_summaries,
               list(transition_summaries[0]))
    _write_csv(output_dir / "order_differences.csv", differences, [
        "training_seed", "split", "metric", "variant", "original_order_percent",
        "canonical_percent", "original_minus_canonical_pp",
    ])
    _write_csv(output_dir / "order_summary.csv", order_summaries, list(order_summaries[0]))
    anchors = []
    for record in records:
        if not record["anchor"]:
            continue
        original = next(r for r in plan["anchor_source_evaluations"]
                        if r["cycle"] == record["cycle"] and r["split"] == record["split"])
        source_value = float(original["metrics"][record["metric"]]) * 100
        anchors.append({
            "cycle": record["cycle"], "split": record["split"], "metric": record["metric"],
            "training_seed": record["training_seed"], "original_percent": source_value,
            "retrained_percent": record["value_percent"],
            "retrained_minus_original_pp": record["value_percent"] - source_value,
        })
    _write_csv(output_dir / "anchor_reproduction.csv", anchors, [
        "cycle", "split", "metric", "training_seed", "original_percent", "retrained_percent",
        "retrained_minus_original_pp",
    ])
    _write_csv(output_dir / "object_per_class.csv", per_class, [
        "index", "task_id", "kind", "cycle", "training_seed", "variant", "anchor", "split",
        "class_name", "precision", "recall", "f1", "support",
    ])
    _write_csv(output_dir / "incomplete_tasks.csv", [
        {**r["task"], "status": r["status"]} for r in incomplete
    ], ["index", "task_id", "kind", "cycle", "training_seed", "variant", "status"])
    summary = {
        "planned_tasks": len(plan["tasks"]), "completed_tasks": len(result_by_task),
        "all_tasks_complete": not incomplete, "incomplete_task_indices":
        [r["task"]["index"] for r in incomplete],
        "transition": transition_summaries, "order": order_summaries,
        "interval_scope": "Paired-seed percentile bootstrap (5000 resamples). Training-randomness "
                          "uncertainty on one fixed acquisition trajectory; not dataset uncertainty.",
        "anchor_excluded_from_repeat_summary": True,
        "initialization_fingerprints_available": sum(
            bool(r.get("initialization")) for r in result_by_task.values()
        ),
        "warning": "Incomplete jobs/pairs are explicitly listed. Do not interpret a partial report "
                   "as a completed experiment; timeouts can selectively censor slower fits.",
    }
    write_json(output_dir / "summary.json", summary)
    if plots:
        _plot(output_dir, paired, records)
    return summary


def _plot(output: Path, paired: list[dict], records: list[dict]) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = {"canonical": "#222222", "random": "#777777", "entropy": "#D89000",
              "entropy_diversity": "#78459A", "entropy_balanced_explore": "#456EA7"}
    for split in ("validation", "test"):
        group = [r for r in paired if not r["anchor"] and r["split"] == split
                 and r["metric"] == "f1_macro"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for i, r in enumerate(group):
            axes[0].plot([6, 7], [r["cycle_6_percent"], r["cycle_7_percent"]],
                         color="#456EA7", alpha=0.45, marker="o", linewidth=1)
            axes[1].scatter(i + 1, r["delta_pp"], color="#456EA7", s=24)
        axes[0].set(xticks=[6, 7], xlabel="Saved dataset cycle", ylabel="Macro-F1 (%)")
        axes[1].axhline(0, color="#222222", linestyle="--", linewidth=0.8)
        axes[1].set(xlabel="Completed paired training replicate", ylabel="Cycle 7 − cycle 6 (pp)")
        fig.suptitle(f"Fixed acquisition trajectory: {split} ({len(group)} complete fresh pairs)")
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"transition_{split}.{suffix}", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        seeds = (135, 246, 357)
        for offset, (variant, color) in enumerate(colors.items()):
            x, y = [], []
            for i, seed in enumerate(seeds):
                group = [r for r in records if r["kind"] == "order" and r["split"] == split
                         and r["metric"] == "f1_macro" and r["variant"] == variant
                         and r["training_seed"] == seed]
                if group:
                    x.append(i + (offset - 2) * 0.09)
                    y.append(group[0]["value_percent"])
            ax.scatter(x, y, color=color, label=variant.replace("_", " "), s=35)
        ax.set(xticks=range(3), xticklabels=seeds, xlabel="Training seed", ylabel="Macro-F1 (%)")
        ax.set_title(f"Full-data order comparison: {split}")
        ax.legend(frameon=False, fontsize=8)
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"full_order_{split}.{suffix}", dpi=180)
        plt.close(fig)
