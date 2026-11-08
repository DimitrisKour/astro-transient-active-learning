#!/usr/bin/env python3
"""Read-only summary of saved AL selections for an illustrative cycle-2 overlap.

Requires NO model, image data, or oracle labels. Reads historical selection.json
files and publishes only aggregate intersections/Jaccards, never object IDs.

Comparisons within a suite can be more controlled than those across suites.
The exported pairs retain suite identity; matching seeds and budgets alone do
not verify a shared candidate universe or training protocol. Different-L0 runs
are not included here.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

# Chosen for a small, legible 5x5 matrix; can be changed in code after review.
POLICIES = (
    ("entropy", "astrominn_closed_loop_warm_v1", "entropy"),
    ("entropy_diversity", "astrominn_closed_loop_warm_v1", "entropy_diversity"),
    ("badge", "astrominn_task11_advanced_acquisition_v1", "badge"),
    ("bald", "astrominn_task12_bald_mc20_v1", "bald"),
    ("tailor_diversity", "astrominn_task13_tailor_pilot_v1", "tailor_diversity"),
)
SEEDS = (135, 246, 357)
FIELDS = (
    "seed", "policy_a", "suite_a", "policy_b", "suite_b",
    "cycle2_intersection", "cycle2_union", "cycle2_jaccard",
    "cumulative_intersection", "cumulative_union", "cumulative_jaccard",
)
MEAN_FIELDS = ("policy_a", "policy_b", "n_seeds", "cycle2_mean", "cumulative_mean", "cycle2_sd", "cumulative_sd")


def read_selected(root: Path, suite: str, policy: str, seed: int, cycle: int,
                  *, batch_size: int = 473) -> set[str]:
    path = root / "runs" / suite / policy / f"seed_{seed}" / f"cycle_{cycle:03d}" / "selection.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    selected = data.get("ids") if isinstance(data, dict) else None
    if not isinstance(selected, list) or len(selected) != batch_size:
        raise ValueError(f"Unexpected cycle selection size/format in {path}: expected {batch_size}")
    if any(not isinstance(value, str) or not value for value in selected):
        raise ValueError(f"Invalid object identifier in {path}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate acquisition IDs in {path}")
    return set(selected)


def jaccard(a: set[str], b: set[str]) -> tuple[int, int, float]:
    union = len(a | b)
    if not union:
        raise ValueError("Cannot calculate overlap for empty selections")
    common = len(a & b)
    return common, union, common / union


def csv_text(rows: list[dict], fields: tuple[str, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def collect(root: Path, policies=POLICIES, seeds=SEEDS, batch_size=473):
    if len({name for name, _, _ in policies}) != len(policies):
        raise ValueError("Policy display keys must be unique")
    if not policies or len(policies) > 7:
        raise ValueError("Select one to seven policies for an uncluttered comparison")
    rows = []
    for seed in seeds:
        groups = []
        for name, suite, policy in policies:
            c1 = read_selected(root, suite, policy, seed, 1, batch_size=batch_size)
            c2 = read_selected(root, suite, policy, seed, 2, batch_size=batch_size)
            if c1 & c2:
                raise ValueError(f"{suite}/{policy}/seed_{seed}: repeated selection across cycles")
            groups.append((name, suite, c2, c1 | c2))
        for i, (name_a, suite_a, round_a, cum_a) in enumerate(groups):
            for name_b, suite_b, round_b, cum_b in groups[i:]:
                a_i, a_u, a_j = jaccard(round_a, round_b)
                c_i, c_u, c_j = jaccard(cum_a, cum_b)
                rows.append(dict(
                    seed=seed, policy_a=name_a, suite_a=suite_a, policy_b=name_b, suite_b=suite_b,
                    cycle2_intersection=a_i, cycle2_union=a_u, cycle2_jaccard=a_j,
                    cumulative_intersection=c_i, cumulative_union=c_u, cumulative_jaccard=c_j,
                ))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["policy_a"], r["policy_b"])].append(r)
    averaged = []
    for (a, b), matches in agg.items():
        vals2 = [x["cycle2_jaccard"] for x in matches]
        valsc = [x["cumulative_jaccard"] for x in matches]
        averaged.append(dict(policy_a=a, policy_b=b, n_seeds=len(matches),
                             cycle2_mean=statistics.mean(vals2), cumulative_mean=statistics.mean(valsc),
                             cycle2_sd=statistics.stdev(vals2) if len(vals2)>1 else 0,
                             cumulative_sd=statistics.stdev(valsc) if len(valsc)>1 else 0))
    meta = {
        "schema": "astro_cycle2_selected_overlap_v1",
        "comparison": "descriptive, same-seed selected-object overlap; cross-suite matching not fully audited",
        "seeds": list(seeds),
        "policies": [{"key": a, "suite": b, "strategy": c} for a,b,c in policies],
        "cycle": 2, "batch_size": batch_size,
        "objects_per_cumulative_selection": 2 * batch_size,
        "round2": "objects selected specifically in acquisition cycle 2",
        "cumulative": "union of selections in acquisition cycles 1 and 2; excludes initial L0",
        "excluded": ["selected object IDs", "candidate IDs", "candidate labels", "test metrics", "model weights", "raw images"],
    }
    return rows, averaged, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists():
        parser.error(f"Refusing to overwrite existing file: {destination}")
    rows, mean_rows, meta = collect(args.persistent_root.expanduser())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", ZIP_DEFLATED) as z:
        z.writestr("astro_cycle2_overlap_by_seed.csv", csv_text(rows,FIELDS))
        z.writestr("astro_cycle2_overlap_mean.csv",csv_text(mean_rows,MEAN_FIELDS))
        z.writestr("astro_cycle2_overlap_provenance.json",json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"exported":str(destination),"policies":len(POLICIES),"seeds":len(SEEDS),
                      "policy_pairs":len(mean_rows),"per_seed_pair_rows":len(rows)},indent=2))

if __name__ == "__main__":
    main()