#!/usr/bin/env python3
"""Audit Task-4 checkpoint ancestry and export the unsmoothed AL trajectories.

Read-only; safe on incomplete experiments. This is diagnostic, NOT a mechanism
for selecting or accepting checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

POLICIES = ("random", "entropy", "entropy_diversity", "entropy_balanced_explore")
SEEDS = (135, 246, 357)


def collect(persistent: Path, data: Path, suite: str) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    audit: list[dict] = []
    for seed in SEEDS:
        for policy in POLICIES:
            state_root = persistent / "runs" / suite / policy / f"seed_{seed}"
            artifact_root = data / "runs" / suite / policy / f"seed_{seed}"
            state_file = state_root / "run_state.json"
            if not state_file.is_file():
                audit.append({"seed":seed, "policy":policy, "status":"not_started", "cycle":0})
                continue
            state = json.loads(state_file.read_text())
            completed = int(state.get("last_completed_cycle",0))
            active = state.get("active_cycle")
            audit.append({"seed":seed, "policy":policy,
                          "status":"in_progress" if active is not None else "idle",
                          "cycle": completed})
            used_ids: set[str] = set()
            bootstrap = data / "runs" / "astrominn_historical_replay_v1" / "shared" / f"seed_{seed}" / "bootstrap_training" / "best_vit_tower.pth"
            for c in range(1, completed+1):
                summary_file = state_root / f"cycle_{c:03d}" / "cycle_summary.json"
                if not summary_file.is_file():
                    raise ValueError(f"Completed cycle has no summary: {summary_file}")
                summary = json.loads(summary_file.read_text())
                previous = (bootstrap if c==1 else
                            artifact_root / f"cycle_{c-1:03d}" / "training" / "best_vit_tower.pth")
                for field in ("inference_checkpoint", "training_initial_checkpoint"):
                    if Path(summary[field]) != previous:
                        raise ValueError(f"Checkpoint ancestry mismatch at {policy} seed {seed} c{c}: {field}")
                own = artifact_root / f"cycle_{c:03d}" / "training" / "best_vit_tower.pth"
                if Path(summary["trained_checkpoint"]) != own:
                    raise ValueError(f"Cycle trained checkpoint mismatch at {policy} seed {seed} c{c}")
                ids = set(summary["selected_ids"])
                if len(ids) != len(summary["selected_ids"]) or used_ids & ids:
                    raise ValueError(f"Duplicate acquisition at {policy} seed {seed} c{c}")
                used_ids |= ids
                if len(ids) != (469 if c==9 else 473):
                    raise ValueError(f"Unexpected acquisition count at {policy} seed {seed} c{c}")
            history = state_root / "evaluation_history.json"
            if not history.exists():
                continue
            seen = set()
            for row in json.loads(history.read_text()):
                key=(int(row["cycle"]),row["split"])
                if key in seen:
                    raise ValueError(f"Duplicate evaluation record {key} in {history}")
                seen.add(key)
                records.append({"seed":seed, "policy":policy, "cycle":int(row["cycle"]),
                                "split":row["split"],"n_labeled":int(row["n_labeled"]),
                                "n_acquired":int(row["n_acquired"]), **row["metrics"]})
    return records,audit


def export_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--persistent',type=Path,default=Path('/projects/bcrv/dkourtesis/applecider_adaptive'))
    p.add_argument('--data',type=Path,default=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive'))
    p.add_argument('--suite',default='astrominn_closed_loop_warm_v1')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    report=args.out or args.persistent/'closed_loop'/args.suite/'report'
    values, audit=collect(args.persistent,args.data,args.suite)
    export_csv(report/'evaluation_history.csv',values)
    export_csv(report/'run_status.csv',audit)
    print(json.dumps({'evaluations':len(values),'runs':audit,'report':str(report)},indent=2))

if __name__=='__main__':
    main()
