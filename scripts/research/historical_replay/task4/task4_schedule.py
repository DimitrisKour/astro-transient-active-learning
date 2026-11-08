#!/usr/bin/env python3
"""Deterministic short-job scheduling for closed-loop Task 4.

Each row is one model fit. Each seed/policy forms its own nine-cycle dependency chain.
The 12 chains may run independently, subject to scheduler GPU availability.
The manifest emitted by this module contains no scheduler identifiers.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

SEEDS = (135, 246, 357)
POLICIES = ("random", "entropy", "entropy_diversity", "entropy_balanced_explore")


@dataclass(frozen=True)
class Job:
    seed: int
    cycle: int
    policy: str
    walltime: str


def validate_walltime(value: str) -> None:
    if not re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", value):
        raise ValueError(f"Invalid SLURM walltime: {value}")
    h, m, s = map(int, value.split(":"))
    if h * 3600 + m * 60 + s == 0:
        raise ValueError("SLURM walltime must be positive")


def schedule(*, cycles: int = 9, early_walltime: str = "00:15:00",
             late_walltime: str = "00:20:00", early_through: int = 2) -> list[Job]:
    if not 1 <= cycles <= 9:
        raise ValueError("cycles must be 1..9")
    if not 0 <= early_through <= 9:
        raise ValueError("early_through must be 0..9")
    validate_walltime(early_walltime)
    validate_walltime(late_walltime)
    jobs = [
        Job(seed, cycle, policy, early_walltime if cycle <= early_through else late_walltime)
        for seed in SEEDS
        for cycle in range(1, cycles + 1)
        for policy in POLICIES
    ]
    assert len(jobs) == len(SEEDS) * len(POLICIES) * cycles
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=9)
    p.add_argument("--early-walltime", default="00:15:00")
    p.add_argument("--late-walltime", default="00:20:00")
    p.add_argument("--early-through", type=int, default=2)
    args = p.parse_args()
    for job in schedule(cycles=args.cycles, early_walltime=args.early_walltime,
                        late_walltime=args.late_walltime, early_through=args.early_through):
        print(f"{job.seed}\t{job.cycle}\t{job.policy}\t{job.walltime}")


if __name__ == "__main__":
    main()
