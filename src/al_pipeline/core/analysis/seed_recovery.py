"""Classifier-independent contracts for a bounded first-rejection recovery experiment.

This package proposes retry cases; it never changes active-learning data or selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
from pathlib import Path

RULE = 'macro_tolerance_1pp'
SOURCE_SUITE = 'astrominn_closed_loop_warm_v1'
POLICIES = ('entropy', 'entropy_balanced_explore')
SEEDS = (135, 246, 357)
RETRY_SEEDS = (1001, 1002, 1003, 1004, 1005)
EXPECTED_CYCLES = {
    ('entropy', 135): 4, ('entropy', 246): 4, ('entropy', 357): 5,
    ('entropy_balanced_explore', 135): 4,
    ('entropy_balanced_explore', 246): 3,
    ('entropy_balanced_explore', 357): 6,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class RecoveryCase:
    policy: str
    seed: int
    rejected_cycle: int
    incumbent_cycle: int

    def __post_init__(self):
        if self.policy not in POLICIES or self.seed not in SEEDS:
            raise ValueError('Unknown Task4 trajectory')
        if self.rejected_cycle != EXPECTED_CYCLES[(self.policy, self.seed)]:
            raise ValueError('Case is inconsistent with approved Task8A first-rejection schedule')
        if self.incumbent_cycle != self.rejected_cycle - 1:
            raise ValueError('Only first-rejection cases with previous incumbent permitted')

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryJob:
    case: RecoveryCase
    retry_seed: int

    def __post_init__(self):
        if self.retry_seed not in RETRY_SEEDS:
            raise ValueError('Unapproved retry seed')

    @property
    def name(self) -> str:
        tag = 'ent' if self.case.policy == 'entropy' else 'bal'
        return f'{tag}_s{self.case.seed}_c{self.case.rejected_cycle}_r{self.retry_seed}'

    def as_dict(self) -> dict:
        return {**self.case.as_dict(), 'retry_seed': self.retry_seed, 'name': self.name}


def read_first_rejections(path: Path) -> tuple[RecoveryCase, ...]:
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty Task8A first-rejections file')
    selected = [r for r in rows if r.get('suite_key') == 'task4' and r.get('rule') == RULE]
    if len(selected) != len(EXPECTED_CYCLES):
        raise ValueError(f'Expected {len(EXPECTED_CYCLES)} Task4 rule rows; got {len(selected)}')
    cases = []
    for r in selected:
        if r.get('fully_accepted_saved_trajectory', '').lower() not in ('false', '0'):
            raise ValueError('First-rejection row was marked fully accepted')
        case = RecoveryCase(r['policy'], int(r['seed']),
                            int(r['first_rejected_cycle']), int(r['last_accepted_cycle']))
        cases.append(case)
    if {(c.policy, c.seed) for c in cases} != set(EXPECTED_CYCLES):
        raise ValueError('Task8A has missing or duplicate Task4 trajectories')
    return tuple(sorted(cases, key=lambda c: (POLICIES.index(c.policy), c.seed)))


def job_grid(cases: tuple[RecoveryCase, ...]) -> tuple[RecoveryJob, ...]:
    jobs = tuple(RecoveryJob(case, seed) for case in cases for seed in RETRY_SEEDS)
    if len(jobs) != 30 or len({j.name for j in jobs}) != 30:
        raise ValueError('Task8B expects exactly 30 distinct fits')
    return jobs


def macro_tolerance_accept(delta_macro_f1: float) -> bool:
    """Operationalization of the selected exploratory 1 pp Task8A rule."""
    return float(delta_macro_f1) >= -0.01 - 1e-12
