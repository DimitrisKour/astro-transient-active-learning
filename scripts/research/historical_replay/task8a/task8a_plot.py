#!/usr/bin/env python3
"""Optional read-only Task 8A summary figures from audit CSV outputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def plot(report: Path) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    paired = read_csv(report / 'adjacent_comparisons.csv')
    per_class = read_csv(report / 'per_class_changes.csv')
    first = read_csv(report / 'first_rejections.csv')
    cases = [(seed, policy) for policy in ('entropy', 'entropy_balanced_explore')
             for seed in (135, 246, 357)]
    labels = [f'{seed} · {"entropy" if policy == "entropy" else "balanced"}'
              for seed, policy in cases]
    task4 = [r for r in paired if r['suite_key'] == 'task4']
    matrix = np.full((6, 9), np.nan)
    for row in task4:
        key = (int(row['seed']), row['policy'])
        matrix[cases.index(key), int(row['candidate_cycle']) - 1] = 100 * float(row['delta_macro_f1'])

    paths = []
    fig, ax = plt.subplots(figsize=(10.8, 4.7), constrained_layout=True)
    lim = max(1, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap='coolwarm', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(9), [str(x) for x in range(1, 10)])
    ax.set_yticks(range(6), labels)
    ax.set_xlabel('Acquisition cycle')
    ax.set_title('Task 4 · validation macro-F1 changes between saved checkpoints (pp)')
    for row in range(6):
        for col in range(9):
            if np.isfinite(matrix[row, col]):
                ax.text(col, row, f'{matrix[row, col]:+.1f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='Δ macro-F1 (percentage points)', shrink=0.84)
    path = report / 'task8a_task4_macro_regression_map.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10.8, 4.5), constrained_layout=True)
    for name in ('TDE', 'SN II', 'CV'):
        data = [r for r in per_class if r['suite_key'] == 'task4' and r['class'] == name]
        by_cycle = [[100 * float(r['delta_recall']) for r in data if int(r['candidate_cycle']) == c]
                    for c in range(1, 10)]
        means = [float(np.mean(values)) if values else np.nan for values in by_cycle]
        ax.plot(range(1, 10), means, marker='o', label=name)
    ax.axhline(0, linestyle='--', linewidth=1, color='gray')
    ax.set_title('Task 4 · mean adjacent rare-class recall changes (6 trajectories)')
    ax.set_xlabel('Acquisition cycle')
    ax.set_ylabel('Mean Δ recall (percentage points)')
    ax.set_xticks(range(1, 10))
    ax.legend()
    path = report / 'task8a_task4_rare_recall_changes.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    rules = sorted({r['rule'] for r in first if r['suite_key'] == 'task4'})
    fig, ax = plt.subplots(figsize=(10.8, 5.0), constrained_layout=True)
    grid = np.full((len(rules), 6), np.nan)
    for r in first:
        if r['suite_key'] != 'task4':
            continue
        key = (int(r['seed']), r['policy'])
        value = r['first_rejected_cycle'].strip()
        if value:
            grid[rules.index(r['rule']), cases.index(key)] = int(value)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap='viridis_r', vmin=1, vmax=9, aspect='auto')
    ax.set_xticks(range(6), labels, rotation=20, ha='right')
    ax.set_yticks(range(len(rules)), rules)
    ax.set_title('Task 4 · first illustrative rejection cycle (blank = none)')
    for r in range(len(rules)):
        for c in range(6):
            if np.isfinite(grid[r, c]):
                ax.text(c, r, str(int(grid[r, c])), ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='First rejected cycle', shrink=0.9)
    path = report / 'task8a_task4_first_rejections.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path(
        '/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task8a_retrospective_acceptance_v1'))
    args = parser.parse_args()
    for output in plot(args.report):
        print(output)


if __name__ == '__main__':
    main()
