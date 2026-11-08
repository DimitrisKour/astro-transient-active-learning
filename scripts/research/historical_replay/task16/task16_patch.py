"""Single narrowly-scoped change to a verified frozen Task11 source tree."""
from pathlib import Path


def patch_source(root, overlay=None):
    runner = Path(root) / 'scripts/run_astrominn_closed_loop.py'
    data = runner.read_text(encoding='utf-8')
    old = '''    if set(args.strategies) - {"badge", "core_set", "fisher_batch"}:
        raise ValueError("Task11 runner permits only the three frozen Task11 acquisition policies")
'''
    new = '''    if set(args.strategies) - {"random", "entropy"}:
        raise ValueError("Task16 runner permits only random and entropy acquisition policies")
'''
    if data.count(old) != 1:
        raise ValueError(f'Task16 runner guard anchor occurs {data.count(old)} times; refusing patch')
    runner.write_text(data.replace(old, new, 1), encoding='utf-8')
    return [str(runner.relative_to(root))]
