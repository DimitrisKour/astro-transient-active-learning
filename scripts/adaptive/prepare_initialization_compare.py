#!/usr/bin/env python
"""Prepare matched-data initialization checks without training/acquisition."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from al_pipeline.astronomy.initialization_compare import prepare_comparison
from al_pipeline.astronomy.training_diagnostics import load_plan

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--comparison-id', default='astrominn_initialization_compare_v1')
    for name in ('source-suite', 'source-artifact-root', 'persistent-root', 'data-root'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[135,246,357])
    p.add_argument('--max-cycle', type=int, default=9)
    p.add_argument('--profile', default='astrominn_ztf')
    p.add_argument('--legacy-code-dir', type=Path, default=Path('/projects/bcrv/ffontinelenunes/things_for_hailey'))
    a = vars(p.parse_args()); a['registry_path']=a.pop('registry'); a['profile_name']=a.pop('profile')
    path = prepare_comparison(**a); plan = load_plan(path)
    print(json.dumps({'plan':str(path), 'new_fits':len(plan['tasks']),
          'existing_random_fits_reused':len(plan['references']), 'existing_l0_checkpoints_reused':len(plan['seeds']),
          'shared_cycle_1_fits':4*len(plan['seeds']), 'seeds':plan['seeds'], 'max_cycle':plan['max_cycle'],
          'policies':plan['strategies'], 'new_modes':['restart_from_l0','warm_start'],
          'acquisition_executed':False,'interpretation':plan['interpretation']},indent=2))
if __name__ == '__main__': main()
