#!/usr/bin/env python3
"""Read-only object-ID split audit for comparing legacy ViT and new AstroMiNN.

Run only after the recipe is frozen. No labels, model predictions, checkpoints,
or test-set metrics are accessed. Reports memberships and cross-overlap only.
"""
import argparse
import csv
import json
from pathlib import Path

def ids(path: Path, column: str) -> set[str]:
    with Path(path).open(newline="") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"{path}: required ID column {column!r} not present; got {reader.fieldnames}")
        ans=set()
        for row in reader:
            value=str(row[column]).strip()
            if not value:
                raise ValueError(f"{path}: empty object id")
            ans.add(value)
    if not ans:
        raise ValueError(f"Empty manifest: {path}")
    return ans

def main():
    p=argparse.ArgumentParser()
    for name in ('old_train','old_validation','old_test','new_train','new_validation','new_test'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--old-id-column',default='object_id')
    p.add_argument('--new-id-column',default='object_id')
    a=p.parse_args()
    paths={key:getattr(a,key) for key in (
        'old_train','old_validation','old_test','new_train','new_validation','new_test')}
    data={key:ids(path, a.old_id_column if key.startswith('old_') else a.new_id_column)
          for key,path in paths.items()}
    for prefix in ('old','new'):
        x=data[f'{prefix}_train'];y=data[f'{prefix}_validation'];z=data[f'{prefix}_test']
        if x & y or x & z or y & z:
            raise SystemExit(f'FAIL: {prefix} has train/validation/test object overlap')
    old_test=data['old_test'];new_test=data['new_test']
    common=old_test & new_test
    eligible=common - data['old_train'] - data['old_validation'] - data['new_train'] - data['new_validation']
    cross={
        o:{n:len(os & ns) for n,ns in data.items() if n.startswith('new_')}
        for o,os in data.items() if o.startswith('old_')
    }
    report={
        'set_counts':{k:len(v) for k,v in data.items()},
        'all_within_model_splits_disjoint':True,
        'same_test_object_sets':old_test==new_test,
        'common_test_objects':len(common),
        'common_test_objects_held_out_from_both_models':len(eligible),
        'old_to_new_split_overlaps':cross,
        'safe_common_test_reference':(len(eligible)==len(common) and len(common)>0),
        'note':'Object-membership audit only. Separately audit identical label taxonomy, object-level aggregation, source provenance and test-set tuning exclusion before comparing metrics.'
    }
    print(json.dumps(report,indent=2))
    if not report['safe_common_test_reference']:
        raise SystemExit('No valid shared test population; do not claim a head-to-head test comparison')

if __name__=='__main__':
    main()
