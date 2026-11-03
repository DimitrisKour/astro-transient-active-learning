#!/usr/bin/env python3
"""Export complete Task17/Task18 final scientific results plus matched Task4 controls.

Read-only on Delta; no model loading, CUDA, or changes to frozen experiments.
Requires all 9 cycles and checks object acquisition counts and archival paths.
Never exports an unrevealed candidate-label table or training checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
T17='astrominn_task17_random_l0_v1'
T18='astrominn_task18_anchoral_v1'
T4='astrominn_closed_loop_warm_v1'
BASE_BOOT='astrominn_historical_replay_v1'
T17_BOOT='astrominn_task17_random_l0_bootstrap_v1'
SEEDS=(135,246,357)
CYCLES=tuple(range(1,10))
EVAL_FILES=('evaluation_summary.json','validation_per_class_metrics.json',
            'validation_object_confusion.csv','validation_sample_confusion.csv',
            'validation_object_metrics.json','validation_object_recall.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,
        default=Path('/u/dkourtesis/tasks17_18_final_review.zip'))
    a=parser.parse_args()
    output=a.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Existing results archive, refusing overwrite: {output}')

    groups=(
        ('task17',T17,('random','entropy')),
        ('task18',T18,('anchoral',)),
        ('reference_task4',T4,('random','entropy')),
    )
    files={}
    missing_optional=[]
    status=[]
    def add(path,where,*,required=True):
        path=Path(path)
        if not path.is_file():
            if required:raise FileNotFoundError(f'Required result missing: {path}')
            missing_optional.append(str(path))
            return
        if where in files:
            if files[where]!=path:raise RuntimeError('Duplicate archive name')
            return
        files[where]=path

    for group,suite,policies in groups:
        for policy in policies:
            for seed in SEEDS:
                relative=Path(suite)/policy/f'seed_{seed}'
                run=PERSISTENT/'runs'/relative
                art=DATA/'runs'/relative
                label=f'{group}/{policy}/seed_{seed}'
                state=read(run/'run_state.json')
                completed=int(state.get('last_completed_cycle',-1))
                if completed!=9 or state.get('active_cycle') is not None:
                    raise RuntimeError(f'{label}: incomplete trajectory ({completed}/9)')
                for name in ('run_state.json','experiment.json','evaluation_history.json'):
                    add(run/name,f'{label}/{name}')
                total_acquired=0
                for cycle in (0,*CYCLES):
                    key=f'cycle_{cycle:03d}'
                    if cycle:
                        selected=run/key/'selection.json'
                        summary=run/key/'cycle_summary.json'
                        revealed=run/key/'revealed_labels.json'
                        chosen=read(selected)['ids']
                        expected=473 if cycle<9 else 469
                        if len(chosen)!=expected or len(set(chosen))!=expected:
                            raise RuntimeError(f'{label}: cycle {cycle} expected {expected} unique IDs')
                        revealed_ids=read(revealed)['ids']
                        if chosen!=revealed_ids:
                            raise RuntimeError(f'{label}: cycle {cycle} selection and revealed IDs disagree')
                        details=read(summary)
                        if int(details.get('cycle',-1))!=cycle:
                            raise RuntimeError(f'{label}: cycle summary number mismatch at {cycle}')
                        total_acquired+=len(chosen)
                        for file in (selected,summary,revealed):
                            add(file,f'{label}/{key}/{file.name}')
                    eval_root=art/key/'evaluation'
                    for name in EVAL_FILES:
                        add(eval_root/name,f'{label}/{key}/evaluation/{name}',required=False)
                if total_acquired!=4253:
                    raise RuntimeError(f'{label}: total acquired {total_acquired} instead of 4253')
                status.append({'suite':suite,'group':group,'policy':policy,'seed':seed,
                               'completed_cycles':completed,'total_acquired':total_acquired})
    for group,suite in (('task17',T17),('task18',T18),('reference_task4',T4)):
        split=PERSISTENT/'runs'/suite/'shared/historical_replay_split'
        for name in ('split_summary.json','initial_labeled_manifest.csv'):
            add(split/name,f'{group}/shared/{name}',required=(group!='reference_task4'))
    controls=(
        ('task17',T17,('scientific_plan.json','preflight_audit.json',
                     'bootstrap_jobs.json','source_parent.json',
                     'frozen_manifest.json','pilot/seed_135/result.json')),
        ('task17_full9','astrominn_task17_random_l0_full9_controller_v1',
                    ('manifest.json','submitted.json')),
        ('task18',T18,('scientific_plan.json','source_parent.json',
                    'frozen_manifest.json','submitted.json',
                    'pilot/seed_135/result.json','repair_entropy_input_v1.json')),
    )
    for group,controller,names in controls:
        root=PERSISTENT/'closed_loop'/controller
        for name in names:
            add(root/name,f'{group}/control/{name}',required=False)

    for seed in SEEDS:
        for group,suite in (('task17',T17_BOOT),('reference_task4',BASE_BOOT)):
            root=DATA/'runs'/suite/'shared'/f'seed_{seed}/bootstrap_training'
            for name in ('training_summary.json','training_history.json'):
                add(root/name,f'{group}/bootstrap/seed_{seed}/{name}',required=False)

    manifest={'purpose':'Task17 random L0 / Task18 AnchorAL final review',
              'trajectories':status,
              'expected_trajectories':15,
              'cycles_per_trajectory':9,
              'included_task4_reference':True,
              'missing_optional_files':missing_optional,
              'file_count_excluding_manifest':len(files),
              'exclusions':['unrevealed candidate-label mapping',
                            'model checkpoints','raw images','test-set per-object predictions']}
    if len(status)!=15:raise RuntimeError('Expected 15 full trajectories')

    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,'x',ZIP_DEFLATED) as archive:
        archive.writestr('REVIEW_MANIFEST.json',json.dumps(manifest,indent=2))
        for name,path in sorted(files.items()):
            archive.write(path,arcname=name)
    print(json.dumps({'archive':str(output),'sha256':digest(output),
                      'full_trajectories':len(status),'cycle_records':len(status)*9,
                      'included_files':len(files),
                      'optional_files_absent':len(missing_optional)},indent=2))


if __name__=='__main__':
    main()
