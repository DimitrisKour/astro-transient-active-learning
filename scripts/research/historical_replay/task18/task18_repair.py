#!/usr/bin/env python3
"""Audited one-line recovery of frozen Task18 after the first-cycle scorer type failure.

Run only when seed135 cycle1 failed before selection was written. Preserve
a copy of the original frozen manifest and original simulated.py. Replace the
frozen entropy input boundary and rehash it with an explicit repair audit.
Cancel the original 27-job chain and submit fresh held jobs with valid afterok
dependencies. Does not touch Task4, Task11, pilot, or L0 checkpoints.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import task18

JOB=task18.JOB
SOURCE=JOB/'source/src/al_pipeline/astronomy/experiments/simulated.py'
AUDIT=JOB/'repair_entropy_input_v1.json'
BAD='                scores = scorer.score(acquisition_data)\n'
GOOD='                scores = scorer.score(inference.predictions if self.config.strategy == "anchoral" else acquisition_data)\n'

def repair():
    if AUDIT.exists():
        task18.verify()
        print('Task18 repair already applied and frozen hash verified')
        return
    task18.verify()
    old=SOURCE.read_text()
    if old.count(BAD)!=1 or GOOD in old:
        raise RuntimeError('Expected failed AnchorAL scoring boundary not present exactly once')
    state=task18.PERSISTENT/'runs'/task18.SUITE/'anchoral/seed_135'
    for seed in task18.SEEDS:
        st=task18.PERSISTENT/'runs'/task18.SUITE/'anchoral'/f'seed_{seed}'
        if (st/'cycle_001/selection.json').exists() or (st/'cycle_001/revealed_labels.json').exists():
            raise RuntimeError('Already selected or revealed labels; manual recovery needed')
        if (st/'run_state.json').exists():
            d=task18.load(st/'run_state.json')
            if int(d.get('last_completed_cycle',0)) != 0:
                raise RuntimeError('Run already completed a cycle; manual recovery needed')
    original_manifest=task18.load(JOB/'frozen_manifest.json')
    archive=JOB/'frozen_manifest_before_repair.json'
    archive_source=JOB/'simulated_before_repair.py'
    if archive.exists() or archive_source.exists():
        raise RuntimeError('Unexpected existing repair archives')
    shutil.copy2(SOURCE,archive_source)
    task18.dump(archive,original_manifest)
    new=old.replace(BAD,GOOD,1)
    temp=SOURCE.with_suffix('.py.tmp')
    temp.write_text(new)
    temp.replace(SOURCE)
    task18.dump(JOB/'frozen_manifest.json',task18.hashes(JOB/'source'))
    task18.dump(AUDIT,{
        'reason':'AnchorALInput is selector-only; EntropyScorer needs ClassificationPredictions',
        'old_simulated_sha256':task18.sha(archive_source),
        'new_simulated_sha256':task18.sha(SOURCE),
        'old_manifest_sha256':task18.sha(archive),
        'fixed_statement':GOOD.strip(),
        'original_pilot_preserved':True,
        'no_candidate_oracle_labels_revealed':True,
    })
    task18.verify()
    print('Task18 frozen source repaired; previous hashes and source archived')

def resubmit(walltime):
    repair()
    ledger=JOB/'submitted.json'
    data=task18.load(ledger)
    jobs=data['jobs']
    if len(jobs)!=27 or {(j['seed'],j['cycle']) for j in jobs} != {
        (seed,cycle) for seed in task18.SEEDS for cycle in task18.CYCLES}:
        raise RuntimeError('Expected original 27-job ledger')
    backup=JOB/'submitted_failed_first_cycle.json'
    if backup.exists():
        raise RuntimeError('Already migrated failed job ledger; do not double submit')
    # Cancel old user-held jobs, including descendants of the failed first job.
    subprocess.run(['scancel',*[str(j['job_id']) for j in jobs]],check=True)
    shutil.copy2(ledger,backup)
    ledger.unlink()
    print('Archived original ledger and canceled its 27 jobs')
    task18.submit(walltime)
    print('Task18 new dependency chains submitted HELD; release seed135 cycle1 only')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('repair','resubmit'))
    p.add_argument('--walltime',default='00:20:00')
    a=p.parse_args()
    if a.command=='repair':repair()
    else:resubmit(a.walltime)

if __name__=='__main__':main()
