#!/usr/bin/env python3
"""Execute the CPU astronomy notebook, save all figures, and check provenance.

    python scripts/examples/execute_astro_notebook.py --output astro_executed.ipynb

On a new clone the notebook uses its explicit synthetic fallback. A vetted
real NPZ in examples/data/ takes precedence; no GPU, checkpoints or Slurm.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLBACKEND','Agg')

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'examples'/'02_applecider_astronomy_active_learning.ipynb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    dest=args.output.expanduser().resolve()
    if dest == SOURCE.resolve() or dest.suffix.lower() != '.ipynb':
        parser.error('Output must be a different .ipynb file')
    if dest.exists():
        parser.error(f'Refusing to overwrite {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    nb=nbformat.read(SOURCE,as_version=4)
    nbformat.validate(nb)
    try:
        executed=NotebookClient(nb,timeout=450,kernel_name='python3',
                                resources={'metadata':{'path':str(ROOT)}},
                                allow_errors=False).execute()
        count=sum(bool(o.get('data',{}).get('image/png'))
                  for c in executed.cells if c.cell_type=='code' for o in c.get('outputs',()))
        if count < 5:
            raise ValueError(f'Expected at least five embedded figures; got {count}')
    except Exception:
        nbformat.write(nb,dest)
        print('Notebook execution failed; diagnostic notebook:',dest)
        raise
    nbformat.write(executed,dest)
    print('PASS: executed astronomy notebook saved:',dest)
    print('Embedded figures:',count)


if __name__=='__main__':
    main()