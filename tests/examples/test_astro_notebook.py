"""Offline CPU execution and non-misrepresentation checks for notebook 02."""
from __future__ import annotations

import ast
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT/'examples'/'02_applecider_astronomy_active_learning.ipynb'


def test_astronomy_notebook_cpu_executes_entirely_offline():
    nb = nbformat.read(NOTEBOOK,as_version=4)
    nbformat.validate(nb)
    codes = [c for c in nb.cells if c.cell_type == 'code']
    for cell in codes:
        ast.parse(cell.source)
    assert len(codes)>=10
    assert 'make_synthetic_demo' in ''.join(c.source for c in codes)
    assert 'synthetic' in ''.join(c.source for c in nb.cells).lower()
    result = NotebookClient(nb,timeout=450,kernel_name='python3',allow_errors=False,
                            resources={'metadata':{'path':str(ROOT)}}).execute()
    pngs = [o['data']['image/png'] for c in result.cells if c.cell_type=='code'
            for o in c.get('outputs',()) if o.get('data',{}).get('image/png')]
    assert len(pngs)>=5
    combined='\n'.join(str(o.get('text','')) for c in result.cells if c.cell_type=='code'
                       for o in c.get('outputs',()) if o.get('output_type')=='stream')
    assert 'NOT ZTF' in combined or 'REAL ASTRONOMICAL EXPORT' in ''.join(str(o) for c in result.cells for o in c.get('outputs',()))
    assert 'ONE sealed-test score:' in combined