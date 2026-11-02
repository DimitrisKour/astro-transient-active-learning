"""Task15 CLI and Slurm regression tests, without GPU/Slurm."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/adaptive'))
import task15


def test_release_all_no_implicit_seed(monkeypatch):
    calls=[]
    monkeypatch.setattr(task15,'release',lambda *args:calls.append(args))
    monkeypatch.setattr(sys,'argv',['task15.py','release','--all'])
    task15.main()
    assert calls==[(None,None,None,True)]


def test_pilot_default_seed_retained(monkeypatch):
    calls=[]
    monkeypatch.setattr(task15,'pilot',lambda policy,seed:calls.append((policy,seed)))
    monkeypatch.setattr(sys,'argv',['task15.py','pilot','direct'])
    task15.main()
    monkeypatch.setattr(sys,'argv',['task15.py','pilot','direct','--seed','246'])
    task15.main()
    assert calls==[('direct',135),('direct',246)]
