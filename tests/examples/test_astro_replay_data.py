"""Source and shape guardrails for bundled historical validation measurements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'examples/data'


def test_validation_only_replay_is_complete_and_matches_provenance():
    raw = (DATA / 'astro_recent_validation.csv').read_bytes()
    audit = json.loads((DATA / 'astro_recent_provenance.json').read_text())
    df = pd.read_csv(DATA / 'astro_recent_validation.csv')
    assert hashlib.sha256(raw).hexdigest() == audit['validation_csv_sha256']
    assert len(df) == audit['records'] == 1020
    assert df.groupby(['suite','strategy','seed']).ngroups == audit['trajectories'] == 51
    assert df.suite.nunique() == 8
    assert set(df.metric) == {'f1_macro', 'balanced_accuracy'}
    assert set(df.split) == {'validation'}
    assert not df.isna().any().any()
    assert not df.duplicated(['suite','strategy','seed','cycle','metric']).any()
    for _, sub in df.groupby(['suite','strategy','seed','metric']):
        assert set(sub.cycle) == set(range(10))
        assert sub.set_index('cycle').n_acquired_total.to_dict() == (
            {0:0, **{c:473*c for c in range(1,9)},9:4253})
    assert {'object_id','checkpoint_path','class_name','test'} .isdisjoint(df.columns)