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


def test_per_class_export_reproduces_validation_macro_f1_and_support():
    pc = pd.read_csv(DATA / 'astro_recent_per_class.csv')
    f1 = pd.read_csv(DATA / 'astro_recent_validation.csv').query("metric == 'f1_macro'")
    provenance = json.loads((DATA / 'astro_recent_per_class_provenance.json').read_text())
    assert len(pc) == provenance['class_rows'] == 2550
    assert set(pc.class_name) == {'AGN', 'TDE', 'SN I', 'SN II', 'CV'}
    assert set(pc.cycle) == set(range(10))
    assert not pc.duplicated(['suite', 'strategy', 'seed', 'cycle', 'class_name']).any()
    assert pc[['precision', 'recall', 'f1']].apply(lambda col: col.between(0, 1).all()).all()
    assert pc.groupby('class_name').support.nunique().eq(1).all()
    assert pc.groupby('class_name').support.first().to_dict() == {
        'AGN': 435, 'CV': 92, 'SN I': 1480, 'SN II': 341, 'TDE': 17,
    }
    agg = pc.groupby(['suite','strategy','seed','cycle']).f1.mean()
    import numpy as np
    keys = pd.MultiIndex.from_frame(f1[['suite','strategy','seed','cycle']])
    assert np.allclose(agg.reindex(keys).to_numpy(), f1.value.to_numpy(),atol=1e-9)
    assert provenance['baseline_history_sha256'] == hashlib.sha256((DATA / 'astro_recent_validation.csv').read_bytes()).hexdigest()


def test_cycle2_cumulative_overlap_is_matched_and_symmetric():
    import numpy as np
    by_seed = pd.read_csv(DATA / 'astro_cycle2_overlap_by_seed.csv')
    means = pd.read_csv(DATA / 'astro_cycle2_overlap_mean.csv')
    provenance = json.loads((DATA / 'astro_cycle2_overlap_provenance.json').read_text())
    assert provenance['cycle'] == 2 and provenance['objects_per_cumulative_selection'] == 946
    assert len(by_seed) == 45 and len(means) == 15
    assert set(by_seed.seed) == {135,246,357}
    assert by_seed.groupby(['policy_a','policy_b']).seed.nunique().eq(3).all()
    assert not by_seed.duplicated(['seed','policy_a','policy_b']).any()
    assert by_seed.cumulative_intersection.between(0,946).all()
    assert by_seed.cumulative_union.between(946,1892).all()
    assert np.allclose(by_seed.cumulative_intersection / by_seed.cumulative_union,
                       by_seed.cumulative_jaccard)
    recomputed=by_seed.groupby(['policy_a','policy_b']).cumulative_jaccard.mean()
    stored=means.set_index(['policy_a','policy_b']).cumulative_mean
    assert np.allclose(recomputed.reindex(stored.index),stored)
    assert len(set(means.policy_a).union(means.policy_b)) == 5
    assert means[means.policy_a == means.policy_b].cumulative_mean.eq(1).all()
    assert {'object_id','selected_id','candidate_id'} .isdisjoint(by_seed.columns)