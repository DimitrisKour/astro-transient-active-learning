#!/usr/bin/env python3
"""Materialize deterministic L0/U0 split before verifying it; no model use."""
from __future__ import annotations
import argparse
from pathlib import Path
from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.experiments.historical_replay import materialize_astrominn_historical_replay_split

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--persistent',type=Path,required=True)
    p.add_argument('--suite',required=True)
    a=p.parse_args()
    registry=DatasetRegistry.from_toml(a.registry)
    resolved=registry.resolve_profile('astrominn_ztf')
    registry.validate_paths(resolved.dataset.name,require_historical_splits=True)
    split=a.persistent/'runs'/a.suite/'shared'/'historical_replay_split'
    result=materialize_astrominn_historical_replay_split(
        dataset=resolved.dataset,destination=split,
        initial_fraction=0.1,minimum_initial_per_class=5,
        split_seed=135,pool_id=a.suite+'_historical_u0',
        excluded_object_manifests=[a.persistent/'runs'/'boom_tns_new_labelled_inference_v1'/'hidden_labels.csv'],
    )
    print('Split materialized:',result.summary_path)

if __name__=='__main__':main()
