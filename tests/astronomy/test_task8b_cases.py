import json
import csv
from pathlib import Path
import types
import sys
import pytest
from al_pipeline.core.analysis.seed_recovery import RecoveryCase,sha256
from al_pipeline.astronomy.analysis.seed_recovery import verify_source_case


def test_verified_source_case(tmp_path,monkeypatch):
    # Replace heavy AstroMiNN registry by a minimal test-only dataset/profile.
    fake_module=types.ModuleType('al_pipeline.astronomy.data.registry')
    base=tmp_path
    class Registry:
        @classmethod
        def from_toml(cls, path): return cls()
        def resolve_profile(self,name):
            assert name=='astrominn_ztf'
            return types.SimpleNamespace(dataset=types.SimpleNamespace(validation_manifest=base/'validation.csv'),
                                         profile=types.SimpleNamespace(options={'checkpoint':str(base/'models'/'published.pth')}))
    fake_module.DatasetRegistry=Registry
    monkeypatch.setitem(sys.modules,'al_pipeline.astronomy.data.registry',fake_module)
    (base/'models').mkdir();(base/'models'/'published.pth').write_bytes(b'model')
    (base/'validation.csv').write_bytes(b'ids\n1\n')
    persistent=base/'persistent'; data=base/'data';suite='astrominn_closed_loop_warm_v1'
    per=persistent/'runs'/suite/'entropy'/'seed_135';dat=data/'runs'/suite/'entropy'/'seed_135'
    split=persistent/'runs'/suite/'shared'/'historical_replay_split';split.mkdir(parents=True)
    (split/'initial_labeled_manifest.csv').write_text('a,b\n1,2\n')
    manifest=per/'manifests'/'train_cycle_004.csv';manifest.parent.mkdir(parents=True)
    manifest.write_text('a,b\n1,2\n')
    inc=dat/'cycle_003'/'evaluation';inc.mkdir(parents=True)
    cand=dat/'cycle_004'/'evaluation';cand.mkdir(parents=True)
    ck0=dat/'cycle_003'/'training'/'best_vit_tower.pth';ck1=dat/'cycle_004'/'training'/'best_vit_tower.pth'
    ck0.parent.mkdir(parents=True);ck1.parent.mkdir(parents=True)
    ck0.write_bytes(b'c0');ck1.write_bytes(b'c1')
    for path,ck in [(inc,ck0),(cand,ck1)]:
        (path/'evaluation_summary.json').write_text(json.dumps({'checkpoint_path':str(ck),
            'checkpoint_sha256':sha256(ck),'model_class_names':['AGN','TDE','SN I','SN II','CV']}))
    prior=per/'cycle_003'/'cycle_summary.json';prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({'cycle':3,'trained_checkpoint':str(ck0)}))
    summ=per/'cycle_004'/'cycle_summary.json';summ.parent.mkdir(parents=True)
    summ.write_text(json.dumps({'cumulative_manifest':str(manifest), 'trained_checkpoint':str(ck1),
                                'cycle':4,'strategy':'entropy'}))
    tr=dat/'cycle_004'/'training'/'training_summary.json'
    tr.write_text(json.dumps({'cycle':4,'seed':135,'strategy':'entropy', 'initial_checkpoint':str(ck0),
                              'checkpoint_path':str(ck1),'initialization':'pretrained',
                              'train_manifest_sha256':sha256(manifest),
                              'validation_manifest_sha256':sha256(base/'validation.csv'),
                              'policy':{'class_weight_exponent':.85,'new_object_weighting':'per_sample',
                              'class_weight_source':'training_manifest','augment_validation':False,
                              'legacy_seed_sequence':False, 'loader_seed':135,'max_epochs':50,
                              'patience':10,'batch_size':512,'validation_batch_size':2048,
                              'learning_rate':.0001,'overwrite':False}}))
    case=RecoveryCase('entropy',135,4,3)
    result=verify_source_case(case,persistent,data,base/'registry.toml')
    assert result.training_manifest == manifest
    assert result.initial_checkpoint == ck0
    tr.write_text(tr.read_text().replace('"loader_seed": 135','"loader_seed": 888'))
    with pytest.raises(ValueError,match='optimization protocol'):
        verify_source_case(case,persistent,data,base/'registry.toml')
