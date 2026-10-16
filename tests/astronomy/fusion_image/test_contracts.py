from __future__ import annotations
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from al_pipeline.astronomy.fusion_image.contracts import (
    CLASS_NAMES, LABEL_INDEX, class_order_config, compare_splits, label_index, load_index,
)
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model, load_checkpoint
from al_pipeline.astronomy.fusion_image.data import image_metadata_from_npz, prepare_canonical_image


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test_original_five_class_order_and_mapping():
    assert CLASS_NAMES == ('AGN','TDE','SN I','SN II','CV')
    assert tuple(class_order_config()) == CLASS_NAMES
    assert label_index('SN IIP') == 3
    assert label_index('Cataclysmic') == 4
    with pytest.raises(ValueError): label_index('Stellar')


def test_duplicate_and_missing_index(tmp_path):
    p=tmp_path/'built_all.csv'
    write_csv(p,[{'object_id':'a','filepath':'a.npz'},{'object_id':'a','filepath':'b.npz'}])
    with pytest.raises(ValueError, match='Duplicate'): load_index(p)
    write_csv(p,[{'object_id':'a','filepath':'a.npz'},{'object_id':'b','filepath':'b.npz'}])
    idx=load_index(p)
    assert idx['a'] == tmp_path/'a.npz'
    assert idx['b'] == tmp_path/'b.npz'


def test_audit_rejects_cross_split_leakage(tmp_path):
    a=tmp_path/'train.csv';b=tmp_path/'val.csv'
    write_csv(a,[{'obj_id':'x','target':'AGN','alert':0}])
    write_csv(b,[{'obj_id':'x','target':'AGN','alert':0}])
    with pytest.raises(ValueError, match='overlap'):
        compare_splits(historical={'train':a,'validation':b},new_index={'x':tmp_path/'x.npz'})


def test_audit_reports_differences_without_dropping(tmp_path):
    a=tmp_path/'train.csv';b=tmp_path/'val.csv'
    write_csv(a,[{'obj_id':'x','target':'AGN','alert':0},
                 {'obj_id':'x','target':'AGN','alert':1},
                 {'obj_id':'y','target':'TDE','alert':0}])
    write_csv(b,[{'obj_id':'z','target':'SN Ia','alert':0}])
    out=compare_splits(historical={'train':a,'validation':b},new_index={'x':tmp_path/'x.npz'})
    assert out['historical_splits']['train']['objects']==2
    assert out['historical_splits']['train']['sample_rows']==3
    assert out['historical_splits']['train']['missing_new_npz']==1
    assert not out['full_historical_npz_coverage']


def test_reject_conflicting_labels(tmp_path):
    a=tmp_path/'train.csv'
    write_csv(a,[{'obj_id':'x','target':'AGN'}, {'obj_id':'x','target':'TDE'}])
    with pytest.raises(ValueError, match='Conflicting'):
        compare_splits(historical={'train':a},new_index={})


def test_model_specs_are_explicit():
    with pytest.raises(ValueError,match='git SHA'):
        NewImageModelSpec(variant='patch_images',image_initialization='randomized')
    with pytest.raises(ValueError,match='no image-pretraining'):
        NewImageModelSpec(variant='patch_images',image_initialization='pretrained',source_revision='abc')
    with pytest.raises(ValueError,match='local validated'):
        NewImageModelSpec(variant='btsbot_images',image_initialization='pretrained',source_revision='abc')


def test_metadata_builder_is_called_and_missing_stage_fails(tmp_path, monkeypatch):
    pkg=types.ModuleType('fusion_model_basics'); pkg.__path__=[]
    ds=types.ModuleType('fusion_model_basics.dataset')
    pp=types.ModuleType('fusion_model_basics.preprocessing')
    class StubDataset:
        def _build_metadata_tensor(self, meta, cols, jd, fid, provenance):
            assert len(cols)==55
            return torch.tensor(meta,dtype=torch.float32)
    ds.AlertStreamFusionDataset=StubDataset
    ds._as_str_list=lambda x:list(x)
    pp._prepare_astrominn_image=lambda x, cfg, augment=False:torch.from_numpy(x[:,7:56,7:56].copy())
    monkeypatch.setitem(sys.modules,'fusion_model_basics',pkg)
    monkeypatch.setitem(sys.modules,'fusion_model_basics.dataset',ds)
    monkeypatch.setitem(sys.modules,'fusion_model_basics.preprocessing',pp)
    p=tmp_path/'a.npz'
    np.savez(p,images=np.ones((2,3,63,63),np.float32),meta_data=np.ones((2,55),np.float32),
             meta_columns=np.array([f'c{i}' for i in range(55)]),
             jd=np.array([1.,2.]),fid=np.array([1,2]),
             provenance=np.array([{'alert_idx':0},{'alert_idx':3}],dtype=object))
    m,img,stage=image_metadata_from_npz(p,3,runtime_config={'image_encoder_type':'patch','patch_size':[49,7]})
    assert m.shape == (55,) and img.shape == (3,49,49) and stage['new_stage_index']==1
    with pytest.raises(ValueError,match='matches 0'):
        image_metadata_from_npz(p,1,runtime_config={'image_encoder_type':'patch','patch_size':[49,7]})


def test_fake_upstream_model_checkpoint_roundtrip(tmp_path, monkeypatch):
    pkg=types.ModuleType('fusion_model_basics');pkg.__path__=[]
    cfg=types.ModuleType('fusion_model_basics.config')
    enc=types.ModuleType('fusion_model_basics.astrominn_encoder')
    cfg.build_astrominn_config=lambda variant,**kwargs:kwargs
    class StubModel(torch.nn.Module):
        def __init__(self,config):
            super().__init__()
            self.class_names=list(config['classes'])
            self.coord_tower=torch.nn.Linear(2,2)
            self.vit_tower=torch.nn.Module()
            self.vit_tower.fc_out=torch.nn.Linear(3,5)
        def forward(self,metadata,image):
            return {'logits':self.vit_tower.fc_out(metadata[:,:3])}
    enc.AstroMiNNNPZClassifier=StubModel
    for k,v in {'fusion_model_basics':pkg,'fusion_model_basics.config':cfg,
                'fusion_model_basics.astrominn_encoder':enc}.items():
        monkeypatch.setitem(sys.modules,k,v)
    spec=NewImageModelSpec(variant='patch_images',image_initialization='randomized',source_revision='abc')
    model,_=build_model(spec,seed=42)
    ckpt=tmp_path/'best.pt'
    from dataclasses import asdict
    torch.save({'model_state':model.state_dict(),'model_spec':asdict(spec),'class_names':list(CLASS_NAMES)},ckpt)
    restored,_,_=load_checkpoint(ckpt,spec=spec,device='cpu',seed=99)
    assert torch.allclose(model.vit_tower.fc_out.weight, restored.vit_tower.fc_out.weight)
    with pytest.raises(ValueError,match='architecture'):
        load_checkpoint(ckpt,spec=NewImageModelSpec(variant='patch_images',image_initialization='randomized',source_revision='changed'),device='cpu',seed=99)


def test_upstream_and_historical_mapping_identical_when_repo_present():
    # Runs on Delta where al-pipeline full source tree is installed.
    try:
        from al_pipeline.astronomy.astrominn.training import MODEL_CLASS_NAMES, LABEL_TO_INDEX
    except ImportError:
        pytest.skip('Historical training module not installed in isolated overlay test')
    assert tuple(MODEL_CLASS_NAMES) == CLASS_NAMES
    assert dict(LABEL_TO_INDEX) == LABEL_INDEX


def test_preprocessing_guard_rejects_uncropped_patch_input(monkeypatch):
    """Prevent recurrence of 81 image tokens vs 49 positional embeddings."""
    pkg=types.ModuleType('fusion_model_basics');pkg.__path__=[]
    pp=types.ModuleType('fusion_model_basics.preprocessing')
    pp._prepare_astrominn_image=lambda img, config, augment=False: torch.as_tensor(img.copy())
    monkeypatch.setitem(sys.modules,'fusion_model_basics',pkg)
    monkeypatch.setitem(sys.modules,'fusion_model_basics.preprocessing',pp)
    raw=np.zeros((3,63,63),np.float32)
    with pytest.raises(RuntimeError, match=r'expected \(3,49,49\)'):
        prepare_canonical_image(raw,runtime_config={'image_encoder_type':'patch','patch_size':[49,7]})


def test_preprocessing_guard_keeps_full_btsbot_cutout(monkeypatch):
    pkg=types.ModuleType('fusion_model_basics');pkg.__path__=[]
    pp=types.ModuleType('fusion_model_basics.preprocessing')
    pp._prepare_astrominn_image=lambda img, config, augment=False: torch.as_tensor(img.copy())
    monkeypatch.setitem(sys.modules,'fusion_model_basics',pkg)
    monkeypatch.setitem(sys.modules,'fusion_model_basics.preprocessing',pp)
    raw=np.zeros((3,63,63),np.float32)
    result=prepare_canonical_image(raw,runtime_config={'image_encoder_type':'btsbot_convnext','cutout_size':63})
    assert result.shape == (3,63,63)
    with pytest.raises(ValueError, match='canonical cutout'):
        prepare_canonical_image(raw[:,2:,:],runtime_config={'image_encoder_type':'btsbot_convnext'})
