"""No GPU, external datasets or Delta required; checks byte-exact resident arrays."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

BASE=Path(__file__).resolve().parents[3]
F=BASE/'src/al_pipeline/astronomy/fusion_image/resident_cache.py'
TRAINER=BASE/'src/al_pipeline/astronomy/fusion_image/recipe_training_resident.py'
RUNNER=BASE/'scripts/research/fusion_image/recipe/fusion_pre_task0_resident.py'
PROBE=BASE/'scripts/research/fusion_image/recipe/fusion_resident_probe.py'
LAUNCHER=BASE/'scripts/research/fusion_image/recipe/submit_fusion_pre_task0_resident.sh'
spec=importlib.util.spec_from_file_location('resident_cache_v3_test',F)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeSplit:
    def __init__(self, root, split, positions):
        self.split=split
        self.positions=list(positions)
        self.images=np.load(root/'images.npy',mmap_mode='r')
        self.metadata=np.load(root/'metadata.npy',mmap_mode='r')
    def __len__(self):return len(self.positions)
    def get(self,i):
        p=self.positions[i]
        return self.images[p].copy(),self.metadata[p].copy()


def make_data(tmp_path):
    # The production manifest uses n=43309, but the storage helper supports
    # small synthetic datasets to test exact byte/position behavior.
    n=7
    images=np.arange(n*3*63*63,dtype=np.float32).reshape(n,3,63,63)
    meta=np.arange(n*55,dtype=np.float32).reshape(n,55)
    np.save(tmp_path/'images.npy',images)
    np.save(tmp_path/'metadata.npy',meta)
    (tmp_path/'fast_cache_manifest.json').write_text(json.dumps({
        'format_version':1,'source':module.EXPECTED_SOURCE,
        'labels_saved_in_cache':False,'test_split_accessed':False,
        'unique_stages':n,'images_nbytes':(tmp_path/'images.npy').stat().st_size,
        'metadata_nbytes':(tmp_path/'metadata.npy').stat().st_size}))
    return images,meta


def test_shared_resident_byte_exact_and_duplicate_positions(tmp_path):
    images,meta=make_data(tmp_path)
    tr=FakeSplit(tmp_path,'train',[6,0,6,2,4])
    val=FakeSplit(tmp_path,'validation',[1,5])
    before_tr=[tr.get(i) for i in range(len(tr))]
    before_val=[val.get(i) for i in range(len(val))]
    r=module.attach_resident_cache(tr,val,cache_root=tmp_path)
    assert r['shared_between_splits'] is True
    assert r['resident_bytes']==images.nbytes+meta.nbytes
    assert isinstance(tr.images,np.ndarray) and not isinstance(tr.images,np.memmap)
    assert tr.images is val.images and tr.metadata is val.metadata
    assert not tr.images.flags.writeable and not tr.metadata.flags.writeable
    for ds,befores in [(tr,before_tr),(val,before_val)]:
        for i,(a,b) in enumerate(befores):
            assert np.array_equal(a,ds.get(i)[0])
            assert np.array_equal(b,ds.get(i)[1])


def test_manifest_rejects_label_bearing_cache(tmp_path):
    make_data(tmp_path)
    m=json.loads((tmp_path/'fast_cache_manifest.json').read_text())
    m['labels_saved_in_cache']=True
    (tmp_path/'fast_cache_manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='label-bearing'):
        module.attach_resident_cache(FakeSplit(tmp_path,'train',[1]),
                                      FakeSplit(tmp_path,'validation',[2]),cache_root=tmp_path)


def test_rejects_wrong_split_or_out_of_range_positions(tmp_path):
    make_data(tmp_path)
    with pytest.raises(ValueError,match='split'):
        module.attach_resident_cache(FakeSplit(tmp_path,'validation',[0]),
                                      FakeSplit(tmp_path,'train',[1]),cache_root=tmp_path)
    with pytest.raises(ValueError,match='position'):
        module.attach_resident_cache(FakeSplit(tmp_path,'train',[100]),
                                      FakeSplit(tmp_path,'validation',[1]),cache_root=tmp_path)


def test_independent_frozen_plan_and_probe_gate():
    code=RUNNER.read_text()
    assert 'new_image_pre_task0_recipe_resident_v3' in code
    assert 'recipe_training_resident' in code
    assert "'resident_cache':sha256(resident_cache.__file__)" in code
    assert "'train':29161,'validation':14148" in code
    assert 'test_access' in code and 'acquisition' in code
    trainer=TRAINER.read_text()
    assert 'attach_resident_cache(train_ds, val_ds' in trainer
    assert 'resident_load_seconds' in trainer
    assert 'best_object_f1.pth' in trainer
    assert '--probe' in LAUNCHER.read_text()
    assert 'Resident-RAM probe must complete before training' in LAUNCHER.read_text()
    assert 'shared_host_resident_arrays_v3' in PROBE.read_text() or "resident['strategy']" in PROBE.read_text()
