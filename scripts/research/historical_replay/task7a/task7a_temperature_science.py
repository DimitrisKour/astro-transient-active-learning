"""Task 7A scientific helper: fit one NLL-optimal scalar temperature from VALIDATION ONLY.

Saved evaluation logits are latest-alert/object-level. Pool inference logits are left
untouched on disk; acquisition receives an ephemeral logits/T view. No hidden-pool
labels, test predictions or training adjustments are accessed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CLASSES = ("AGN", "TDE", "SN I", "SN II", "CV")


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    z=np.asarray(logits,dtype=np.float64)/temperature
    high=np.max(z,axis=1)
    return float(np.mean(high + np.log(np.exp(z-high[:,None]).sum(axis=1)) - z[np.arange(len(y)),y]))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Same [0.05,20] log-temperature golden-section optimizer as Task6A audit."""
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(y,dtype=np.int64)
    if (x.ndim!=2 or x.shape[1]!=5 or y.shape!=(len(x),) or
        len(x)<5 or np.any(y<0) or np.any(y>=5) or not np.isfinite(x).all()):
        raise ValueError('Invalid five-class labeled validation logits')
    lo,hi=math.log(.05),math.log(20.)
    tau=(math.sqrt(5)-1)/2
    a,b=hi-tau*(hi-lo),lo+tau*(hi-lo)
    fa,fb=nll(x,y,math.exp(a)),nll(x,y,math.exp(b))
    for _ in range(115):
        if fa>fb:
            lo,a,fa=a,b,fb
            b=lo+tau*(hi-lo);fb=nll(x,y,math.exp(b))
        else:
            hi,b,fb=b,a,fa
            a=hi-tau*(hi-lo);fa=nll(x,y,math.exp(a))
    return float(math.exp((lo+hi)/2))


def load_calibration(logits_path: Path, summary_path: Path, checkpoint: Path) -> tuple[np.ndarray,np.ndarray,dict]:
    for file in (logits_path,summary_path,checkpoint):
        if not Path(file).is_file():raise FileNotFoundError(file)
    summary=json.loads(summary_path.read_text())
    if tuple(summary.get('model_class_names',[]))!=CLASSES:
        raise ValueError('Validation class order mismatch')
    if Path(summary.get('checkpoint_path','')).resolve()!=checkpoint.resolve():
        raise ValueError('Validation predictions do not belong to inference checkpoint')
    actual=digest(checkpoint)
    if summary.get('checkpoint_sha256')!=actual:
        raise ValueError('Evaluation checkpoint digest does not match trained checkpoint')
    if 'validation' not in [x['split'] for x in summary.get('splits',[])]:
        raise ValueError('Validation split missing from evaluation')
    with np.load(logits_path,allow_pickle=False) as f:
        ids=np.asarray(f['object_ids'],dtype=str)
        y=np.asarray(f['object_targets'],dtype=np.int64)
        values=np.asarray(f['object_values'],dtype=np.float64)
    if values.shape!=(len(ids),5) or y.shape!=(len(ids),) or len(set(ids.tolist()))!=len(ids):
        raise ValueError('Invalid latest-alert object validation predictions')
    return values,y,{'checkpoint_sha256':actual,'validation_sha256':digest(logits_path),
                     'n_validation_objects':int(len(ids)),'counts':np.bincount(y,minlength=5).tolist()}


def fitted_temperature(*,cycle:int,checkpoint:Path,evaluation_dir:Path,record_path:Path)->float:
    """Idempotent calibration from previous completed checkpoint's validation logits."""
    if not 1<=cycle<=9:raise ValueError('Invalid cycle')
    logits,y,provenance=load_calibration(evaluation_dir/'validation_predictions.npz',
                                         evaluation_dir/'evaluation_summary.json',checkpoint)
    expected={"schema":"task7a_temperature_v1", "cycle":cycle,
              "checkpoint":str(checkpoint), "temperature_source":"validation_object_logits_latest_alert",
              "objective":"unweighted_object_nll", "calibration":provenance}
    if record_path.is_file():
        stored=json.loads(record_path.read_text())
        for k,v in expected.items():
            if stored.get(k)!=v:raise ValueError(f'Stale temperature calibration record {record_path} field {k}')
        value=float(stored['temperature'])
        if not np.isfinite(value) or value<=0:raise ValueError('Invalid stored temperature')
        return value
    T=fit_temperature(logits,y)
    payload={**expected,'temperature':T,'nll_original':nll(logits,y,1.),
             'nll_scaled':nll(logits,y,T)}
    record_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=record_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    tmp.replace(record_path)
    return T
