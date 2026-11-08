"""Validation-only Task9 prediction reader, aligned with Task8B object-level metrics.

Logits in validation_predictions.npz are converted to softmax probabilities.
No test prediction files are accepted here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

CLASSES=('AGN','TDE','SN I','SN II','CV')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_prediction(path,expected_checkpoint):
    path=Path(path)
    if path.name!='validation_predictions.npz':
        raise ValueError('Task9 may only read validation_predictions.npz')
    evaluation=json.loads((path.parent/'evaluation_summary.json').read_text(encoding='utf-8'))
    if Path(evaluation['checkpoint_path'])!=Path(expected_checkpoint):
        raise ValueError('Validation evaluation belongs to another checkpoint')
    if tuple(evaluation['model_class_names'])!=CLASSES:
        raise ValueError('Task9 class order differs from original AstroMiNN')
    if sha256(expected_checkpoint)!=evaluation['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 changed since evaluation')
    if not any(s.get('split')=='validation' for s in evaluation['splits']):
        raise ValueError('Validation split missing')
    with np.load(path,allow_pickle=False) as data:
        ids=np.asarray(data['object_ids'],dtype=str)
        y=np.asarray(data['object_targets'],dtype=np.int64)
        logits=np.asarray(data['object_values'],dtype=np.float64)
    if (len(ids)!=len(set(ids.tolist())) or len(y)!=len(ids) or logits.shape!=(len(y),5) or
        not np.isfinite(logits).all() or np.any(y<0) or np.any(y>4)):
        raise ValueError('Invalid validation object predictions')
    order=np.argsort(ids)
    logits=logits[order]
    logits-=logits.max(axis=1,keepdims=True)
    e=np.exp(logits);prob=e/e.sum(axis=1,keepdims=True)
    return ids[order],y[order],prob


def aligned(ref,other):
    if not np.array_equal(ref[0],other[0]) or not np.array_equal(ref[1],other[1]):
        raise ValueError('Validation object IDs or targets not aligned')


def scores(y,p):
    pred=np.argmax(p,axis=1);n_classes=5
    tp=np.array([np.sum((y==k)&(pred==k)) for k in range(n_classes)],dtype=float)
    support=np.array([np.sum(y==k) for k in range(n_classes)],dtype=float)
    if np.any(support==0):raise ValueError('Missing validation class')
    guesses=np.array([np.sum(pred==k) for k in range(n_classes)],dtype=float)
    f1=np.divide(2*tp,support+guesses,out=np.zeros(n_classes),where=(support+guesses)>0)
    recall=tp/support
    onehot=np.eye(n_classes)[y]
    return {'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),
            'accuracy':float(np.mean(pred==y)),
            'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-300)).mean()),
            'brier':float(np.mean(np.sum((p-onehot)**2,axis=1))),
            'recall':recall,'f1':f1,'support':support.astype(int),'pred':pred}
