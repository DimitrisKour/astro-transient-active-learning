import numpy as np
import pytest
from al_pipeline.core.analysis.retention_gate import ObjectPredictions, RetentionRule, paired_retention

CLASSES=('AGN','TDE','SN I','SN II','CV')


def make(y, preds, order=None):
    y=np.asarray(y,dtype=int)
    p=np.asarray(preds,dtype=int)
    v=np.full((len(y),5),-6.0)
    v[np.arange(len(y)),p]=6
    ids=[f'obj_{k}' for k in range(len(y))]
    if order is not None:
        ids=[ids[i] for i in order]
        return ObjectPredictions(tuple(ids),y[order],v[order])
    return ObjectPredictions(tuple(ids),y,v)


def fixed_data():
    # >=1 object in each of five classes; constant prediction is stable.
    labels=np.repeat(np.arange(5),20)
    return labels


def test_identity_accepted():
    y=fixed_data();a=make(y,y)
    result=paired_retention(a,a,RetentionRule(),CLASSES)
    assert result['accepted'] and result['shadow_rule']['would_accept']
    assert result['macro_delta']==0
    assert all(x['net_correct_loss']==0 for x in result['classes'].values())


def test_reject_large_macro_decline():
    y=fixed_data();p=y.copy();p[45:50]=0
    result=paired_retention(make(y,y),make(y,p),RetentionRule(),CLASSES)
    assert not result['accepted']
    assert not result['shadow_rule']['would_accept']


def test_shadow_class_violation_does_not_control_model():
    y=np.repeat(np.arange(5),200)
    p=y.copy()
    # class-specific loss 25 correct, below shadow 100 for SN I but above 3 for TDE.
    p[200:204]=0
    result=paired_retention(make(y,y),make(y,p),RetentionRule(),CLASSES)
    # Macro-F1 likely falls less than .01 on 1000 objects.
    assert result['accepted']
    assert result['shadow_rule']['would_accept'] is False
    assert 'TDE' in result['shadow_rule']['class_recall_violations']


def test_alignment_permuted_inputs():
    y=fixed_data()
    a=make(y,y)
    b=make(y,y,order=np.arange(len(y))[::-1])
    assert paired_retention(a,b,RetentionRule(),CLASSES)['accepted']


def test_different_objects_refused():
    y=fixed_data();a=make(y,y)
    b=ObjectPredictions(('wrong',)+a.ids[1:],a.targets,a.values)
    with pytest.raises(ValueError,match='object set mismatch'):
        paired_retention(a,b,RetentionRule(),CLASSES)


def test_different_labels_refused():
    y=fixed_data();a=make(y,y)
    labels=y.copy();labels[0]=1
    b=make(labels,labels)
    with pytest.raises(ValueError,match='targets mismatch'):
        paired_retention(a,b,RetentionRule(),CLASSES)


def test_duplicate_object_refused():
    y=fixed_data();v=make(y,y)
    with pytest.raises(ValueError,match='unique'):
        ObjectPredictions(v.ids[:-1]+(v.ids[0],),v.targets,v.values)


def test_zero_class_support_refused():
    labels=np.asarray([0,1,2,3],int)
    obj=make(labels,labels)
    with pytest.raises(ValueError,match='without positives'):
        paired_retention(obj,obj,RetentionRule(),CLASSES)


def test_shadow_rule_is_not_primary_gate():
    # Keep smaller macro threshold but make shadow BA tight; primary unaffected.
    y=np.repeat(np.arange(5),200)
    p=y.copy();p[400:410]=1
    a=make(y,y);b=make(y,p)
    result=paired_retention(a,b,RetentionRule(.025,.001,(1,1,1,1,1)),CLASSES)
    assert result['accepted']
    assert not result['shadow_rule']['would_accept']
    assert result['shadow_rule']['ba_violation']


def test_invalid_rule():
    with pytest.raises(ValueError):RetentionRule(-.01)
