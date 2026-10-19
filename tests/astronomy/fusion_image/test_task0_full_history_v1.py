"""Non-GPU Task 0 invariants. Run in al-pipeline on Delta."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from al_pipeline.astronomy.fusion_image.recipe_training_resident import Recipe
from al_pipeline.astronomy.fusion_image.contracts import CLASS_NAMES


def load_script(path, name):
    spec=importlib.util.spec_from_file_location(name,Path(path))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_six_runs_have_identical_frozen_recipe_except_seed_and_identity():
    m=load_script('scripts/adaptive/task0_full_history.py','task0_runner')
    assert tuple(m.SEEDS)==(135,246,357)
    assert len(m.CASES)==6 and len(m.RUN_INIT)==6
    assert set(m.RUN_INIT.values())=={'pretrained','randomized'}
    for ini in m.INITIALIZATIONS:
        for seed in m.SEEDS:
            key=f'{ini}_s{seed}'
            recipe=m.CASES[key]
            assert recipe.seed==seed and recipe.name==key
            baseline=Recipe(name=key, seed=seed,lr_schedule='warmup_cosine',image_augmentation=True)
            assert recipe==baseline
            assert not recipe.ema
            assert recipe.class_weighting=='inverse_frequency'
            assert recipe.max_epochs==80 and recipe.patience==15


def test_object_label_multiset_is_order_independent_and_flags_changed_stages(tmp_path):
    m=load_script('scripts/adaptive/task0_vit_test_compare.py','task0_test_compare')
    assert tuple(CLASS_NAMES)==('AGN','TDE','SN I','SN II','CV')
    cols=['obj_id','target','alert']
    def put(name,records):
        p=tmp_path/name
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(records)
        return p
    records=[{'obj_id':'x','target':'SN Ia','alert':'1'},
             {'obj_id':'x','target':'SN Ia','alert':'2'},
             {'obj_id':'y','target':'TDE','alert':'1'}]
    p=put('one.csv',records)
    q=put('two.csv',list(reversed(records)))
    facts_a, labels_a=m.manifest_facts(p,'obj_id','target')
    facts_b, labels_b=m.manifest_facts(q,'obj_id','target')
    assert labels_a==labels_b and facts_a['canonical_stage_label_multiset_sha256']==facts_b['canonical_stage_label_multiset_sha256']
    changed=[dict(r) for r in records];changed[0]['alert']='3'
    r=put('changed.csv',changed)
    facts_c,_=m.manifest_facts(r,'obj_id','target')
    assert facts_a['canonical_stage_label_multiset_sha256']!=facts_c['canonical_stage_label_multiset_sha256']


def test_report_writes_expected_columns_and_rejects_forged_identity(tmp_path, monkeypatch):
    m=load_script('scripts/adaptive/task0_full_history.py','task0_runner_report')
    from al_pipeline.astronomy.fusion_image.contracts import sha256
    plan={'experiment':'task0_full_history_btsbot_pretrained_vs_randomized',
          'recipes':{'pretrained_s135':{'seed':135}},
          'run_initialization':{'pretrained_s135':'pretrained'}}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    case_dir=tmp_path/'pretrained_s135';case_dir.mkdir()
    (case_dir/'best_vit_tower.pth').write_bytes(b'checkpoint')
    (case_dir/'best_object_f1.pth').write_bytes(b'checkpoint2')
    metrics={k:0.5 for k in ('f1_macro','balanced_accuracy','accuracy','roc_auc_macro','average_precision_macro','ece','log_loss')}
    summary={'status':'completed','checkpoint_weights':'raw',
             'identity':{'case':'pretrained_s135','plan_sha256':sha256(tmp_path/'plan.json'),
                         'image_initialization':'pretrained','training_seed':135},
             'training_rows':29161,'validation_rows':14148,'validation_object_metrics':metrics,
             'best_epoch':2,'best_macro_f1_epoch':3,'best_validation_loss':.1,
             'best_macro_f1_during_fit':.5,'epochs_completed':5,
             'fit_seconds':100,'end_to_end_seconds':110,'setup_timing':{}}
    (case_dir/'training_summary.json').write_text(json.dumps(summary))
    m.report(tmp_path)
    report=tmp_path/'comparison.csv'
    assert report.is_file()
    with report.open() as f:
        records=list(csv.DictReader(f))
    assert records[0]['case']=='pretrained_s135' and records[0]['seed']=='135'
    assert float(records[0]['macro_average_precision'])==.5
    summary['identity']['image_initialization']='randomized'
    (case_dir/'training_summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='identity'):
        m.report(tmp_path)
