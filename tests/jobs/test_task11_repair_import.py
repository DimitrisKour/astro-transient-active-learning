import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/adaptive/task11_repair_import.py'
spec=importlib.util.spec_from_file_location('task11_repair_import', SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ORIGINAL='''"""Optional advanced selectors."""\nfrom al_pipeline.core.advanced_acquisition.badge import BadgeInput\nfrom al_pipeline.core.advanced_acquisition.core_set import CoreSetInput\nfrom al_pipeline.core.advanced_acquisition.ensemble_disagreement import (\n    EnsembleDisagreementScorer,\n)\n__all__ = [\n    "BadgeInput",\n    "CoreSetInput",\n    "EnsembleDisagreementScorer",\n]\n'''

def make_job(tmp_path):
    job=tmp_path/'closed_loop'/module.SUITE
    source=job/'source'
    target=source/module.REL
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL)
    (job/'scientific_plan.json').write_text(json.dumps({'suite':module.SUITE, 'fits':81, 'policies':['badge','core_set','fisher_batch']}))
    (job/'frozen_manifest.json').write_text(json.dumps(module.all_source_hashes(source)))
    return job,target

def test_repair_is_audited_idempotent_and_only_changes_import(tmp_path):
    job,target=make_job(tmp_path)
    before=target.read_text()
    result=module.repair(job)
    assert 'ensemble_disagreement' not in target.read_text()
    assert 'BadgeInput' in target.read_text()
    assert 'CoreSetInput' in target.read_text()
    assert (job/'repairs/advanced_acquisition_init.original.py').read_text()==before
    assert result['scientific_changes'] is False
    assert result==module.repair(job)
    assert module.all_source_hashes(job/'source')==module.json_read(job/'frozen_manifest.json')

def test_refuse_changed_snapshot(tmp_path):
    job,target=make_job(tmp_path)
    target.write_text('corruption')
    with pytest.raises(ValueError,match='hash mismatch'):
        module.repair(job)

def test_refuse_submitted(tmp_path):
    job,target=make_job(tmp_path)
    (job/'submitted.json').write_text('{}')
    with pytest.raises(RuntimeError,match='submitted'):
        module.repair(job)
    assert target.read_text()==ORIGINAL

def test_refuse_started_pilot(tmp_path):
    job,target=make_job(tmp_path)
    (job/'pilot/badge/seed_135').mkdir(parents=True)
    (job/'pilot/badge/seed_135/result.json').write_text('{}')
    with pytest.raises(RuntimeError,match='pilot'):
        module.repair(job)

def test_refuse_unexpected_init(tmp_path):
    job,target=make_job(tmp_path)
    unexpected=ORIGINAL.replace('EnsembleDisagreementScorer', 'OtherScorer')
    target.write_text(unexpected)
    (job/'frozen_manifest.json').write_text(json.dumps(module.all_source_hashes(job/'source')))
    with pytest.raises(ValueError,match='Unexpected'):
        module.repair(job)
