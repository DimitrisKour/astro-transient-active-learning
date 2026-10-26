import csv
import pytest
from al_pipeline.core.analysis.seed_recovery import (RecoveryCase, read_first_rejections,
    job_grid, macro_tolerance_accept, EXPECTED_CYCLES)


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def rows():
    return [{'suite_key':'task4','policy':policy,'seed':seed,'rule':'macro_tolerance_1pp',
             'first_rejected_cycle':cycle,'last_accepted_cycle':cycle-1,
             'fully_accepted_saved_trajectory':'False'}
            for (policy,seed),cycle in EXPECTED_CYCLES.items()]


def test_cases_and_grid(tmp_path):
    path=tmp_path/'first_rejections.csv';write_rows(path,rows())
    cases=read_first_rejections(path)
    assert len(cases)==6
    jobs=job_grid(cases)
    assert len(jobs)==30
    assert len(set(job.name for job in jobs))==30
    assert {j.retry_seed for j in jobs}=={1001,1002,1003,1004,1005}


def test_tampered_cycles_fail(tmp_path):
    data=rows();data[0]['first_rejected_cycle']=9
    path=tmp_path/'first_rejections.csv';write_rows(path,data)
    with pytest.raises(ValueError,match='schedule'):
        read_first_rejections(path)


def test_missing_or_duplicate_case_fail(tmp_path):
    data=rows();data[0]=data[1].copy()
    path=tmp_path/'first_rejections.csv';write_rows(path,data)
    with pytest.raises(ValueError,match='duplicate'):
        read_first_rejections(path)


def test_no_loose_gate():
    assert macro_tolerance_accept(-0.01)
    assert macro_tolerance_accept(-.0002)
    assert not macro_tolerance_accept(-.0101)
    with pytest.raises(ValueError):
        RecoveryCase('entropy',135,4,1)
