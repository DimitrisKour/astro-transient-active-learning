import pytest

from al_pipeline.core.experiments.budget import plan_cycle_budget


def test_current_historical_pool_has_469_object_ninth_round():
    plan = plan_cycle_budget(
        candidate_objects=4253, acquisition_size=473, max_cycles=8, complete_pool=True
    )
    assert plan.max_cycles == 9
    assert plan.total_acquisitions == 4253
    assert plan.final_cycle_size == 469
    assert plan.allow_partial_final_cycle


def test_original_eight_round_budget_is_unchanged():
    plan = plan_cycle_budget(candidate_objects=4253, acquisition_size=473, max_cycles=8)
    assert plan.total_acquisitions == 3784
    assert plan.final_cycle_size == 473
    assert not plan.allow_partial_final_cycle


@pytest.mark.parametrize(
    "candidate,size,cycles,partial",
    [
        (5, 2, 3, False),
        (5, 2, 4, True),
        (0, 2, 1, True),
        (5, 0, 1, False),
    ],
)
def test_invalid_or_empty_final_round_rejected(candidate, size, cycles, partial):
    with pytest.raises(ValueError):
        plan_cycle_budget(
            candidate_objects=candidate,
            acquisition_size=size,
            max_cycles=cycles,
            allow_partial_final_cycle=partial,
        )
