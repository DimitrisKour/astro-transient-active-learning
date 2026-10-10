"""Object-budget planning, including an exact final remainder."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CycleBudget:
    max_cycles: int
    total_acquisitions: int
    final_cycle_size: int
    allow_partial_final_cycle: bool


def plan_cycle_budget(
    *,
    candidate_objects: int,
    acquisition_size: int,
    max_cycles: int,
    allow_partial_final_cycle=False,
    complete_pool=False,
) -> CycleBudget:
    for name, value in [
        ("candidate_objects", candidate_objects),
        ("acquisition_size", acquisition_size),
        ("max_cycles", max_cycles),
    ]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if complete_pool:
        max_cycles = math.ceil(candidate_objects / acquisition_size)
        allow_partial_final_cycle = True
    requested = max_cycles * acquisition_size
    if requested > candidate_objects:
        if (
            not allow_partial_final_cycle
            or (max_cycles - 1) * acquisition_size >= candidate_objects
        ):
            raise ValueError("Requested acquisition budget exceeds U0 before a valid final round.")
    total = min(requested, candidate_objects)
    return CycleBudget(
        max_cycles,
        total,
        total - (max_cycles - 1) * acquisition_size,
        bool(allow_partial_final_cycle),
    )
