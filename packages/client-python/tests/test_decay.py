"""
Decay computation tests
"""

from sbp.decay import compute_intensity
from sbp.types import Pheromone, StepDecay


def _pheromone(**overrides) -> Pheromone:
    defaults = dict(
        id="p1",
        trail="t",
        type="e",
        emitted_at=0,
        last_reinforced_at=0,
        initial_intensity=1.0,
        decay_model=StepDecay(steps=[{"at_ms": 0, "intensity": 1.0}]),
    )
    defaults.update(overrides)
    return Pheromone(**defaults)


class TestStepDecay:
    def test_returns_correct_intensity_at_each_step_boundary(self) -> None:
        steps = [
            {"at_ms": 0, "intensity": 1.0},
            {"at_ms": 5000, "intensity": 0.7},
            {"at_ms": 10000, "intensity": 0.3},
        ]
        p = _pheromone(decay_model=StepDecay(steps=steps), initial_intensity=1.0)
        assert compute_intensity(p, 0) == 1.0
        assert compute_intensity(p, 5000) == 0.7
        assert compute_intensity(p, 10000) == 0.3

    def test_never_rises_above_initial_intensity_over_time(self) -> None:
        # Regression: a step intensity above initial_intensity must not make
        # the computed intensity increase as elapsed time grows.
        p = _pheromone(
            decay_model=StepDecay(steps=[{"at_ms": 1, "intensity": 1.0}]),
            initial_intensity=0.0,
        )
        assert compute_intensity(p, 0) == 0.0
        assert compute_intensity(p, 1) == 0.0

    def test_clamps_negative_step_intensity_at_zero(self) -> None:
        p = _pheromone(
            decay_model=StepDecay(steps=[{"at_ms": 0, "intensity": -0.5}]),
            initial_intensity=0.5,
        )
        assert compute_intensity(p, 1) == 0.0
