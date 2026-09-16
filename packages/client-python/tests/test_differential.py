"""
Differential tests: this package's real decay/merge/condition code against the
shared frozen fixtures in differential/fixtures/. Same fixtures feed the TypeScript
consumer (packages/server/src/differential.test.ts); absolute tolerance 1e-9 everywhere.
"""
import json
from pathlib import Path

import pytest

from sbp.blackboard import LocalBlackboard
from sbp.decay import compute_intensity
from sbp.evaluator import EvaluationContext, evaluate_condition
from sbp.types import (
    EmitParams,
    ExponentialDecay,
    ImmortalDecay,
    LinearDecay,
    Pheromone,
    StepDecay,
    TagFilter,
    ThresholdCondition,
)

# tests/ -> client-python -> packages -> repo root
FIXTURES_DIR = Path(__file__).parents[3] / "differential" / "fixtures"

DECAY_MODEL_CLASSES = {
    "exponential": ExponentialDecay,
    "linear": LinearDecay,
    "step": StepDecay,
    "immortal": ImmortalDecay,
}


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES_DIR / name).read_text())


def _build_decay(d: dict):
    cls = DECAY_MODEL_CLASSES[d["type"]]
    return cls(**{k: v for k, v in d.items() if k != "type"})


def _build_tag_filter(d: dict) -> TagFilter:
    return TagFilter(**d)


def _build_condition(d: dict) -> ThresholdCondition:
    kwargs = {k: v for k, v in d.items() if k != "tags"}
    if "tags" in d:
        kwargs["tags"] = _build_tag_filter(d["tags"])
    return ThresholdCondition(**kwargs)


class TestDifferentialDecay:
    @pytest.mark.parametrize("fixture", _load("decay.json"), ids=lambda f: f["name"])
    def test_decay_matches_fixture(self, fixture):
        pheromone = Pheromone(
            id="p",
            trail="t",
            type="e",
            emitted_at=0,
            last_reinforced_at=0,
            initial_intensity=fixture["initial_intensity"],
            decay_model=_build_decay(fixture["decay_model"]),
        )
        assert compute_intensity(pheromone, fixture["elapsed_ms"]) == pytest.approx(
            fixture["expected_intensity"], abs=1e-9
        )


class TestDifferentialMerge:
    @pytest.mark.parametrize("fixture", _load("merge.json"), ids=lambda f: f["name"])
    def test_merge_matches_fixture(self, fixture):
        now_holder = [0]
        bb = LocalBlackboard()
        bb._now = lambda: now_holder[0]  # pin the clock; bump it between the two emits

        bb.emit(
            EmitParams(
                trail="t",
                type="e",
                intensity=fixture["existing_intensity"],
                decay=_build_decay(fixture["existing_decay"]),
                merge_strategy="new",
            )
        )
        now_holder[0] = fixture["elapsed_before_merge_ms"]
        result = bb.emit(
            EmitParams(
                trail="t",
                type="e",
                intensity=fixture["emitted_intensity"],
                merge_strategy=fixture["merge_strategy"],
            )
        )
        assert result.new_intensity == pytest.approx(fixture["expected_intensity"], abs=1e-9)


class TestDifferentialTrigger:
    @pytest.mark.parametrize("fixture", _load("trigger.json"), ids=lambda f: f["name"])
    def test_trigger_matches_fixture(self, fixture):
        pheromones = [
            Pheromone(
                id=f"p{i}",
                trail=p["trail"],
                type=p["type"],
                # Bake each pheromone's own elapsed_ms in: evaluate at now=0 with
                # negative timestamps so elapsed = 0 - (-elapsed_ms) = elapsed_ms.
                emitted_at=-p["elapsed_ms"],
                last_reinforced_at=-p["elapsed_ms"],
                initial_intensity=p["initial_intensity"],
                decay_model=_build_decay(p["decay_model"]),
                tags=p["tags"],
                ttl_floor=p["ttl_floor"],
            )
            for i, p in enumerate(fixture["pheromones"])
        ]
        result = evaluate_condition(
            _build_condition(fixture["condition"]), EvaluationContext(pheromones, 0)
        )
        assert result.met is fixture["expected_met"]
        assert result.value == pytest.approx(fixture["expected_value"], abs=1e-9)
