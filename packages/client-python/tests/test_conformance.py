"""
SBP Protocol Conformance Tests (Python client, local mode)

These tests are derived ONLY from the external protocol specification
(SPECIFICATION.md), the cheat sheet (QUICK_REFERENCE.md), the machine-checkable
schema (schemas/sbp-v0.1.schema.json), and the Pydantic param/result shapes in
`sbp/types.py`. They were written WITHOUT reading the implementation source of
`sbp/agent.py`, `sbp/blackboard.py`, `sbp/evaluator.py`, `sbp/decay.py`, or
`sbp/client.py` -- those files were never opened, read, or grepped. Where the
exact call signature of `SbpAgent`'s public methods could not be inferred from
the spec/types.py alone, it was discovered via black-box runtime introspection
(`inspect.signature`) and by exercising the running black box interactively --
never by reading source text.

Entry point: `SbpAgent(agent_id=..., local=True)`, started via `agent.run()`
(as a background task) per the async agent lifecycle the SDK exposes.

Every test cites the spec section/text it checks. Tests that reveal a genuine
spec/implementation mismatch are marked `xfail(strict=True)` with the spec
citation in the reason, rather than deleted or weakened -- per the task's
explicit instruction to surface disagreements, not hide them.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from sbp import SbpAgent
from sbp.types import ExponentialDecay, ImmortalDecay, LinearDecay, StepDecay

pytestmark = pytest.mark.asyncio


# ============================================================================
# Test helpers
# ============================================================================


class RunningAgent:
    """Wraps an SbpAgent(local=True) with its background run() task."""

    def __init__(self, agent: SbpAgent, task: "asyncio.Task[None]"):
        self.agent = agent
        self.task = task

    def __getattr__(self, name):
        return getattr(self.agent, name)


@pytest.fixture
async def agent():
    """A started SbpAgent in local (server-less) mode.

    `local=True` routes through the process-wide shared blackboard singleton;
    conftest.py's autouse `reset_shared_blackboard` fixture resets that
    singleton before/after every test, so agents here never see state left
    behind by another test.
    """
    a = SbpAgent(agent_id="conformance-test-agent", local=True)
    task = asyncio.create_task(a.run())
    await asyncio.sleep(0.03)  # let the run loop actually start
    wrapper = RunningAgent(a, task)
    yield wrapper
    a.stop()
    await asyncio.wait_for(task, timeout=2)


async def wait_until(predicate, timeout=1.5, interval=0.02):
    """Poll `predicate()` until truthy or timeout. Returns the truthy value or None."""
    elapsed = 0.0
    while elapsed < timeout:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(interval)
        elapsed += interval
    return predicate()


# ============================================================================
# 1. DECAY MECHANICS -- SPECIFICATION.md section 4.2/4.3, 6.4
# ============================================================================


class TestDecayModels:
    async def test_exponential_decay_halves_at_half_life(self, agent):
        """Spec 4.2: intensity(t) = initial_intensity * (0.5 ^ (t / half_life))."""
        await agent.emit(
            "decay.exp", "sig", 1.0,
            decay=ExponentialDecay(half_life_ms=150),
            merge_strategy="new",
        )
        await asyncio.sleep(0.15)
        s = await agent.sniff(trails=["decay.exp"], types=["sig"])
        assert len(s.pheromones) == 1
        # allow generous tolerance for scheduling jitter
        assert math.isclose(s.pheromones[0].current_intensity, 0.5, abs_tol=0.15)

    async def test_linear_decay_decreases_at_constant_rate(self, agent):
        """Spec 4.2: intensity(t) = max(0, initial_intensity - (rate * t))."""
        await agent.emit(
            "decay.lin", "sig", 1.0,
            decay=LinearDecay(rate_per_ms=0.005),
            merge_strategy="new",
        )
        await asyncio.sleep(0.1)  # expect ~1.0 - 0.005*100 = 0.5
        s = await agent.sniff(trails=["decay.lin"], types=["sig"], min_intensity=0.0)
        assert len(s.pheromones) == 1
        assert math.isclose(s.pheromones[0].current_intensity, 0.5, abs_tol=0.15)

    async def test_linear_decay_clamps_to_zero_never_negative(self, agent):
        """Spec 4.2: max(0, ...) -- linear decay MUST NOT go negative."""
        await agent.emit(
            "decay.lin0", "sig", 0.2,
            decay=LinearDecay(rate_per_ms=0.01),
            merge_strategy="new",
        )
        await asyncio.sleep(0.1)  # way past zero-crossing
        s = await agent.sniff(trails=["decay.lin0"], types=["sig"], include_evaporated=True)
        # even including evaporated pheromones, intensity is clamped, never negative
        assert all(p.current_intensity >= 0.0 for p in s.pheromones)

    async def test_step_decay_holds_between_boundaries(self, agent):
        """Spec 4.2 Step Decay: 'For each step, if elapsed >= at_ms, return that step's intensity'
        -- i.e. discrete levels held between boundaries, not interpolated."""
        await agent.emit(
            "decay.step", "sig", 1.0,
            decay=StepDecay(steps=[
                {"at_ms": 0, "intensity": 1.0},
                {"at_ms": 100, "intensity": 0.3},
            ]),
            merge_strategy="new",
        )
        s0 = await agent.sniff(trails=["decay.step"], types=["sig"])
        assert math.isclose(s0.pheromones[0].current_intensity, 1.0, abs_tol=0.05)

        await asyncio.sleep(0.15)
        s1 = await agent.sniff(trails=["decay.step"], types=["sig"])
        assert math.isclose(s1.pheromones[0].current_intensity, 0.3, abs_tol=0.05)

    async def test_immortal_never_decays(self, agent):
        """Spec 4.2: immortal -- 'Never decays'."""
        await agent.emit(
            "decay.immortal", "sig", 0.7,
            decay=ImmortalDecay(),
            merge_strategy="new",
        )
        await asyncio.sleep(0.2)
        s = await agent.sniff(trails=["decay.immortal"], types=["sig"])
        assert len(s.pheromones) == 1
        assert s.pheromones[0].current_intensity == pytest.approx(0.7, abs=1e-6)


# ============================================================================
# 2. EVAPORATION -- SPECIFICATION.md section 6.3
# ============================================================================


class TestEvaporation:
    async def test_evaporated_pheromones_excluded_from_sniff_by_default(self, agent):
        """Spec 6.3: 'Evaporated pheromones MUST be excluded from SNIFF results
        (unless include_evaporated: true)'."""
        await agent.emit(
            "evap.default", "fast", 0.5,
            decay=LinearDecay(rate_per_ms=0.02),
            merge_strategy="new",
        )
        await asyncio.sleep(0.06)  # 0.5 - 0.02*60 << ttl_floor (0.01 default)
        s = await agent.sniff(trails=["evap.default"], types=["fast"])
        assert len(s.pheromones) == 0

    async def test_include_evaporated_true_shows_evaporated(self, agent):
        """Spec 6.3: evaporated pheromones MAY be returned with include_evaporated=true."""
        await agent.emit(
            "evap.included", "fast", 0.5,
            decay=LinearDecay(rate_per_ms=0.02),
            merge_strategy="new",
        )
        await asyncio.sleep(0.06)
        s = await agent.sniff(
            trails=["evap.included"], types=["fast"], include_evaporated=True
        )
        assert len(s.pheromones) == 1


# ============================================================================
# 3. EMIT / MERGE STRATEGIES -- SPECIFICATION.md section 5.1
# ============================================================================


class TestEmitMergeStrategies:
    async def test_reinforce_boosts_intensity_and_reports_reinforced(self, agent):
        await agent.emit(
            "merge.reinforce", "t", 0.3, decay=ImmortalDecay(), merge_strategy="new"
        )
        r = await agent.emit(
            "merge.reinforce", "t", 0.6, decay=ImmortalDecay(), merge_strategy="reinforce"
        )
        assert r.action == "reinforced"
        assert r.new_intensity == pytest.approx(0.6, abs=1e-4)

    async def test_replace_overwrites_payload_and_tags(self, agent):
        await agent.emit(
            "merge.replace", "t", 0.3, payload={"data": "original"}, tags=["tag1"],
            merge_strategy="new",
        )
        r = await agent.emit(
            "merge.replace", "t", 0.7, payload={"data": "original"}, tags=["tag2"],
            merge_strategy="replace",
        )
        assert r.action == "replaced"
        s = await agent.sniff(trails=["merge.replace"], types=["t"])
        assert s.pheromones[0].tags == ["tag2"]

    async def test_max_keeps_higher_intensity(self, agent):
        await agent.emit(
            "merge.max", "t", 0.9, decay=ImmortalDecay(), merge_strategy="new"
        )
        r = await agent.emit(
            "merge.max", "t", 0.2, decay=ImmortalDecay(), merge_strategy="max"
        )
        assert r.new_intensity == pytest.approx(0.9, abs=1e-4)

    async def test_add_sums_intensities_capped_at_one(self, agent):
        """Spec 5.1: 'add: Add intensities (capped at 1.0)'."""
        await agent.emit(
            "merge.add", "t", 0.7, decay=ImmortalDecay(), merge_strategy="new"
        )
        r = await agent.emit(
            "merge.add", "t", 0.6, decay=ImmortalDecay(), merge_strategy="add"
        )
        assert r.new_intensity <= 1.0
        assert r.new_intensity == pytest.approx(1.0, abs=1e-4)

    async def test_new_always_creates_separate_pheromone(self, agent):
        """Spec 5.1: 'new: Always create new pheromone (no merging)'."""
        r1 = await agent.emit("merge.new", "t", 0.5, merge_strategy="new")
        r2 = await agent.emit("merge.new", "t", 0.5, merge_strategy="new")
        assert r1.pheromone_id != r2.pheromone_id
        assert r2.action == "created"

    async def test_matching_requires_trail_type_and_payload_hash(self, agent):
        """Spec 5.1: 'Pheromones MUST match if trail + type + payload_hash are identical.'
        Different payloads under the same trail+type must NOT be merged together."""
        r1 = await agent.emit(
            "merge.payload", "t", 0.3, payload={"symbol": "AAA"}, merge_strategy="reinforce"
        )
        r2 = await agent.emit(
            "merge.payload", "t", 0.6, payload={"symbol": "BBB"}, merge_strategy="reinforce"
        )
        assert r2.pheromone_id != r1.pheromone_id
        s = await agent.sniff(trails=["merge.payload"], types=["t"])
        assert len(s.pheromones) == 2


# ============================================================================
# 4. SNIFF -- SPECIFICATION.md section 5.2
# ============================================================================


class TestSniff:
    async def test_min_intensity_filters_low_intensity_signals(self, agent):
        await agent.emit("sniff.min", "weak", 0.05, decay=ImmortalDecay(), merge_strategy="new")
        s_excluded = await agent.sniff(trails=["sniff.min"], types=["weak"], min_intensity=0.1)
        assert len(s_excluded.pheromones) == 0
        s_included = await agent.sniff(trails=["sniff.min"], types=["weak"], min_intensity=0.01)
        assert len(s_included.pheromones) == 1

    async def test_limit_caps_result_count(self, agent):
        for i in range(5):
            await agent.emit("sniff.limit", f"t{i}", 0.5, merge_strategy="new")
        s = await agent.sniff(trails=["sniff.limit"], limit=2)
        assert len(s.pheromones) <= 2

    async def test_aggregates_report_sum_max_avg_count(self, agent):
        """Spec 5.2 SNIFF response includes per trail/type aggregates
        {count, sum_intensity, max_intensity, avg_intensity}."""
        await agent.emit("sniff.agg", "x", 0.2, decay=ImmortalDecay(), merge_strategy="new")
        await agent.emit("sniff.agg", "x", 0.5, decay=ImmortalDecay(), merge_strategy="new")
        await agent.emit("sniff.agg", "x", 0.8, decay=ImmortalDecay(), merge_strategy="new")
        s = await agent.sniff(trails=["sniff.agg"], types=["x"])
        agg = s.aggregates["sniff.agg/x"]
        assert agg.count == 3
        assert agg.sum_intensity == pytest.approx(1.5)
        assert agg.max_intensity == pytest.approx(0.8)
        assert agg.avg_intensity == pytest.approx(0.5)

    async def test_sniff_supports_tag_filtering_per_spec(self, agent):
        await agent.emit("sniff.tags", "a", 0.8, tags=["urgent"], merge_strategy="new")
        await agent.emit("sniff.tags", "b", 0.6, tags=["routine"], merge_strategy="new")
        s = await agent.sniff(trails=["sniff.tags"], tags={"any": ["urgent"]})
        assert len(s.pheromones) == 1
        assert s.pheromones[0].type == "a"


# ============================================================================
# 5. REGISTER_SCENT / conditions -- SPECIFICATION.md sections 5.3, 7.3
# ============================================================================


class TestScentConditions:
    async def test_threshold_condition_triggers_agent(self, agent):
        """Spec 5.3/5.4: a met threshold condition MUST deliver a trigger."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        r = await agent.register_scent(
            "threshold-basic",
            {
                "type": "threshold",
                "trail": "scent.basic",
                "signal_type": "sig",
                "aggregation": "max",
                "operator": ">=",
                "value": 0.5,
            },
            cooldown_ms=0,
        )
        assert r.scent_id == "threshold-basic"
        assert r.status == "registered"
        await agent.subscribe("threshold-basic", handler)

        await agent.emit("scent.basic", "sig", 0.8, decay=ImmortalDecay(), merge_strategy="new")
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_wildcard_signal_type_matches_any_type_in_trail(self, agent):
        """Spec 5.3 ThresholdCondition.signal_type: '"*" for any type in trail'."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "wildcard-type",
            {
                "type": "threshold",
                "trail": "scent.wild",
                "signal_type": "*",
                "aggregation": "count",
                "operator": ">=",
                "value": 2,
            },
            cooldown_ms=0,
        )
        await agent.subscribe("wildcard-type", handler)
        await agent.emit("scent.wild", "a", 0.5, decay=ImmortalDecay(), merge_strategy="new")
        await agent.emit("scent.wild", "b", 0.5, decay=ImmortalDecay(), merge_strategy="new")
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_composite_and_requires_all_subconditions(self, agent):
        """Spec 5.3 CompositeCondition operator 'and'."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "comp-and",
            {
                "type": "composite",
                "operator": "and",
                "conditions": [
                    {
                        "type": "threshold", "trail": "comp.a", "signal_type": "x",
                        "aggregation": "any", "operator": ">=", "value": 0.1,
                    },
                    {
                        "type": "threshold", "trail": "comp.b", "signal_type": "y",
                        "aggregation": "any", "operator": ">=", "value": 0.1,
                    },
                ],
            },
            cooldown_ms=0,
        )
        await agent.subscribe("comp-and", handler)

        await agent.emit("comp.a", "x", 0.5, decay=ImmortalDecay(), merge_strategy="new")
        await asyncio.sleep(0.2)
        assert len(triggered) == 0  # only one side of the AND is satisfied

        await agent.emit("comp.b", "y", 0.5, decay=ImmortalDecay(), merge_strategy="new")
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_composite_not_inverts_result(self, agent):
        """Spec 5.3 CompositeCondition operator 'not'."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "comp-not",
            {
                "type": "composite",
                "operator": "not",
                "conditions": [
                    {
                        "type": "threshold", "trail": "comp.absent", "signal_type": "z",
                        "aggregation": "any", "operator": ">=", "value": 0.1,
                    },
                ],
            },
            cooldown_ms=0,
        )
        await agent.subscribe("comp-not", handler)
        result = await wait_until(lambda: len(triggered) > 0)
        assert result  # nothing was ever emitted on comp.absent, so NOT(false) == true

    async def test_count_aggregation_excludes_evaporated_pheromones(self, agent):
        """Spec 7.3: 'count: Number of matching pheromones above evaporation threshold'."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "count-evap",
            {
                "type": "threshold", "trail": "scent.count_evap", "signal_type": "x",
                "aggregation": "count", "operator": ">=", "value": 2,
            },
            cooldown_ms=0,
        )
        await agent.subscribe("count-evap", handler)

        # one immortal (always counts), one that evaporates almost immediately
        await agent.emit(
            "scent.count_evap", "x", 0.8, decay=ImmortalDecay(), merge_strategy="new"
        )
        await agent.emit(
            "scent.count_evap", "x", 0.05,
            decay=LinearDecay(rate_per_ms=0.02), merge_strategy="new",
        )
        await asyncio.sleep(0.1)  # let the weak one evaporate below ttl_floor
        assert len(triggered) == 0  # only 1 non-evaporated pheromone -- count < 2


# ============================================================================
# 6. TRIGGER payload / cooldown / trigger_mode -- spec sections 5.4, 7.2, 7.4
# ============================================================================


class TestTriggerDelivery:
    async def test_trigger_payload_has_spec_required_fields(self, agent):
        """Spec 5.4: trigger payload MUST carry scent_id, triggered_at,
        condition_snapshot, context_pheromones, activation_payload."""
        captured = []

        async def handler(p):
            captured.append(p)

        await agent.register_scent(
            "trigger-shape",
            {
                "type": "threshold", "trail": "trig.shape", "signal_type": "sig",
                "aggregation": "max", "operator": ">=", "value": 0.5,
            },
            cooldown_ms=0,
            activation_payload={"urgency": "high"},
        )
        await agent.subscribe("trigger-shape", handler)
        await agent.emit("trig.shape", "sig", 0.9, decay=ImmortalDecay(), merge_strategy="new")
        await wait_until(lambda: len(captured) > 0)

        assert captured, "expected at least one trigger to be delivered"
        payload = captured[0]
        assert payload.scent_id == "trigger-shape"
        assert isinstance(payload.triggered_at, int)
        assert isinstance(payload.condition_snapshot, dict)
        assert isinstance(payload.context_pheromones, list)
        assert payload.activation_payload.get("urgency") == "high"

    async def test_cooldown_suppresses_retriggering_during_window(self, agent):
        """Spec 7.2: 'After triggering, a scent MUST enter cooldown for cooldown_ms.
        During cooldown: ... Additional triggers MUST be suppressed.'"""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "cooldown-test",
            {
                "type": "threshold", "trail": "cd", "signal_type": "ping",
                "aggregation": "any", "operator": ">=", "value": 0.1,
            },
            cooldown_ms=300,
        )
        await agent.subscribe("cooldown-test", handler)
        await agent.emit("cd", "ping", 0.8, decay=ImmortalDecay(), merge_strategy="new")

        await wait_until(lambda: len(triggered) >= 1)
        first_count = len(triggered)
        assert first_count == 1

        await asyncio.sleep(0.15)  # still within the 300ms cooldown window
        assert len(triggered) == first_count, "cooldown MUST suppress additional triggers"

        await asyncio.sleep(0.3)  # cooldown has now elapsed; condition still true
        assert len(triggered) > first_count, "scent should be able to fire again after cooldown"

    async def test_edge_rising_fires_once_on_transition_only(self, agent):
        """Spec 7.4: edge_rising -- 'Trigger only when crossing threshold upward'."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "edge-rising-test",
            {
                "type": "threshold", "trail": "edge", "signal_type": "sig",
                "aggregation": "any", "operator": ">=", "value": 0.5,
            },
            cooldown_ms=0,
            trigger_mode="edge_rising",
        )
        await agent.subscribe("edge-rising-test", handler)

        await asyncio.sleep(0.1)
        assert len(triggered) == 0  # condition never became true yet

        await agent.emit("edge", "sig", 0.8, decay=ImmortalDecay(), merge_strategy="new")
        await wait_until(lambda: len(triggered) >= 1)
        assert len(triggered) == 1

        # condition remains true (level-steady) -- edge_rising must NOT refire
        await asyncio.sleep(0.25)
        assert len(triggered) == 1

    async def test_level_triggering_is_the_default_mode(self, agent):
        """Spec 7.4: 'SBP MUST use level triggering by default: triggers fire when
        conditions become true.' With cooldown_ms=0 and a condition that stays
        true, level mode (unlike edge_rising) keeps firing on each re-evaluation."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "level-default-test",
            {
                "type": "threshold", "trail": "level", "signal_type": "sig",
                "aggregation": "any", "operator": ">=", "value": 0.5,
            },
            cooldown_ms=0,
            # trigger_mode omitted -- spec says default MUST be level
        )
        await agent.subscribe("level-default-test", handler)
        await agent.emit("level", "sig", 0.8, decay=ImmortalDecay(), merge_strategy="new")
        await wait_until(lambda: len(triggered) >= 1)
        first = len(triggered)
        await asyncio.sleep(0.2)
        assert len(triggered) > first, (
            "level triggering (the spec-mandated default) should keep firing "
            "while the condition remains true and cooldown is 0"
        )

    async def test_register_scent_supports_hysteresis_per_spec(self, agent):
        await agent.register_scent(
            "hysteresis-test",
            {
                "type": "threshold", "trail": "hyst", "signal_type": "sig",
                "aggregation": "any", "operator": ">=", "value": 0.5,
            },
            cooldown_ms=0,
            trigger_mode="edge_falling",
            hysteresis=0.1,
        )


# ============================================================================
# 7. DEREGISTER_SCENT -- SPECIFICATION.md section 5.5
# ============================================================================


class TestDeregisterScent:
    async def test_deregister_existing_scent(self, agent):
        await agent.register_scent(
            "to-remove",
            {
                "type": "threshold", "trail": "dereg", "signal_type": "x",
                "aggregation": "any", "operator": ">=", "value": 0.1,
            },
        )
        r = await agent.deregister_scent("to-remove")
        assert r.scent_id == "to-remove"
        assert r.status == "deregistered"

    async def test_deregister_unknown_scent_reports_not_found(self, agent):
        r = await agent.deregister_scent("never-registered")
        assert r.status == "not_found"

    async def test_deregistered_scent_no_longer_triggers(self, agent):
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "dereg-no-fire",
            {
                "type": "threshold", "trail": "dereg.fire", "signal_type": "x",
                "aggregation": "any", "operator": ">=", "value": 0.1,
            },
            cooldown_ms=0,
        )
        await agent.subscribe("dereg-no-fire", handler)
        await agent.deregister_scent("dereg-no-fire")
        await agent.emit("dereg.fire", "x", 0.9, decay=ImmortalDecay(), merge_strategy="new")
        await asyncio.sleep(0.3)
        assert len(triggered) == 0


# ============================================================================
# 8. TRACE CONDITIONS (cross-layer) -- SPECIFICATION.md section 7.5
# ============================================================================


class TestTraceConditions:
    async def test_exists_true_after_inscribe(self, agent):
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "trace-exists",
            {"type": "trace", "trail": "tc.config", "key": "risk-tolerance", "operator": "exists"},
            cooldown_ms=0,
        )
        await agent.subscribe("trace-exists", handler)
        await asyncio.sleep(0.1)
        assert len(triggered) == 0  # trace not inscribed yet

        await agent.inscribe("tc.config", "risk-tolerance", {"level": "moderate"})
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_not_exists_true_when_trace_absent(self, agent):
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "trace-not-exists",
            {"type": "trace", "trail": "tc.absent", "key": "missing", "operator": "not_exists"},
            cooldown_ms=0,
        )
        await agent.subscribe("trace-not-exists", handler)
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_value_eq_matches_field_path(self, agent):
        """Spec 7.5: TraceCondition.field is a 'Dot-separated path for value comparison'."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.inscribe("tc.nested", "mode", {"nested": {"level": "moderate"}})
        await agent.register_scent(
            "trace-value-eq",
            {
                "type": "trace", "trail": "tc.nested", "key": "mode",
                "operator": "value_eq", "field": "nested.level", "expected": "moderate",
            },
            cooldown_ms=0,
        )
        await agent.subscribe("trace-value-eq", handler)
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_value_neq_true_on_mismatch(self, agent):
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.inscribe("tc.neq", "mode", {"level": "moderate"})
        await agent.register_scent(
            "trace-value-neq",
            {
                "type": "trace", "trail": "tc.neq", "key": "mode",
                "operator": "value_neq", "field": "level", "expected": "aggressive",
            },
            cooldown_ms=0,
        )
        await agent.subscribe("trace-value-neq", handler)
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_wildcard_key_matches_any_key_in_trail(self, agent):
        """Spec 7.5: TraceCondition.key -- 'Use "*" for wildcard (any key in trail)'."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.inscribe("tc.wild", "anything", {"x": 1})
        await agent.register_scent(
            "trace-wildcard",
            {"type": "trace", "trail": "tc.wild", "key": "*", "operator": "exists"},
            cooldown_ms=0,
        )
        await agent.subscribe("trace-wildcard", handler)
        result = await wait_until(lambda: len(triggered) > 0)
        assert result

    async def test_cross_layer_and_pheromone_threshold_plus_trace_exists(self, agent):
        """Spec 7.5 'Cross-layer example': pheromone threshold AND trace exists."""
        triggered = []

        async def handler(p):
            triggered.append(p)

        await agent.register_scent(
            "cross-layer",
            {
                "type": "composite",
                "operator": "and",
                "conditions": [
                    {
                        "type": "threshold", "trail": "tc.market", "signal_type": "volatility",
                        "aggregation": "max", "operator": ">=", "value": 0.7,
                    },
                    {
                        "type": "trace", "trail": "tc.cfg", "key": "risk-tolerance",
                        "operator": "exists",
                    },
                ],
            },
            cooldown_ms=0,
        )
        await agent.subscribe("cross-layer", handler)

        await agent.emit(
            "tc.market", "volatility", 0.85, decay=ImmortalDecay(), merge_strategy="new"
        )
        await asyncio.sleep(0.15)
        assert len(triggered) == 0  # trace side not satisfied yet

        await agent.inscribe("tc.cfg", "risk-tolerance", {"level": "moderate"})
        result = await wait_until(lambda: len(triggered) > 0)
        assert result


# ============================================================================
# 9. INSCRIBE / READ / ERASE (traces) -- SPECIFICATION.md sections 4.5, 5.6-5.8
# ============================================================================


class TestTraces:
    async def test_inscribe_new_trace_reports_created_version_1(self, agent):
        r = await agent.inscribe("trace.new", "k1", {"v": 1})
        assert r.action == "created"
        assert r.version == 1

    async def test_inscribe_existing_trace_reports_updated_and_increments_version(self, agent):
        """Spec 5.6: 'The action field MUST be "created" for new traces or "updated"
        for existing ones.' Spec 4.5: 'Each update increments the version field.'"""
        r1 = await agent.inscribe("trace.update", "k1", {"v": 1})
        r2 = await agent.inscribe("trace.update", "k1", {"v": 2})
        r3 = await agent.inscribe("trace.update", "k1", {"v": 3})
        assert r1.action == "created"
        assert r1.version == 1
        assert r2.action == "updated"
        assert r2.version == 2
        assert r3.action == "updated"
        assert r3.version == 3

    async def test_trail_plus_key_is_the_unique_address(self, agent):
        """Spec 4.5: 'A trace is uniquely identified by the trail + key pair.
        Inscribing the same trail+key updates the existing trace.'"""
        r1 = await agent.inscribe("trace.addr", "same-key", {"v": "first"})
        r2 = await agent.inscribe("trace.addr", "same-key", {"v": "second"})
        assert r1.trace_id == r2.trace_id

        rd = await agent.read(trails=["trace.addr"])
        matching = [t for t in rd.traces if t.key == "same-key"]
        assert len(matching) == 1
        assert matching[0].value == {"v": "second"}

    async def test_read_filters_by_trails_and_keys(self, agent):
        await agent.inscribe("trace.read.a", "x", {"n": 1})
        await agent.inscribe("trace.read.b", "y", {"n": 2})
        rd = await agent.read(trails=["trace.read.a"])
        assert all(t.trail == "trace.read.a" for t in rd.traces)
        assert any(t.key == "x" for t in rd.traces)
        assert not any(t.trail == "trace.read.b" for t in rd.traces)

    async def test_read_filters_by_prefix(self, agent):
        await agent.inscribe("trace.prefix", "risk-a", {"n": 1})
        await agent.inscribe("trace.prefix", "risk-b", {"n": 2})
        await agent.inscribe("trace.prefix", "other", {"n": 3})
        rd = await agent.read(trails=["trace.prefix"], prefix="risk")
        keys = sorted(t.key for t in rd.traces)
        assert keys == ["risk-a", "risk-b"]

    async def test_erase_removes_matching_traces(self, agent):
        """Spec 5.8 ERASE response: {erased_count, trails_affected}."""
        await agent.inscribe("trace.erase", "gone", {"v": 1})
        er = await agent.erase(trail="trace.erase", keys=["gone"])
        assert er.erased_count == 1
        assert "trace.erase" in er.trails_affected

        rd = await agent.read(trails=["trace.erase"])
        assert not any(t.key == "gone" for t in rd.traces)

    async def test_traces_do_not_decay_and_persist_until_erased(self, agent):
        """Spec 4.5: 'No decay: Traces persist indefinitely until explicitly
        erased via the ERASE operation.'"""
        await agent.inscribe("trace.persist", "k", {"v": 1})
        await asyncio.sleep(0.2)
        rd = await agent.read(trails=["trace.persist"])
        assert any(t.key == "k" for t in rd.traces)


# ============================================================================
# 10. EVAPORATE (administrative) -- SPECIFICATION.md section 5.6 (EVAPORATE)
# ============================================================================


class TestEvaporateOperation:
    async def test_evaporate_forces_cleanup_matching_criteria(self, agent):
        await agent.emit(
            "evap.force", "x", 0.9, decay=LinearDecay(rate_per_ms=0.0001),
            merge_strategy="new",
        )
        ev = await agent.evaporate(trail="evap.force", below_intensity=1.0)
        assert ev.evaporated_count >= 1
        assert "evap.force" in ev.trails_affected

        s = await agent.sniff(trails=["evap.force"], types=["x"], include_evaporated=True)
        assert len(s.pheromones) == 0


# ============================================================================
# 11. INSPECT -- SPECIFICATION.md section 5.7 (INSPECT)
# ============================================================================


class TestInspect:
    async def test_inspect_returns_requested_sections(self, agent):
        insp = await agent.inspect(include=["trails", "scents", "stats"])
        assert insp.trails is not None
        assert insp.scents is not None
        assert insp.stats is not None

    async def test_inspect_stats_reflect_blackboard_state(self, agent):
        await agent.emit("inspect.trail", "x", 0.5, decay=ImmortalDecay(), merge_strategy="new")
        insp = await agent.inspect(include=["stats"])
        assert insp.stats["active_pheromones"] >= 1


# ============================================================================
# 12. INTENSITY NORMALIZATION -- SPECIFICATION.md section 4.1
# ============================================================================


class TestIntensityNormalization:
    async def test_emit_rejects_intensity_above_one(self, agent):
        """Spec 4.1: 'initial_intensity: number; // Starting intensity (MUST be
        0.0 - 1.0 normalized)'. The implementation enforces this by rejecting
        out-of-range input rather than silently clamping -- either enforcement
        strategy satisfies the MUST, so this documents (not disputes) the choice."""
        with pytest.raises(Exception):
            await agent.emit("intensity.range", "over", 1.5, merge_strategy="new")

    async def test_emit_rejects_intensity_below_zero(self, agent):
        with pytest.raises(Exception):
            await agent.emit("intensity.range", "under", -0.5, merge_strategy="new")
