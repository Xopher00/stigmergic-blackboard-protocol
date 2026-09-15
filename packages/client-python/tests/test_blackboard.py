"""
LocalBlackboard tests — the foundation every other layer (SbpAgent, SbpWorker) sits
on. Constructs a fresh LocalBlackboard directly per test, no singleton involved, no
LangChain, no network.
"""
import asyncio

import pytest

from sbp.blackboard import LocalBlackboard
from sbp.types import (
    EmitParams, SniffParams, RegisterScentParams, DeregisterScentResult,
    InscribeParams, ReadParams, EraseParams, EvaporateParams, InspectParams,
    ThresholdCondition, ImmortalDecay,
)


def _emit(bb, trail="t", type="e", intensity=0.8, **kw):
    return bb.emit(EmitParams(trail=trail, type=type, intensity=intensity, **kw))


def _set_value(bb, intensity, trail="t", type="e"):
    """Pin the aggregate a threshold condition sees: wipe the trail, emit one
    immortal pheromone so the observed value equals `intensity` exactly."""
    bb.evaporate(EvaporateParams(trail=trail))
    bb.emit(EmitParams(trail=trail, type=type, intensity=intensity, decay=ImmortalDecay(), merge_strategy="new"))


async def _eval(bb):
    await bb.evaluate_scents()
    await asyncio.sleep(0)  # dispatch is fire-and-forget; let the task run


class TestEmit:
    def test_creates_new_pheromone(self):
        bb = LocalBlackboard()
        result = _emit(bb)
        assert result.action == "created"
        assert result.new_intensity == 0.8
        assert len(bb.pheromones) == 1

    def test_reinforce_updates_existing_not_create_new(self):
        bb = LocalBlackboard()
        _emit(bb, payload={"x": 1}, intensity=0.5)
        result = _emit(bb, payload={"x": 1}, intensity=0.9)  # same trail/type/payload
        assert result.action == "reinforced"
        assert result.new_intensity == 0.9
        assert len(bb.pheromones) == 1

    def test_replace_overwrites_payload_and_tags(self):
        bb = LocalBlackboard()
        _emit(bb, payload={"x": 1}, tags=["old"])
        result = _emit(bb, payload={"x": 1}, intensity=0.3, tags=["new"], merge_strategy="replace")
        assert result.action == "replaced"
        p = next(iter(bb.pheromones.values()))
        assert p.tags == ["new"]

    def test_max_keeps_higher_intensity(self):
        bb = LocalBlackboard()
        _emit(bb, payload={"x": 1}, intensity=0.8)
        result = _emit(bb, payload={"x": 1}, intensity=0.3, merge_strategy="max")
        assert result.action == "merged"
        assert result.new_intensity == pytest.approx(0.8, abs=0.01)

    def test_add_sums_capped_at_one(self):
        bb = LocalBlackboard()
        _emit(bb, payload={"x": 1}, intensity=0.7)
        result = _emit(bb, payload={"x": 1}, intensity=0.7, merge_strategy="add")
        assert result.new_intensity == 1.0

    def test_new_strategy_never_merges(self):
        bb = LocalBlackboard()
        _emit(bb, payload={"x": 1})
        result = _emit(bb, payload={"x": 1}, merge_strategy="new")
        assert result.action == "created"
        assert len(bb.pheromones) == 2


class TestSniff:
    def test_filters_by_trail_and_type(self):
        bb = LocalBlackboard()
        _emit(bb, trail="a", type="x")
        _emit(bb, trail="b", type="y")
        result = bb.sniff(SniffParams(trails=["a"]))
        assert len(result.pheromones) == 1
        assert result.pheromones[0].trail == "a"

    def test_min_intensity_filter(self):
        bb = LocalBlackboard()
        _emit(bb, intensity=0.2)
        result = bb.sniff(SniffParams(min_intensity=0.5))
        assert result.pheromones == []

    def test_excludes_evaporated_unless_requested(self):
        bb = LocalBlackboard()
        _emit(bb, intensity=0.005)  # below default ttl_floor 0.01
        assert bb.sniff(SniffParams()).pheromones == []
        assert len(bb.sniff(SniffParams(include_evaporated=True)).pheromones) == 1


class TestRegisterDeregisterScent:
    def test_register_then_register_again_is_update(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        first = bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        second = bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        assert first.status == "registered"
        assert second.status == "updated"

    def test_deregister_found_vs_not_found(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        found = bb.deregister_scent("s1")
        missing = bb.deregister_scent("s1")
        assert found == DeregisterScentResult(scent_id="s1", status="deregistered")
        assert missing == DeregisterScentResult(scent_id="s1", status="not_found")


class TestInscribeReadErase:
    def test_inscribe_created_then_updated_with_version_increment(self):
        bb = LocalBlackboard()
        first = bb.inscribe(InscribeParams(trail="claims", key="k1", value={"v": 1}))
        second = bb.inscribe(InscribeParams(trail="claims", key="k1", value={"v": 2}))
        assert first.action == "created"
        assert first.version == 1
        assert second.action == "updated"
        assert second.version == 2

    def test_inscribe_distinct_keys_never_collide(self):
        bb = LocalBlackboard()
        r1 = bb.inscribe(InscribeParams(trail="claims", key="k1", value={}))
        r2 = bb.inscribe(InscribeParams(trail="claims", key="k2", value={}))
        assert r1.action == "created"
        assert r2.action == "created"

    def test_read_filters_by_trail_keys_prefix(self):
        bb = LocalBlackboard()
        bb.inscribe(InscribeParams(trail="a", key="alpha-1", value={}))
        bb.inscribe(InscribeParams(trail="a", key="beta-1", value={}))
        bb.inscribe(InscribeParams(trail="b", key="alpha-1", value={}))

        by_trail = bb.read(ReadParams(trails=["a"]))
        assert len(by_trail.traces) == 2

        by_key = bb.read(ReadParams(trails=["a"], keys=["alpha-1"]))
        assert len(by_key.traces) == 1

        by_prefix = bb.read(ReadParams(trails=["a"], prefix="alpha"))
        assert len(by_prefix.traces) == 1

    def test_erase_by_keys(self):
        bb = LocalBlackboard()
        bb.inscribe(InscribeParams(trail="a", key="k1", value={}))
        bb.inscribe(InscribeParams(trail="a", key="k2", value={}))
        result = bb.erase(EraseParams(trail="a", keys=["k1"]))
        assert result.erased_count == 1
        assert len(bb.read(ReadParams(trails=["a"])).traces) == 1


class TestEvaporate:
    def test_removes_matching_by_trail(self):
        bb = LocalBlackboard()
        _emit(bb, trail="a")
        _emit(bb, trail="b")
        result = bb.evaporate(EvaporateParams(trail="a"))
        assert result.evaporated_count == 1
        assert result.trails_affected == ["a"]
        assert len(bb.pheromones) == 1

    def test_below_intensity_filter(self):
        bb = LocalBlackboard()
        _emit(bb, intensity=0.9, payload={"a": 1}, decay=ImmortalDecay())
        _emit(bb, intensity=0.1, payload={"b": 1}, decay=ImmortalDecay())
        result = bb.evaporate(EvaporateParams(below_intensity=0.5))
        assert result.evaporated_count == 1
        assert len(bb.pheromones) == 1


class TestInspect:
    def test_respects_include_filter(self):
        bb = LocalBlackboard()
        _emit(bb, trail="a")
        only_stats = bb.inspect(InspectParams(include=["stats"]))
        assert only_stats.trails is None
        assert only_stats.scents is None
        assert only_stats.stats is not None
        assert only_stats.stats["total_pheromones"] == 1

    def test_trails_and_scents_reported(self):
        bb = LocalBlackboard()
        _emit(bb, trail="a")
        cond = ThresholdCondition(trail="a", signal_type="e", value=0.5)
        bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        result = bb.inspect(InspectParams(include=["trails", "scents"]))
        assert result.trails[0]["name"] == "a"
        assert result.scents[0]["scent_id"] == "s1"


class TestEvaluateScents:
    @pytest.mark.asyncio
    async def test_dispatches_trigger_when_threshold_met(self):
        bb = LocalBlackboard()
        fired = []

        async def handler(payload):
            fired.append(payload)

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        bb.subscribe("s1", handler)

        _emit(bb, trail="t", type="e", intensity=0.9)
        await bb.evaluate_scents()
        await asyncio.sleep(0)  # dispatch is fire-and-forget; let the task run

        assert len(fired) == 1
        assert fired[0].scent_id == "s1"

    @pytest.mark.asyncio
    async def test_respects_cooldown(self):
        bb = LocalBlackboard()
        fired = []

        async def handler(payload):
            fired.append(payload)

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond, cooldown_ms=60_000,
        ))
        bb.subscribe("s1", handler)

        _emit(bb, trail="t", type="e", intensity=0.9)
        await bb.evaluate_scents()
        await bb.evaluate_scents()  # immediately again -- should be suppressed by cooldown
        await asyncio.sleep(0)  # dispatch is fire-and-forget; let the task run

        assert len(fired) == 1

    @pytest.mark.asyncio
    async def test_slow_handler_does_not_block_other_scents(self):
        """Registered first so a sequential-await dispatch loop would reach it
        before the fast scent -- proves dispatch is fire-and-forget, not
        one-handler-blocks-the-next."""
        bb = LocalBlackboard()
        fired = []
        release = asyncio.Event()

        async def slow_handler(payload):
            await release.wait()
            fired.append(("slow", payload))

        async def fast_handler(payload):
            fired.append(("fast", payload))

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(scent_id="slow", agent_endpoint="test://a", condition=cond))
        bb.subscribe("slow", slow_handler)
        bb.register_scent(RegisterScentParams(scent_id="fast", agent_endpoint="test://b", condition=cond))
        bb.subscribe("fast", fast_handler)

        _emit(bb, trail="t", type="e", intensity=0.9)

        await asyncio.wait_for(bb.evaluate_scents(), timeout=0.5)
        await asyncio.sleep(0.05)  # let scheduled dispatch tasks get a tick

        assert any(tag == "fast" for tag, _ in fired)
        assert all(tag != "slow" for tag, _ in fired)  # still blocked on release

        release.set()
        await asyncio.sleep(0.05)
        assert any(tag == "slow" for tag, _ in fired)


class TestHysteresis:
    @pytest.mark.asyncio
    async def test_zero_hysteresis_is_unchanged_default_behavior(self):
        bb = LocalBlackboard()
        fired = []

        async def handler(payload):
            fired.append(payload)

        cond = ThresholdCondition(trail="hyst", signal_type="sig", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond, trigger_mode="edge_rising",
        ))
        bb.subscribe("s1", handler)

        _set_value(bb, 0.6, trail="hyst", type="sig")
        await _eval(bb)
        assert len(fired) == 1

        # plain edge_rising (hysteresis=0): any dip below threshold and back up re-fires
        _set_value(bb, 0.45, trail="hyst", type="sig")
        await _eval(bb)
        _set_value(bb, 0.6, trail="hyst", type="sig")
        await _eval(bb)
        assert len(fired) == 2

    @pytest.mark.asyncio
    async def test_hysteresis_suppresses_chatter_until_value_falls_below_band(self):
        bb = LocalBlackboard()
        fired = []

        async def handler(payload):
            fired.append(payload)

        cond = ThresholdCondition(trail="hyst", signal_type="sig", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond,
            trigger_mode="edge_rising", hysteresis=0.1,
        ))
        bb.subscribe("s1", handler)

        _set_value(bb, 0.6, trail="hyst", type="sig")
        await _eval(bb)
        assert len(fired) == 1  # first rising edge fires; scent is now disarmed

        # dip to 0.45 -- above threshold-hysteresis (0.4), not enough to re-arm
        _set_value(bb, 0.45, trail="hyst", type="sig")
        await _eval(bb)
        _set_value(bb, 0.6, trail="hyst", type="sig")
        await _eval(bb)
        assert len(fired) == 1, "still disarmed -- chatter near the threshold must not re-fire"

        # fall all the way below the band -- re-arms (met is False here, so no fire yet)
        _set_value(bb, 0.3, trail="hyst", type="sig")
        await _eval(bb)
        assert len(fired) == 1

        _set_value(bb, 0.6, trail="hyst", type="sig")
        await _eval(bb)
        assert len(fired) == 2, "re-armed and met again -- should fire"
