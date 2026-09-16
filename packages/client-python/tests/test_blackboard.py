"""
LocalBlackboard tests — the foundation every other layer (SbpAgent, SbpWorker) sits
on. Constructs a fresh LocalBlackboard directly per test, no singleton involved, no
LangChain, no network.
"""
import asyncio
import json

import pytest

from sbp.blackboard import LocalBlackboard
from sbp.types import (
    EmitParams, SniffParams, RegisterScentParams, DeregisterScentResult,
    InscribeParams, ReadParams, EraseParams, EvaporateParams, InspectParams,
    ThresholdCondition, CompositeCondition, ImmortalDecay,
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


def _stall_pheromones(bb):
    return [p for p in bb.pheromones.values() if p.trail == "system.errors"]


def _json_log_lines(capsys):
    """Structured log records on stdout: only lines that parse as JSON dicts
    count; legacy plain-text prints are skipped."""
    records = []
    for line in capsys.readouterr().out.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


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

    def test_register_params_default_trigger_mode_and_cooldown(self):
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        params = RegisterScentParams(scent_id="x", agent_endpoint="test://a", condition=cond)
        assert params.trigger_mode == "edge_rising"
        assert params.cooldown_ms == 1000

    def test_register_scent_defaults_cooldown_for_explicit_level_mode(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond,
            trigger_mode="level",
        ))
        assert bb.scents["s1"]["cooldown_ms"] == 1000


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


class TestOneActivationPerTrigger:
    @pytest.mark.asyncio
    async def test_always_true_level_condition_yields_one_activation_and_skip_count(self, capsys):
        bb = LocalBlackboard()
        fired = []
        release = asyncio.Event()

        async def handler(payload):
            fired.append(payload)
            await release.wait()

        # Explicit level/0ms so the guard is tested independently of default-value changes
        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond,
            trigger_mode="level", cooldown_ms=0,
        ))
        bb.subscribe("s1", handler)

        _set_value(bb, 0.9, trail="t", type="e")

        await _eval(bb)  # fires; handler starts and blocks on release
        for _ in range(4):
            await _eval(bb)  # condition still true, activation still running

        assert len(fired) == 1
        assert bb.scents["s1"]["running"] is True
        assert bb.scents["s1"]["skipped_fires"] == 4

        release.set()
        await asyncio.gather(*bb._dispatch_tasks)

        assert "4 fires were skipped" in capsys.readouterr().out
        assert bb.scents["s1"]["running"] is False
        assert bb.scents["s1"]["skipped_fires"] == 0

        # No deadlock: the scent must be usable again after the activation finishes
        release.clear()
        await _eval(bb)
        assert len(fired) == 2
        assert bb.scents["s1"]["running"] is True
        release.set()
        await asyncio.gather(*bb._dispatch_tasks)


class TestActivationTimeout:
    @pytest.mark.asyncio
    async def test_activation_timeout_cancels_stuck_handler(self, capsys):
        bb = LocalBlackboard()
        fired = []

        async def stuck_handler(payload):
            fired.append("stuck")
            await asyncio.Event().wait()  # never set -- hangs unless the timeout cancels it

        async def good_handler(payload):
            fired.append("good")

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond,
            trigger_mode="level", cooldown_ms=0, max_execution_ms=50,
        ))
        bb.subscribe("s1", stuck_handler)

        _set_value(bb, 0.9, trail="t", type="e")
        await _eval(bb)  # fires; handler enters and blocks
        assert fired == ["stuck"]

        # wait_for keeps a broken (never-resolving) dispatch from hanging pytest:
        # the working implementation must resolve it on its own at ~50ms.
        await asyncio.wait_for(asyncio.gather(*bb._dispatch_tasks), timeout=2.0)

        assert "timed out" in capsys.readouterr().out
        assert bb.scents["s1"]["running"] is False

        # The scent recovers: the next evaluation fires and completes normally.
        bb.unsubscribe("s1")  # timed-out handler is still registered
        bb.subscribe("s1", good_handler)
        await _eval(bb)
        await asyncio.gather(*bb._dispatch_tasks)

        assert fired == ["stuck", "good"]
        assert bb.scents["s1"]["running"] is False


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
            cooldown_ms=0,  # subject is hysteresis, not cooldown; rapid re-fires by design
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
            cooldown_ms=0,  # subject is hysteresis, not cooldown; rapid re-fires by design
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


class TestScentNamespacingAndOwnership:
    def test_register_auto_prefixes_bare_scent_id(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        result = bb.register_scent(
            RegisterScentParams(scent_id="watch", agent_endpoint="test://a", condition=cond),
            agent_id="agent-a",
        )
        assert result.scent_id == "agent-a:watch"
        assert "agent-a:watch" in bb.scents
        assert "watch" not in bb.scents

    def test_register_with_own_prefix_does_not_double_prefix(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        result = bb.register_scent(
            RegisterScentParams(scent_id="agent-a:watch", agent_endpoint="test://a", condition=cond),
            agent_id="agent-a",
        )
        assert result.scent_id == "agent-a:watch"
        assert "agent-a:watch" in bb.scents
        assert "agent-a:agent-a:watch" not in bb.scents

    def test_register_rejects_foreign_prefix(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        with pytest.raises(PermissionError, match="foreign prefix"):
            bb.register_scent(
                RegisterScentParams(scent_id="agent-b:watch", agent_endpoint="test://a", condition=cond),
                agent_id="agent-a",
            )
        assert "agent-b:watch" not in bb.scents
        assert "agent-a:agent-b:watch" not in bb.scents

    def test_agent_cannot_deregister_another_agents_scent(self):
        bb = LocalBlackboard()
        fired = []
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)

        async def handler(payload):
            fired.append(payload)

        bb.register_scent(
            RegisterScentParams(scent_id="watch", agent_endpoint="test://b", condition=cond),
            agent_id="agent-b",
        )
        bb.subscribe("watch", handler, agent_id="agent-b")

        with pytest.raises(PermissionError):
            bb.deregister_scent("agent-b:watch", agent_id="agent-a")

        # the point of the feature: B's scent AND its trigger wiring survive A's attack
        assert "agent-b:watch" in bb.scents
        assert "agent-b:watch" in bb.handlers

        # positive control: the block must not be vacuous -- the owner can still tear down
        result = bb.deregister_scent("agent-b:watch", agent_id="agent-b")
        assert result.status == "deregistered"

    def test_bare_name_deregister_resolves_into_own_namespace(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        bb.register_scent(
            RegisterScentParams(scent_id="watch", agent_endpoint="test://b", condition=cond),
            agent_id="agent-b",
        )
        # a bare name only ever addresses the caller's own namespace
        result = bb.deregister_scent("watch", agent_id="agent-a")
        assert result.status == "not_found"
        assert "agent-b:watch" in bb.scents


class TestSubscribeOverwrite:
    def test_double_subscribe_same_id_raises(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)

        async def handler1(payload):
            pass

        async def handler2(payload):
            pass

        bb.register_scent(
            RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond),
            agent_id="agent-a",
        )
        bb.subscribe("s1", handler1, agent_id="agent-a")
        # same resolved id via the already-prefixed form -- must hit the same guard
        with pytest.raises(ValueError, match="already subscribed"):
            bb.subscribe("agent-a:s1", handler2, agent_id="agent-a")

    @pytest.mark.asyncio
    async def test_unsubscribe_then_subscribe_replaces_handler(self):
        bb = LocalBlackboard()
        fired1 = []
        fired2 = []

        async def handler1(payload):
            fired1.append(payload)

        async def handler2(payload):
            fired2.append(payload)

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(
            RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond,
                                trigger_mode="level", cooldown_ms=0),
            agent_id="agent-a",
        )
        bb.subscribe("s1", handler1, agent_id="agent-a")
        with pytest.raises(ValueError, match="already subscribed"):
            bb.subscribe("agent-a:s1", handler2, agent_id="agent-a")

        bb.unsubscribe("agent-a:s1", agent_id="agent-a")
        bb.subscribe("agent-a:s1", handler2, agent_id="agent-a")

        _set_value(bb, 0.9, trail="t", type="e")
        await _eval(bb)
        await asyncio.gather(*bb._dispatch_tasks)

        assert fired1 == []
        assert len(fired2) == 1
        assert fired2[0].scent_id == "agent-a:s1"


class TestFreezeList:
    def test_frozen_prefix_refuses_mutating_calls(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)

        async def handler(payload):
            pass

        bb.register_scent(
            RegisterScentParams(scent_id="x", agent_endpoint="test://fz", condition=cond),
            agent_id="fz",
        )
        bb.freeze("fz")

        with pytest.raises(PermissionError, match="frozen"):
            bb.register_scent(
                RegisterScentParams(scent_id="y", agent_endpoint="test://fz", condition=cond),
                agent_id="fz",
            )
        with pytest.raises(PermissionError, match="frozen"):
            bb.deregister_scent("fz:x", agent_id="fz")
        with pytest.raises(PermissionError, match="frozen"):
            bb.subscribe("fz:x", handler, agent_id="fz")
        with pytest.raises(PermissionError, match="frozen"):
            bb.unsubscribe("fz:x", agent_id="fz")

    def test_unfreeze_restores_normal_operation(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        fired = []

        async def handler(payload):
            fired.append(payload)

        bb.register_scent(
            RegisterScentParams(scent_id="x", agent_endpoint="test://fz", condition=cond),
            agent_id="fz",
        )
        bb.freeze("fz")
        with pytest.raises(PermissionError, match="frozen"):
            bb.register_scent(
                RegisterScentParams(scent_id="y", agent_endpoint="test://fz", condition=cond),
                agent_id="fz",
            )
        with pytest.raises(PermissionError, match="frozen"):
            bb.deregister_scent("fz:x", agent_id="fz")

        bb.unfreeze("fz")

        reg = bb.register_scent(
            RegisterScentParams(scent_id="y", agent_endpoint="test://fz", condition=cond),
            agent_id="fz",
        )
        assert reg.scent_id == "fz:y"
        assert "fz:y" in bb.scents
        bb.subscribe("fz:x", handler, agent_id="fz")
        bb.unsubscribe("fz:x", agent_id="fz")
        dereg = bb.deregister_scent("fz:x", agent_id="fz")
        assert dereg.status == "deregistered"

    @pytest.mark.asyncio
    async def test_freeze_blocks_dispatch_unfreeze_restores_it(self):
        bb = LocalBlackboard()
        fired = []

        async def handler(payload):
            fired.append(payload)

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(
            RegisterScentParams(scent_id="watch", agent_endpoint="test://fz", condition=cond,
                                trigger_mode="level", cooldown_ms=0),
            agent_id="fz",
        )
        bb.subscribe("watch", handler, agent_id="fz")

        # level + 0ms cooldown re-fires on every tick while the condition holds --
        # edge_rising would stay silent on a steady-true condition and mask freeze state
        _set_value(bb, 0.9, trail="t", type="e")
        await _eval(bb)
        await asyncio.gather(*bb._dispatch_tasks)
        assert len(fired) == 1

        bb.freeze("fz")
        _set_value(bb, 0.9, trail="t", type="e")  # merge_strategy="new": fresh immortal pheromone
        await _eval(bb)
        await asyncio.gather(*bb._dispatch_tasks)
        assert len(fired) == 1  # dispatch silently skipped while frozen

        bb.unfreeze("fz")
        _set_value(bb, 0.9, trail="t", type="e")
        await _eval(bb)
        await asyncio.gather(*bb._dispatch_tasks)
        assert len(fired) == 2  # dispatch resumed


class TestSnapshotSourceAgent:
    """P1.5: snapshots must carry the emitting agent's identity so consumers
    (e.g. the LangChain worker) can label untrusted data by its source."""

    def test_sniff_propagates_source_agent(self):
        bb = LocalBlackboard()
        _emit(bb, trail="t", type="e", intensity=0.9, decay=ImmortalDecay(), source_agent="writer-1")
        _emit(bb, trail="t", type="e2", intensity=0.5, decay=ImmortalDecay())

        result = bb.sniff(SniffParams(trails=["t"]))
        by_type = {p.type: p for p in result.pheromones}
        assert by_type["e"].source_agent == "writer-1"
        assert by_type["e2"].source_agent is None

    @pytest.mark.asyncio
    async def test_trigger_context_propagates_source_agent(self):
        bb = LocalBlackboard()
        captured = []

        async def handler(payload):
            captured.append(payload)

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond,
            trigger_mode="level", cooldown_ms=0,
        ))
        bb.subscribe("s1", handler)

        _emit(bb, trail="t", type="e", intensity=0.9, decay=ImmortalDecay(), source_agent="writer-1")
        await bb.evaluate_scents()
        await asyncio.gather(*bb._dispatch_tasks)

        assert len(captured) == 1
        assert captured[0].context_pheromones[0].source_agent == "writer-1"


class TestStallSignals:
    """P1.6: handler failures must be observable — a structured JSON log line
    on stdout and a `stalled` signal on the reserved system.errors trail, not
    just a bare print nobody can subscribe to."""

    @pytest.mark.asyncio
    async def test_raising_handler_emits_stall_signal_and_json_log(self, capsys):
        bb = LocalBlackboard()

        async def handler(payload):
            raise ValueError("boom")

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        reg = bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond,
            trigger_mode="level", cooldown_ms=0, max_execution_ms=5000,
        ))
        bb.subscribe(reg.scent_id, handler)

        _set_value(bb, 0.9, trail="t", type="e")
        await _eval(bb)
        await asyncio.wait_for(asyncio.gather(*bb._dispatch_tasks), timeout=2.0)

        stalls = _stall_pheromones(bb)
        assert len(stalls) == 1
        stall = stalls[0]
        assert stall.type == "stalled"
        assert stall.payload["scent_id"] == reg.scent_id
        assert stall.payload["cause"] == "exception"
        assert "boom" in str(stall.payload["error"]) or "ValueError" in str(stall.payload["error"])

        records = _json_log_lines(capsys)
        assert len(records) == 1
        rec = records[0]
        assert {"time", "agent", "event", "detail"} <= set(rec)
        assert rec["event"] == "handler_error"
        assert "boom" in str(rec["detail"])

        # recovery, not stuck: the activation must fully unwind after the failure
        assert bb.scents[reg.scent_id]["running"] is False

    @pytest.mark.asyncio
    async def test_timing_out_handler_emits_stall_signal_and_json_log(self, capsys):
        bb = LocalBlackboard()

        async def handler(payload):
            await asyncio.Event().wait()  # never resolves

        cond = ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5)
        reg = bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a", condition=cond,
            trigger_mode="level", cooldown_ms=0, max_execution_ms=50,
        ))
        bb.subscribe(reg.scent_id, handler)

        _set_value(bb, 0.9, trail="t", type="e")
        await _eval(bb)
        # wait_for keeps a broken (never-resolving) dispatch from hanging pytest:
        # the working implementation must resolve it on its own at ~50ms.
        await asyncio.wait_for(asyncio.gather(*bb._dispatch_tasks), timeout=2.0)

        stalls = _stall_pheromones(bb)
        assert len(stalls) == 1
        stall = stalls[0]
        assert stall.type == "stalled"
        assert stall.payload["scent_id"] == reg.scent_id
        assert stall.payload["cause"] == "timeout"
        assert stall.payload["timeout_ms"] == 50

        records = _json_log_lines(capsys)
        assert len(records) == 1
        assert records[0]["event"] == "activation_timeout"
        assert "timed out" in str(records[0]["detail"])


class TestReservedTrailGuard:
    """P1.6: no scent may watch the reserved system trail, recursively through
    composites. This is also the no-cascade guarantee — a stall cascade on
    system.errors requires a registered scent watching that trail, which these
    tests prove is impossible, so no separate cascade test is needed."""

    def test_rejects_threshold_on_system_errors_trail(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="system.errors", signal_type="x", operator=">=", value=0.5)
        with pytest.raises(PermissionError, match="reserved trail"):
            bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        assert "s1" not in bb.scents

    def test_rejects_bare_system_trail(self):
        bb = LocalBlackboard()
        cond = ThresholdCondition(trail="system", signal_type="x", operator=">=", value=0.5)
        with pytest.raises(PermissionError, match="reserved trail"):
            bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        assert "s1" not in bb.scents

    def test_rejects_composite_containing_reserved_trail(self):
        bb = LocalBlackboard()
        cond = CompositeCondition(operator="and", conditions=[
            ThresholdCondition(trail="ok", signal_type="x", operator=">=", value=0.5),
            ThresholdCondition(trail="system.errors", signal_type="x", operator=">=", value=0.5),
        ])
        with pytest.raises(PermissionError, match="reserved trail"):
            bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        assert "s1" not in bb.scents

    def test_rejects_reserved_trail_nested_two_composites_deep(self):
        bb = LocalBlackboard()
        cond = CompositeCondition(operator="or", conditions=[
            ThresholdCondition(trail="ok", signal_type="x", operator=">=", value=0.5),
            CompositeCondition(operator="and", conditions=[
                ThresholdCondition(trail="also-ok", signal_type="x", operator=">=", value=0.5),
                ThresholdCondition(trail="system", signal_type="x", operator=">=", value=0.5),
            ]),
        ])
        with pytest.raises(PermissionError, match="reserved trail"):
            bb.register_scent(RegisterScentParams(scent_id="s1", agent_endpoint="test://a", condition=cond))
        assert "s1" not in bb.scents
