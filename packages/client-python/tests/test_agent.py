"""
SbpAgent tests, local=True (in-memory, no network, no LangChain, no LLM). Regression
coverage for every bug found and fixed this session: emit()/sniff() dropping their
real return values, register_scent/deregister_scent/subscribe/unsubscribe only being
reachable via decorators staged before run() rather than callable while already
running, evaporate()/inspect() being entirely missing, when() not carrying
tags/activation_payload/context_trails through.
"""
import asyncio

import pytest

from sbp.agent import SbpAgent
from sbp.blackboard import get_shared_blackboard
from sbp.client import AsyncSbpClient
from sbp.types import EmitResult, SniffResult, ThresholdCondition, CompositeCondition
from sbp.types import ImmortalDecay


class TestRaisesBeforeRunning:
    @pytest.mark.asyncio
    async def test_every_op_raises_before_run(self):
        agent = SbpAgent("a", local=True)
        with pytest.raises(RuntimeError):
            await agent.emit("t", "e", 0.5)
        with pytest.raises(RuntimeError):
            await agent.sniff()
        with pytest.raises(RuntimeError):
            await agent.inscribe("t", "k", {})
        with pytest.raises(RuntimeError):
            await agent.read()
        with pytest.raises(RuntimeError):
            await agent.erase()
        with pytest.raises(RuntimeError):
            await agent.register_scent("s", ThresholdCondition(trail="t", signal_type="e", value=0.5))
        with pytest.raises(RuntimeError):
            await agent.deregister_scent("s")
        with pytest.raises(RuntimeError):
            await agent.evaporate()
        with pytest.raises(RuntimeError):
            await agent.inspect()
        with pytest.raises(RuntimeError):
            await agent.subscribe("s", lambda p: None)
        with pytest.raises(RuntimeError):
            await agent.unsubscribe("s")


class TestWhenAndOnScentRegistration:
    def test_when_builds_threshold_condition_with_tags_and_extras(self):
        agent = SbpAgent("a", local=True)

        @agent.when(
            "t", "e", value=0.7, cooldown_ms=1000,
            activation_payload={"k": "v"}, context_trails=["t2"],
        )
        async def handler(trigger):
            pass

        assert len(agent._scents) == 1
        reg = agent._scents[0]
        assert isinstance(reg.condition, ThresholdCondition)
        assert reg.condition.value == 0.7
        assert reg.cooldown_ms == 1000
        assert reg.activation_payload == {"k": "v"}
        assert reg.context_trails == ["t2"]

    def test_on_scent_accepts_composite_condition(self):
        agent = SbpAgent("a", local=True)
        cond = CompositeCondition(operator="and", conditions=[
            ThresholdCondition(trail="t1", signal_type="e", value=0.5),
            ThresholdCondition(trail="t2", signal_type="e", value=0.5),
        ])

        @agent.on_scent("s1", cond)
        async def handler(trigger):
            pass

        assert agent._scents[0].condition is cond


@pytest.fixture
async def running_agent():
    agent = SbpAgent("runner", local=True)
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.2)
    yield agent
    agent.stop()
    await task


class TestLiveAgentOperations:
    @pytest.mark.asyncio
    async def test_emit_returns_real_result_not_none(self, running_agent):
        result = await running_agent.emit("t", "e", 0.8)
        assert isinstance(result, EmitResult)
        assert result.action == "created"

    @pytest.mark.asyncio
    async def test_sniff_accepts_limit_and_include_evaporated_returns_real_result(self, running_agent):
        await running_agent.emit("t", "e", 0.005)  # below default ttl_floor
        excluded = await running_agent.sniff(trails=["t"])
        included = await running_agent.sniff(trails=["t"], limit=5, include_evaporated=True)
        assert isinstance(included, SniffResult)
        assert excluded.pheromones == []
        assert len(included.pheromones) == 1

    @pytest.mark.asyncio
    async def test_register_and_deregister_scent_callable_while_running(self, running_agent):
        fired = []

        async def handler(trigger):
            fired.append(trigger)

        result = await running_agent.register_scent(
            "dynamic", ThresholdCondition(trail="t", signal_type="e", value=0.5), cooldown_ms=10_000
        )
        assert result.status == "registered"
        await running_agent.subscribe("dynamic", handler)

        await running_agent.emit("t", "e", 0.9)
        await asyncio.sleep(0.3)  # let the shared blackboard's evaluation loop dispatch
        assert len(fired) == 1

        await running_agent.unsubscribe("dynamic")
        dereg = await running_agent.deregister_scent("dynamic")
        assert dereg.status == "deregistered"

    @pytest.mark.asyncio
    async def test_evaporate_removes_matching_pheromones(self, running_agent):
        await running_agent.emit("t", "e", 0.9)
        result = await running_agent.evaporate(trail="t")
        assert result.evaporated_count == 1
        sniffed = await running_agent.sniff(trails=["t"])
        assert sniffed.pheromones == []

    @pytest.mark.asyncio
    async def test_inspect_reports_real_stats(self, running_agent):
        await running_agent.emit("t", "e", 0.9)
        result = await running_agent.inspect()
        assert result.stats["total_pheromones"] == 1

    @pytest.mark.asyncio
    async def test_inscribe_read_erase_round_trip(self, running_agent):
        inscribed = await running_agent.inscribe("claims", "k1", {"text": "finding"})
        assert inscribed.action == "created"
        read_back = await running_agent.read(trails=["claims"])
        assert read_back.traces[0].value == {"text": "finding"}
        erased = await running_agent.erase(trail="claims", keys=["k1"])
        assert erased.erased_count == 1


class TestRunLifecycle:
    @pytest.mark.asyncio
    async def test_staged_when_scent_fires_after_run(self):
        fired = []

        agent = SbpAgent("staged", local=True)

        @agent.when("t", "e", value=0.5, cooldown_ms=10_000)
        async def handler(trigger):
            fired.append(trigger)

        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.2)

        await agent.emit("t", "e", 0.9)
        await asyncio.sleep(0.3)

        agent.stop()
        await task

        assert len(fired) == 1
        assert fired[0].scent_id == "staged:t/e"


class TestStopCleansDynamicScents:
    @pytest.mark.asyncio
    async def test_stop_deregisters_scent_registered_while_running(self):
        bb = get_shared_blackboard()
        agent = SbpAgent("dyn", local=True)
        fired = []

        async def handler(trigger):
            fired.append(trigger)

        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.2)

        # exactly what sbp_register_scent does mid-run: register + subscribe
        await agent.register_scent(
            "dynamic", ThresholdCondition(trail="t", signal_type="e", value=0.5), cooldown_ms=0
        )
        await agent.subscribe("dynamic", handler)

        agent.stop()
        await asyncio.wait_for(task, timeout=5)

        assert "dyn:dynamic" not in bb.scents
        assert "dyn:dynamic" not in bb.handlers

        # The shared loop must be alive here or "never fires" passes vacuously even on
        # broken code -- a stopped evaluation loop silences zombie handlers too.
        observer = AsyncSbpClient("http://localhost:3000", agent_id="observer", local=True)
        await observer.connect()
        try:
            await observer.emit("t", "e", 0.9)
            await asyncio.sleep(0.5)
        finally:
            await observer.close()

        assert fired == []


class TestFreezeEndToEnd:
    async def _poll(self, get_count, target, timeout=2.0):
        # the shared blackboard's loop ticks every 100ms; poll instead of a fixed
        # sleep so a slow first dispatch never makes the test flaky
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while get_count() < target and loop.time() < deadline:
            await asyncio.sleep(0.05)
        return get_count()

    @pytest.mark.asyncio
    async def test_frozen_agent_stops_dispatching_while_running(self):
        bb = get_shared_blackboard()
        fired_fz = []
        fired_ctrl = []

        async def fz_handler(trigger):
            fired_fz.append(trigger)

        async def ctrl_handler(trigger):
            fired_ctrl.append(trigger)

        agent_fz = SbpAgent("fz", local=True)
        agent_ctrl = SbpAgent("ctrl", local=True)
        task_fz = asyncio.create_task(agent_fz.run())
        task_ctrl = asyncio.create_task(agent_ctrl.run())

        try:
            await asyncio.sleep(0.2)

            # P1.2 zombie-test style: register + subscribe dynamically while running,
            # each on its OWN trail; the same bare scent_id must resolve per-agent, not collide
            await agent_fz.register_scent(
                "watch", ThresholdCondition(trail="t-fz", signal_type="e", value=0.5),
                trigger_mode="level", cooldown_ms=0,
            )
            await agent_fz.subscribe("watch", fz_handler)
            await agent_ctrl.register_scent(
                "watch", ThresholdCondition(trail="t-ctrl", signal_type="e", value=0.5),
                trigger_mode="level", cooldown_ms=0,
            )
            await agent_ctrl.subscribe("watch", ctrl_handler)

            await agent_fz.emit("t-fz", "e", 0.9, decay=ImmortalDecay())
            await agent_ctrl.emit("t-ctrl", "e", 0.9, decay=ImmortalDecay())
            # both must be armed and firing, or "frozen silence" would pass vacuously
            assert await self._poll(lambda: len(fired_fz), 1) >= 1
            assert await self._poll(lambda: len(fired_ctrl), 1) >= 1

            # level+0ms re-fires every tick while the condition holds, so drain both
            # trails first and snapshot right after the sync freeze() -- no tick slips between
            await agent_fz.evaporate(trail="t-fz")
            await agent_ctrl.evaporate(trail="t-ctrl")
            await asyncio.sleep(0.15)

            bb.freeze("fz")
            fz_before = len(fired_fz)
            ctrl_before = len(fired_ctrl)

            await agent_fz.emit("t-fz", "e", 0.9, decay=ImmortalDecay(), merge_strategy="new")
            await agent_ctrl.emit("t-ctrl", "e", 0.9, decay=ImmortalDecay(), merge_strategy="new")
            await asyncio.sleep(0.3)

            assert len(fired_fz) == fz_before
            assert len(fired_ctrl) > ctrl_before  # canary: shared loop genuinely alive

            bb.unfreeze("fz")
            await agent_fz.emit("t-fz", "e", 0.9, decay=ImmortalDecay(), merge_strategy="new")
            await self._poll(lambda: len(fired_fz), fz_before + 1)
        finally:
            agent_fz.stop()
            agent_ctrl.stop()
            await asyncio.wait_for(task_fz, timeout=5)
            await asyncio.wait_for(task_ctrl, timeout=5)

        assert len(fired_fz) > fz_before
