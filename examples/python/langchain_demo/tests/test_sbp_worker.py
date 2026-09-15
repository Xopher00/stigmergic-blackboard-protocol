"""
SbpWorker tests — zero real API calls, via a scripted fake chat model (conftest.py's
ScriptedChatModel). Regression coverage for every bug found in the deleted paper-triage
experiment: the trail-filtering all-or-nothing rejection, the missing action-surfacing
on inscribe, config params not actually reaching where they're supposed to, and the
active_activations counter not being exercised on the failure path.
"""
import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from sbp_worker import SbpWorker
from sbp.types import LinearDecay, ThresholdCondition, TriggerPayload


def _tool_call_message(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex}])


def _trigger(scent_id: str = "x") -> TriggerPayload:
    return TriggerPayload(
        scent_id=scent_id, triggered_at=0, condition_snapshot={}, context_pheromones=[], activation_payload={}
    )


async def _run_one_activation(worker: SbpWorker, model: FakeMessagesListChatModel):
    """Start the worker and drive exactly one _on_trigger call directly -- bypasses
    the shared blackboard's evaluation-loop timing/cooldown entirely, for a fast,
    deterministic test."""
    import asyncio

    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.2)
    try:
        await worker._on_trigger(_trigger())
    finally:
        worker.stop()
        await task


class TestTrailFiltering:
    @pytest.mark.asyncio
    async def test_sbp_read_serves_valid_trails_and_reports_skipped(self, make_model, capsys):
        model = make_model([
            _tool_call_message("sbp_read", {"trails": ["not-allowed", "claims"]}),
            AIMessage(content="done"),
        ])
        worker = SbpWorker(
            "w1", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1},
            sbp_ops=["read", "inscribe"],
            allowed_trails=["claims"],
        )
        await _run_one_activation(worker, model)

        out = capsys.readouterr().out
        assert "skipped disallowed trails ['not-allowed']" in out
        assert "no traces found" in out  # the valid 'claims' trail was actually served
        assert "not permitted" not in out  # never the old all-or-nothing rejection

    @pytest.mark.asyncio
    async def test_sbp_sniff_same_filtering_behavior(self, make_model, capsys):
        model = make_model([
            _tool_call_message("sbp_sniff", {"trails": ["bogus", "claims"]}),
            AIMessage(content="done"),
        ])
        worker = SbpWorker(
            "w2", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1},
            sbp_ops=["sniff"], allowed_trails=["claims"],
        )
        await _run_one_activation(worker, model)
        out = capsys.readouterr().out
        assert "skipped disallowed trails ['bogus']" in out
        assert "no signals found" in out


class TestInscribeActionSurfacing:
    @pytest.mark.asyncio
    async def test_created_vs_updated_reported(self, make_model, capsys):
        model = make_model([
            _tool_call_message("sbp_inscribe", {"trail": "claims", "key": "k1", "value": {"v": 1}}),
            AIMessage(content="done"),
        ])
        worker = SbpWorker(
            "w3", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1}, sbp_ops=["inscribe"],
        )
        await _run_one_activation(worker, model)
        out = capsys.readouterr().out
        assert "created trace v1 at claims/k1" in out


class TestEmitPassthrough:
    @pytest.mark.asyncio
    async def test_free_form_payload_tags_merge_strategy_reach_the_blackboard(self, make_model):
        model = make_model([
            _tool_call_message("sbp_emit", {
                "trail": "t", "type": "e", "intensity": 0.8,
                "payload": {"a": 1, "b": 2}, "tags": ["urgent"], "merge_strategy": "replace",
            }),
            AIMessage(content="done"),
        ])
        worker = SbpWorker(
            "w4", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1}, sbp_ops=["emit", "sniff"],
        )
        await _run_one_activation(worker, model)
        result = await worker.sbp_agent.sniff(trails=["t"])
        assert result.pheromones[0].payload == {"a": 1, "b": 2}
        assert result.pheromones[0].tags == ["urgent"]


class TestConstructorParamsActuallyWire:
    def test_default_decay_reaches_internal_sbp_agent(self, make_model):
        model = make_model([])
        worker = SbpWorker(
            "w5", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1},
            default_decay=LinearDecay(rate_per_ms=0.001),
        )
        assert worker.sbp_agent.default_decay == LinearDecay(rate_per_ms=0.001)

    def test_recursion_limit_reaches_invoke_config(self, make_model):
        model = make_model([])
        worker = SbpWorker(
            "w6", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1},
            recursion_limit=42,
        )
        assert worker._invoke_config["recursion_limit"] == 42

    def test_cooldown_ms_reaches_scent_registration(self, make_model):
        model = make_model([])
        worker = SbpWorker(
            "w7", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1, "cooldown_ms": 5000},
        )
        assert worker.sbp_agent._scents[0].cooldown_ms == 5000


class TestActiveActivationsCounter:
    @pytest.mark.asyncio
    async def test_returns_to_zero_after_normal_completion(self, make_model):
        model = make_model([AIMessage(content="nothing to do")])
        worker = SbpWorker(
            "w8", model, "sys", listens_for={"trail": "p", "signal_type": "a", "value": 0.1},
        )
        assert worker.active_activations == 0
        await _run_one_activation(worker, model)
        assert worker.active_activations == 0

    @pytest.mark.asyncio
    async def test_returns_to_zero_after_recursion_error(self, make_model, capsys):
        # Script more tool-call steps than recursion_limit allows, forcing GraphRecursionError.
        model = make_model([
            _tool_call_message("sbp_emit", {"trail": "t", "type": "e", "intensity": 0.5, "payload": {}}),
            _tool_call_message("sbp_emit", {"trail": "t", "type": "e", "intensity": 0.5, "payload": {}}),
            _tool_call_message("sbp_emit", {"trail": "t", "type": "e", "intensity": 0.5, "payload": {}}),
        ])
        worker = SbpWorker(
            "w9", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1},
            sbp_ops=["emit"], recursion_limit=2,
        )
        import asyncio
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.2)
        try:
            with pytest.raises(Exception):  # GraphRecursionError
                await worker._on_trigger(_trigger())
        finally:
            worker.stop()
            await task

        assert worker.active_activations == 0
        out = capsys.readouterr().out
        assert "tool_call sbp_emit" in out  # logging survived the failure, not lost


class TestSbpOpsGating:
    def test_requested_ops_present_omitted_ops_absent(self, make_model):
        model = make_model([])
        worker = SbpWorker(
            "w10", model, "sys",
            listens_for={"trail": "p", "signal_type": "a", "value": 0.1},
            sbp_ops=["emit", "inspect", "register_scent"],
        )
        names = {t.name for t in worker._build_sbp_tools(["emit", "inspect", "register_scent"])}
        assert names == {"sbp_emit", "sbp_inspect", "sbp_register_scent"}


class TestListensForBothForms:
    def test_dict_shortcut_routes_through_when(self, make_model):
        model = make_model([])
        worker = SbpWorker(
            "w11", model, "sys",
            listens_for={"trail": "t", "signal_type": "e", "value": 0.5},
        )
        assert isinstance(worker.sbp_agent._scents[0].condition, ThresholdCondition)

    def test_scent_condition_object_routes_through_on_scent(self, make_model):
        model = make_model([])
        cond = ThresholdCondition(trail="t", signal_type="e", value=0.5)
        worker = SbpWorker("w12", model, "sys", listens_for=cond)
        assert worker.sbp_agent._scents[0].condition is cond
        assert worker.sbp_agent._scents[0].scent_id == "w12:trigger"
