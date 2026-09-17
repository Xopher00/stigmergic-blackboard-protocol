"""TEST_IDEAS.md guard #7: a scout registers its own scent on a bloodhound question,
gets woken by it through the real reactive loop, deregisters it, and stop() leaves
nothing behind -- exercising SbpWorker's register_scent/deregister_scent tools and
the SDK's ownership/cleanup guarantees for real, not re-deriving them.
"""
import asyncio
import random
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

import sbp.blackboard as blackboard_module
from sbp.blackboard import LocalBlackboard, get_shared_blackboard
from sbp.client import AsyncSbpClient

from swarm import board, roles
from swarm.board import DecayProfile
from swarm.corpus import load_corpus
from swarm.scripted import ScriptedChatModel, write_scripted_corpus


def _call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex}])


@pytest.mark.asyncio
async def test_scout_self_registers_a_scent_gets_woken_by_it_and_cleans_up_on_stop(tmp_path):
    blackboard_module._shared_blackboard = LocalBlackboard()
    bb = get_shared_blackboard()

    root = tmp_path / "corpus"
    write_scripted_corpus(root)
    corpus = load_corpus(root)
    profile = DecayProfile.scripted()

    model = ScriptedChatModel(responses=[
        _call("claim_file", path="app/auth/LoginActivity.java"),
        _call("mark_visited"),
        _call("mark_covered"),
        _call("sbp_register_scent", scent_id="watch-topic", trail=board.TRAIL_QUESTIONS,
              signal_type="mytopic", value=0.5, cooldown_ms=0),
        AIMessage(content="watching"),
        _call("sbp_deregister_scent", scent_id="watch-topic"),
        AIMessage(content="done"),
    ])
    scout = roles.build_scout("scout-1", model, corpus, profile, random.Random(0))

    await scout.sbp_agent.start()
    task = asyncio.create_task(scout.run())
    bootstrap = AsyncSbpClient(local=True, agent_id="bootstrap")
    await bootstrap.connect()

    try:
        await bootstrap.emit(board.wake_trail("scout-1"), board.TYPE_WAKE, intensity=1.0,
                              decay=profile.wake, payload={})
        await asyncio.sleep(0.3)
        assert "scout-1:watch-topic" in bb.scents

        await bootstrap.emit(board.TRAIL_QUESTIONS, "mytopic", intensity=0.8,
                              decay=profile.question, payload={"topic": "mytopic"})
        await asyncio.sleep(0.3)
        assert "scout-1:watch-topic" not in bb.scents
        assert "scout-1:watch-topic" not in bb.handlers
    finally:
        await bootstrap.close()
        scout.stop()
        await task

    assert "scout-1:watch-topic" not in bb.scents
    assert "scout-1:watch-topic" not in bb.handlers
