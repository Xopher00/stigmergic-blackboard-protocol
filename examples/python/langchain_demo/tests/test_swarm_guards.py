"""TEST_IDEAS.md guards #1, #2, #4, #5, #6 -- offline, no LLM, direct against
LocalBlackboard using the exact emit shapes/decay profiles roles.py's tools use, and
the same end-of-run/halt scent conditions roles.py registers. Guard #3 (exploration
at max heat) lives in test_swarm_scoring.py; guard #7 (dynamic scent) in
test_swarm_dynamic_scent.py.
"""
import pytest

from sbp.blackboard import LocalBlackboard
from sbp.conditions import count_gte
from sbp.evaluator import EvaluationContext, evaluate_condition
from sbp.types import EmitParams, RegisterScentParams

from swarm import board, roles
from swarm.board import DecayProfile


@pytest.fixture
def bb():
    return LocalBlackboard()


@pytest.fixture
def profile():
    return DecayProfile.scripted()


def _emit_evidence(bb, path, kind, profile, now=0):
    bb._now = lambda: now
    return bb.emit(EmitParams(
        trail=board.TRAIL_EVIDENCE, type=kind, intensity=0.8, decay=profile.evidence,
        payload={"file": path, "kind": kind}, merge_strategy="reinforce", source_agent="scout-1",
    ))


def _evidence_count(bb, kind, now):
    condition = count_gte(board.TRAIL_EVIDENCE, kind, 2)
    ctx = EvaluationContext(list(bb.pheromones.values()), now=now)
    return evaluate_condition(condition, ctx)


class TestGuard1NoFalseFires:
    def test_one_scouts_alarm_does_not_confirm(self, bb, profile):
        _emit_evidence(bb, "a/X.java", "hardcoded_secret", profile)
        assert _evidence_count(bb, "hardcoded_secret", now=0).met is False

    def test_same_behavior_in_two_different_files_confirms(self, bb, profile):
        _emit_evidence(bb, "a/X.java", "hardcoded_secret", profile)
        _emit_evidence(bb, "a/Y.java", "hardcoded_secret", profile)
        assert _evidence_count(bb, "hardcoded_secret", now=0).met is True

    def test_two_scouts_alarmed_by_the_same_file_and_kind_is_one_piece_of_evidence(self, bb, profile):
        # Same trail/type/payload merges regardless of source_agent: independence
        # means different files, not different reporters.
        bb.emit(EmitParams(trail=board.TRAIL_EVIDENCE, type="hardcoded_secret", intensity=0.8,
                            decay=profile.evidence, payload={"file": "a/X.java", "kind": "hardcoded_secret"},
                            merge_strategy="reinforce", source_agent="scout-1"))
        bb.emit(EmitParams(trail=board.TRAIL_EVIDENCE, type="hardcoded_secret", intensity=0.8,
                            decay=profile.evidence, payload={"file": "a/X.java", "kind": "hardcoded_secret"},
                            merge_strategy="reinforce", source_agent="scout-2"))
        assert _evidence_count(bb, "hardcoded_secret", now=0).met is False


class TestGuard2NoRunawayFeedback:
    def test_reannouncing_hot_without_new_evidence_does_not_raise_intensity(self, bb, profile):
        first = bb.emit(EmitParams(trail=board.TRAIL_HOT, type="a/auth", intensity=0.6,
                                    decay=profile.hot, payload={"kind": "hardcoded_secret"},
                                    merge_strategy="reinforce", source_agent="bloodhound"))
        second = bb.emit(EmitParams(trail=board.TRAIL_HOT, type="a/auth", intensity=0.6,
                                     decay=profile.hot, payload={"kind": "hardcoded_secret"},
                                     merge_strategy="reinforce", source_agent="bloodhound"))
        assert second.new_intensity == pytest.approx(first.new_intensity)


class TestGuard4Claims:
    def test_silent_scouts_claim_expires_and_file_returns_to_the_pool(self, bb, profile):
        bb._now = lambda: 0
        bb.emit(EmitParams(trail=board.TRAIL_CLAIMS, type=board.TYPE_CLAIM, intensity=1.0,
                            decay=profile.claim, payload={"file": "a/X.java"},
                            merge_strategy="reinforce", source_agent="scout-1"))
        elapsed = int(profile.claim.half_life_ms * 6)
        bb._now = lambda: elapsed
        snap = board.read_board_snapshot(bb, profile)
        assert "a/X.java" not in snap.claimed

    def test_actively_working_scouts_claim_survives_via_heartbeat_reinforcement(self, bb, profile):
        bb._now = lambda: 0
        bb.emit(EmitParams(trail=board.TRAIL_CLAIMS, type=board.TYPE_CLAIM, intensity=1.0,
                            decay=profile.claim, payload={"file": "a/X.java"},
                            merge_strategy="reinforce", source_agent="scout-1"))
        half = profile.claim.half_life_ms
        for t in range(0, half * 6, max(1, half // 3)):
            bb._now = lambda t=t: t
            bb.emit(EmitParams(trail=board.TRAIL_CLAIMS, type=board.TYPE_CLAIM, intensity=1.0,
                                decay=profile.claim, payload={"file": "a/X.java"},
                                merge_strategy="reinforce", source_agent="scout-1"))
        snap = board.read_board_snapshot(bb, profile)
        assert "a/X.java" in snap.claimed

    def test_end_condition_is_false_while_a_scout_is_mid_read(self, bb, profile):
        bb._now = lambda: 0
        bb.emit(EmitParams(trail=board.TRAIL_CLAIMS, type=board.TYPE_CLAIM, intensity=1.0,
                            decay=profile.claim, payload={"file": "a/X.java"},
                            merge_strategy="reinforce", source_agent="scout-1"))
        for f in ("a/X.java", "a/Y.java"):
            bb.emit(EmitParams(trail=board.TRAIL_COVERAGE, type=board.TYPE_FILE, intensity=1.0,
                                decay=profile.coverage, payload={"file": f},
                                merge_strategy="reinforce", source_agent="scout-1"))
        condition = roles.end_of_run_condition(total_files=2, profile=profile)
        ctx = EvaluationContext(list(bb.pheromones.values()), now=0, traces=list(bb.traces.values()))
        assert evaluate_condition(condition, ctx).met is False


class TestGuard5Budget:
    def test_budget_add_trips_exactly_once_at_the_cap(self):
        from swarm.budget import Budget
        budget = Budget(limit_tokens=100)
        assert budget.add(60) is False
        assert budget.add(50) is True
        assert budget.add(1) is False
        assert budget.tripped is True

    def test_halt_signal_ends_the_run_regardless_of_swarm_state(self, bb, profile):
        bb._now = lambda: 0
        bb.emit(EmitParams(trail=board.TRAIL_CONTROL, type=board.TYPE_HALT, intensity=1.0,
                            decay=profile.control, payload={"reason": "budget"},
                            merge_strategy="reinforce", source_agent="budget-supervisor"))
        condition = roles.end_of_run_condition(total_files=999, profile=profile)
        ctx = EvaluationContext(list(bb.pheromones.values()), now=0, traces=list(bb.traces.values()))
        assert evaluate_condition(condition, ctx).met is True


class TestGuard6Freeze:
    def test_freezing_one_agent_blocks_only_its_own_scents(self, bb, profile):
        bb.register_scent(RegisterScentParams(
            scent_id="watch", agent_endpoint="sse://scout-1",
            condition=count_gte(board.TRAIL_EVIDENCE, "x", 1),
        ), agent_id="scout-1")
        bb.freeze("scout-1:")
        with pytest.raises(PermissionError):
            bb.register_scent(RegisterScentParams(
                scent_id="watch2", agent_endpoint="sse://scout-1",
                condition=count_gte(board.TRAIL_EVIDENCE, "y", 1),
            ), agent_id="scout-1")
        result = bb.register_scent(RegisterScentParams(
            scent_id="watch", agent_endpoint="sse://scout-2",
            condition=count_gte(board.TRAIL_EVIDENCE, "z", 1),
        ), agent_id="scout-2")
        assert result.status == "registered"

    @pytest.mark.asyncio
    async def test_frozen_scent_does_not_fire_others_still_do(self, bb, profile):
        import asyncio

        fired = []

        async def handler(trigger):
            fired.append(trigger.scent_id)

        bb.register_scent(RegisterScentParams(
            scent_id="watch", agent_endpoint="sse://scout-1",
            condition=count_gte(board.TRAIL_EVIDENCE, "x", 1), cooldown_ms=0,
        ), agent_id="scout-1")
        bb.subscribe("scout-1:watch", handler, agent_id="scout-1")
        bb.register_scent(RegisterScentParams(
            scent_id="watch", agent_endpoint="sse://scout-2",
            condition=count_gte(board.TRAIL_EVIDENCE, "x", 1), cooldown_ms=0,
        ), agent_id="scout-2")
        bb.subscribe("scout-2:watch", handler, agent_id="scout-2")

        bb.freeze("scout-1:")
        bb.emit(EmitParams(trail=board.TRAIL_EVIDENCE, type="x", intensity=0.8,
                            payload={"file": "a"}, source_agent="scout-1"))
        await bb.evaluate_scents()
        await asyncio.sleep(0.05)

        assert "scout-1:watch" not in fired
        assert "scout-2:watch" in fired
