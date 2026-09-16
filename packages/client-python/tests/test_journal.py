"""
P3.1 operation-journal tests (J1-J14). When LocalBlackboard is constructed with
journal_path, every operation appends exactly one JSON line to an append-only
JSONL file; with the default (None) nothing is written.

Frozen line schema -- camelCase, ALL 10 keys present on every line, absent -> null:
  seq:int  ts:int(epoch ms)  agent:str|null  op:str  trail:str|null
  targetId:str|null  outcome:"ok"|"not_found"|"error"  latencyMs:number>=0
  activationId:str|null  skippedFires:int|null
"""
import asyncio
import itertools
import json
import re
from pathlib import Path

from sbp.blackboard import LocalBlackboard
from sbp.types import (
    EmitParams, SniffParams, RegisterScentParams,
    InscribeParams, ReadParams, EraseParams, EvaporateParams, InspectParams,
    ThresholdCondition, ImmortalDecay,
)

FROZEN_NOW = 1_700_000_000_000

JOURNAL_KEYS = {"seq", "ts", "agent", "op", "trail", "targetId",
                "outcome", "latencyMs", "activationId", "skippedFires"}
JOURNAL_OPS = {"emit", "sniff", "register_scent", "deregister_scent", "trigger",
               "evaporate", "inspect", "inscribe", "read", "erase"}
OUTCOMES = {"ok", "not_found", "error"}


def _set_value(bb, intensity, trail="t", type="e"):
    """Pin the aggregate a threshold condition sees (mirror of test_blackboard.py)."""
    bb.evaporate(EvaporateParams(trail=trail))
    bb.emit(EmitParams(trail=trail, type=type, intensity=intensity,
                       decay=ImmortalDecay(), merge_strategy="new"))


async def _eval(bb):
    await bb.evaluate_scents()
    await asyncio.sleep(0)  # dispatch is fire-and-forget; let the task run


def _journal_bb(tmp_path, name="journal.jsonl", **kw):
    p = tmp_path / name
    return LocalBlackboard(journal_path=str(p), **kw), p


def _assert_line_schema(rec):
    assert set(rec) == JOURNAL_KEYS, f"keys: {sorted(rec)}"
    assert isinstance(rec["seq"], int) and not isinstance(rec["seq"], bool)
    assert isinstance(rec["ts"], int) and not isinstance(rec["ts"], bool)
    assert rec["agent"] is None or isinstance(rec["agent"], str)
    assert rec["op"] in JOURNAL_OPS
    assert rec["trail"] is None or isinstance(rec["trail"], str)
    assert rec["targetId"] is None or isinstance(rec["targetId"], str)
    assert rec["outcome"] in OUTCOMES
    assert isinstance(rec["latencyMs"], (int, float))
    assert not isinstance(rec["latencyMs"], bool) and rec["latencyMs"] >= 0
    assert rec["activationId"] is None or isinstance(rec["activationId"], str)
    assert rec["skippedFires"] is None or (
        isinstance(rec["skippedFires"], int) and not isinstance(rec["skippedFires"], bool))


def _read_journal(path, validate=True):
    lines = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if validate:
        for rec in lines:
            _assert_line_schema(rec)
    return lines


def _ops(lines, op):
    return [l for l in lines if l["op"] == op]


class TestJournalOps:
    def test_j1_emit_created_line(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        r = bb.emit(EmitParams(trail="t", type="e", intensity=0.8, source_agent="agent-9"))
        (line,) = _ops(_read_journal(p), "emit")
        assert line["trail"] == "t"
        assert line["targetId"] == r.pheromone_id
        assert line["agent"] == "agent-9"
        assert line["outcome"] == "ok"
        assert line["latencyMs"] >= 0

    def test_j2_emit_reinforce_targets_existing_pheromone(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        r1 = bb.emit(EmitParams(trail="t", type="e", intensity=0.8))
        r2 = bb.emit(EmitParams(trail="t", type="e", intensity=0.4))
        assert r2.pheromone_id == r1.pheromone_id  # reinforce merges into the existing one
        emits = _ops(_read_journal(p), "emit")
        assert [l["targetId"] for l in emits] == [r1.pheromone_id, r1.pheromone_id]

    def test_j3_sniff_line(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.sniff(SniffParams(trails=["a", "b"]))
        (line,) = _ops(_read_journal(p), "sniff")
        assert line["trail"] == "a,b"
        assert line["targetId"] is None
        assert line["outcome"] == "ok"

    def test_j4_register_and_deregister_lines(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        reg = bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a",
            condition=ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5),
        ), agent_id="agent-42")
        assert reg.scent_id == "agent-42:s1"  # P1.3 auto-prefix; the journal logs the resolved id
        bb.deregister_scent(reg.scent_id, agent_id="agent-42")
        bb.deregister_scent(reg.scent_id, agent_id="agent-42")  # already gone -> not_found
        lines = _read_journal(p)
        (reg_line,) = _ops(lines, "register_scent")
        assert reg_line["targetId"] == "agent-42:s1"
        assert reg_line["agent"] == "agent-42"
        assert reg_line["outcome"] == "ok"
        der_reg, der_ghost = _ops(lines, "deregister_scent")
        assert der_reg["targetId"] == "agent-42:s1"
        assert der_reg["outcome"] == "ok"
        assert der_ghost["outcome"] == "not_found"

    def test_j7_evaporate_line(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.emit(EmitParams(trail="t", type="e", intensity=0.8))
        bb.evaporate(EvaporateParams(trail="t"))
        (line,) = _ops(_read_journal(p), "evaporate")
        assert line["trail"] == "t"
        assert line["targetId"] is None
        assert line["outcome"] == "ok"

    def test_j8_inspect_line(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.inspect(InspectParams())
        (line,) = _ops(_read_journal(p), "inspect")
        assert line["trail"] is None
        assert line["targetId"] is None
        assert line["outcome"] == "ok"

    def test_j9_inscribe_lines_not_deduped(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}, source_agent="agent-7"))
        bb.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 2}, source_agent="agent-7"))
        ins = _ops(_read_journal(p), "inscribe")
        assert len(ins) == 2  # a rewrite to the same key must still be journaled
        assert all(l["trail"] == "notes" and l["targetId"] == "k1"
                   and l["agent"] == "agent-7" and l["outcome"] == "ok" for l in ins)

    def test_j10_read_hit_and_miss(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}))
        hit = bb.read(ReadParams(trails=["notes"], keys=["k1"]))
        assert len(hit.traces) == 1
        miss = bb.read(ReadParams(trails=["notes"], keys=["nope"]))
        assert len(miss.traces) == 0
        reads = _ops(_read_journal(p), "read")
        assert len(reads) == 2  # the miss is journaled too
        assert reads[0]["trail"] == "notes"
        assert reads[0]["targetId"] == "k1"
        assert reads[0]["outcome"] == "ok"
        assert reads[1]["outcome"] == "not_found"


class TestJournalTrigger:
    async def test_j5_clean_fire_single_line(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a",
            condition=ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5),
            trigger_mode="level", cooldown_ms=0,
        ))
        fired = []

        async def handler(payload):
            fired.append(payload)

        bb.subscribe("s1", handler)
        _set_value(bb, 0.9)
        await _eval(bb)
        await asyncio.wait_for(asyncio.gather(*bb._dispatch_tasks), timeout=2.0)
        assert len(fired) == 1
        (line,) = _ops(_read_journal(p), "trigger")
        assert line["targetId"] == "s1"
        assert line["outcome"] == "ok"
        assert line["skippedFires"] == 0
        assert line["latencyMs"] >= 0
        assert re.fullmatch(r"s1@\d+", line["activationId"])  # "<scent_id>@<triggered_at>"

    async def test_j6_skipped_fire_journaled(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a",
            condition=ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5),
            trigger_mode="level", cooldown_ms=0, max_execution_ms=100,
        ))

        async def stuck_handler(payload):
            await asyncio.Event().wait()  # never resolves; the timeout cancels it

        bb.subscribe("s1", stuck_handler)
        _set_value(bb, 0.9)
        await _eval(bb)  # fires; handler enters and blocks
        await _eval(bb)  # condition still true, activation still running -> one skip
        await asyncio.wait_for(asyncio.gather(*bb._dispatch_tasks), timeout=2.0)
        triggers = _ops(_read_journal(p), "trigger")
        assert triggers
        assert any(l["skippedFires"] == 1 for l in triggers)
        assert all(l["skippedFires"] in (0, 1) for l in triggers)
        assert all(l["targetId"] == "s1" and l["outcome"] == "ok" for l in triggers)


class TestJournalFileBehavior:
    def test_j11_erase_tombstone_append_only(self, tmp_path):
        bb, p = _journal_bb(tmp_path)
        bb.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}))
        bb.inscribe(InscribeParams(trail="notes", key="k2", value={"v": 2}))
        bb.erase(EraseParams(trail="notes", keys=["k1"]))
        lines = _read_journal(p)
        k1_at = [i for i, l in enumerate(lines) if l["op"] == "inscribe" and l["targetId"] == "k1"]
        erase_at = [i for i, l in enumerate(lines) if l["op"] == "erase"]
        assert k1_at and erase_at and k1_at[0] < erase_at[0]
        assert lines[erase_at[0]]["targetId"] == "k1"
        assert lines[erase_at[0]]["outcome"] == "ok"
        before_read = len(lines)
        ghost = bb.read(ReadParams(trails=["notes"], keys=["k1"]))
        assert len(ghost.traces) == 0
        final = _read_journal(p)
        assert len(final) > before_read  # the not_found read is journaled
        # append-only: the erase line is the tombstone, earlier lines are never rewritten
        assert final[:before_read] == lines
        assert _ops(final, "read")[-1]["outcome"] == "not_found"

    def test_j12_path_honored_and_default_disabled(self, tmp_path, monkeypatch):
        bb, p = _journal_bb(tmp_path, "str-path.jsonl")
        bb.emit(EmitParams(trail="t", type="e", intensity=0.8))
        assert p.exists() and len(_read_journal(p)) >= 1

        p2 = tmp_path / "pathlike.jsonl"
        bb2 = LocalBlackboard(journal_path=p2)  # os.PathLike variant
        bb2.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}))
        assert p2.exists() and len(_read_journal(p2)) == 1

        monkeypatch.chdir(tmp_path)
        bb3 = LocalBlackboard()  # default: no path -> no journal file anywhere
        bb3.emit(EmitParams(trail="t", type="e", intensity=0.8))
        bb3.sniff(SniffParams(trails=["t"]))
        bb3.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}))
        assert list(tmp_path.rglob("*.jsonl")) == [p, p2]

    def test_j13_injectable_clock(self, tmp_path):
        bb, p = _journal_bb(tmp_path, "frozen.jsonl")
        bb._now = lambda: FROZEN_NOW
        bb.emit(EmitParams(trail="t", type="e", intensity=0.8))
        bb.sniff(SniffParams(trails=["t"]))
        bb.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}))
        lines = _read_journal(p)
        assert len(lines) == 3
        assert all(l["ts"] == FROZEN_NOW for l in lines)
        assert all(l["latencyMs"] == 0 for l in lines)

        bb2, p2 = _journal_bb(tmp_path, "counter.jsonl")
        counter = itertools.count()
        bb2._now = lambda: next(counter)  # every _now() call advances one tick
        bb2.emit(EmitParams(trail="t", type="e", intensity=0.8))
        bb2.sniff(SniffParams(trails=["t"]))
        bb2.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}))
        bb2.read(ReadParams(trails=["notes"], keys=["k1"]))
        lines2 = _read_journal(p2)
        ts = [l["ts"] for l in lines2]
        assert ts == sorted(ts)  # non-decreasing across lines
        assert any(l["latencyMs"] > 0 for l in lines2)


class TestJournalSchemaSweep:
    async def test_j14_every_op_conforms(self, tmp_path):
        bb, p = _journal_bb(tmp_path, "tour.jsonl")
        bb.emit(EmitParams(trail="t", type="e", intensity=0.8, source_agent="agent-1"))
        bb.sniff(SniffParams(trails=["t"]))
        reg = bb.register_scent(RegisterScentParams(
            scent_id="s1", agent_endpoint="test://a",
            condition=ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5),
        ), agent_id="agent-42")
        bb.deregister_scent(reg.scent_id, agent_id="agent-42")

        bb.register_scent(RegisterScentParams(
            scent_id="s2", agent_endpoint="test://b",
            condition=ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5),
            trigger_mode="level", cooldown_ms=0,
        ))

        async def handler(payload):
            pass

        bb.subscribe("s2", handler)
        _set_value(bb, 0.9)
        await _eval(bb)
        await asyncio.wait_for(asyncio.gather(*bb._dispatch_tasks), timeout=2.0)

        bb.evaporate(EvaporateParams(trail="t"))
        bb.inspect(InspectParams())
        bb.inscribe(InscribeParams(trail="notes", key="k1", value={"v": 1}, source_agent="agent-7"))
        bb.read(ReadParams(trails=["notes"], keys=["k1"]))   # hit
        bb.read(ReadParams(trails=["notes"], keys=["nope"]))  # miss
        bb.read(ReadParams(trails=["notes"], prefix="k"))    # prefix query
        bb.erase(EraseParams(trail="notes", keys=["k1"]))

        # _read_journal validates the full frozen schema on every single line
        lines = _read_journal(p)
        by_op = {op: _ops(lines, op) for op in JOURNAL_OPS}
        assert all(by_op[op] for op in JOURNAL_OPS)  # all ten ops journaled at least once

        assert by_op["emit"][0]["trail"] == "t"
        assert by_op["emit"][0]["agent"] == "agent-1"
        assert by_op["sniff"][0]["trail"] == "t"
        assert by_op["register_scent"][0]["targetId"] == "agent-42:s1"
        assert by_op["deregister_scent"][0]["outcome"] == "ok"
        (trigger,) = by_op["trigger"]
        assert trigger["targetId"] == "s2"
        assert trigger["skippedFires"] == 0
        assert re.fullmatch(r"s2@\d+", trigger["activationId"])
        evap = by_op["evaporate"][-1]
        assert evap["trail"] == "t" and evap["targetId"] is None
        (insp,) = by_op["inspect"]
        assert insp["trail"] is None and insp["targetId"] is None
        ins = by_op["inscribe"][-1]
        assert ins["trail"] == "notes" and ins["targetId"] == "k1" and ins["agent"] == "agent-7"
        reads = by_op["read"]
        assert reads[0]["outcome"] == "ok" and reads[0]["targetId"] == "k1"
        assert reads[1]["outcome"] == "not_found"
        assert reads[2]["targetId"] == "k*"  # prefix query -> "prefix*"
        (er,) = by_op["erase"]
        assert er["trail"] == "notes" and er["targetId"] == "k1" and er["outcome"] == "ok"
