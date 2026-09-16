"""Offline dashboard over an SBP operation journal (P3.1 JSONL schema)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_KEYS = (
    "seq", "ts", "agent", "op", "trail", "targetId", "outcome",
    "latencyMs", "activationId", "skippedFires",
)
READER_OPS = {"sniff", "read", "trigger"}
WRITER_OPS = {"emit", "inscribe", "erase", "register_scent", "deregister_scent", "evaporate"}
OP_ALIASES = {"registerScent": "register_scent", "deregisterScent": "deregister_scent"}
UNATTRIBUTED = "(unattributed)"


def _normalize_op(op: str) -> str:
    return OP_ALIASES.get(op, op)


def _valid_line(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    for key in FROZEN_KEYS:
        if key not in obj:
            return False
    if not isinstance(obj["op"], str) or not isinstance(obj["outcome"], str):
        return False
    if not isinstance(obj["ts"], int) or isinstance(obj["ts"], bool):
        return False
    if not isinstance(obj["latencyMs"], (int, float)) or isinstance(obj["latencyMs"], bool):
        return False
    if obj["latencyMs"] < 0:
        return False
    if obj["skippedFires"] is not None and not isinstance(obj["skippedFires"], int):
        return False
    for key in ("agent", "trail", "targetId", "activationId"):
        if obj[key] is not None and not isinstance(obj[key], str):
            return False
    return True


def _parse_journal(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = None
            if obj is None or not _valid_line(obj):
                print(f"warning: {path}:{lineno}: skipping malformed line", file=sys.stderr)
                skipped += 1
                continue
            obj["op"] = _normalize_op(obj["op"])
            records.append(obj)
    return records, skipped


def _agent_of(rec: dict[str, Any]) -> str:
    agent = rec["agent"]
    return agent if agent is not None else UNATTRIBUTED


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _fmt_row(rec: dict[str, Any]) -> str:
    trail = rec["trail"] if rec["trail"] is not None else "-"
    target = rec["targetId"] if rec["targetId"] is not None else "-"
    row = (
        f"  {_fmt_ts(rec['ts'])}  {rec['op']:<18}trail={trail:<15}target={target:<20} "
        f"{rec['outcome']:<10}{rec['latencyMs']:g}ms"
    )
    if rec["op"] == "trigger":
        row += f"  act={rec['activationId']} skips={rec['skippedFires']}"
    return row


def _agent_sort_key(name: str) -> tuple[int, str]:
    return (1, "") if name == UNATTRIBUTED else (0, name)


def _print_timelines(records: list[dict[str, Any]]) -> None:
    print("--- Per-agent timelines ---")
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_agent.setdefault(_agent_of(rec), []).append(rec)
    for agent in sorted(by_agent, key=_agent_sort_key):
        rows = by_agent[agent]
        print(f"{agent} ({len(rows)} ops)")
        for rec in rows:
            print(_fmt_row(rec))


def _leaderboard(records: list[dict[str, Any]], op_set: set[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for rec in records:
        if rec["op"] not in op_set:
            continue
        agent = _agent_of(rec)
        bucket = counts.setdefault(agent, {})
        bucket[rec["op"]] = bucket.get(rec["op"], 0) + 1
    return counts


def _print_leaderboard_section(title: str, counts: dict[str, dict[str, int]]) -> None:
    print(title)
    totals = sorted(
        counts.items(),
        key=lambda kv: (-sum(kv[1].values()), _agent_sort_key(kv[0])),
    )
    shown = totals[:5]
    for i, (agent, ops) in enumerate(shown, start=1):
        total = sum(ops.values())
        breakdown = ", ".join(
            f"{op}={n}" for op, n in sorted(ops.items(), key=lambda kv: -kv[1])
        )
        print(f"{i}. {agent:<16}{total}   ({breakdown})")
    if len(totals) > 5:
        print(f"  (+{len(totals) - 5} more)")


def _print_leaderboards(records: list[dict[str, Any]]) -> None:
    print("--- Top readers / writers ---")
    _print_leaderboard_section(
        "readers (sniff + read + trigger)", _leaderboard(records, READER_OPS)
    )
    _print_leaderboard_section(
        "writers (emit + inscribe + erase + register_scent + deregister_scent + evaporate)",
        _leaderboard(records, WRITER_OPS),
    )


def _print_trigger_fires(records: list[dict[str, Any]]) -> None:
    triggers = [r for r in records if r["op"] == "trigger"]
    print(f"--- Trigger fires ({len(triggers)}) ---")
    if not triggers:
        print("(none)")
        return
    for rec in triggers:
        print(
            f"{rec['targetId']:<22} {rec['activationId']:<30} {rec['outcome']:<10} "
            f"{rec['skippedFires']:<7} {rec['latencyMs']:g}ms"
        )


def _print_activation(records: list[dict[str, Any]], activation_id: str, path: Path) -> int:
    match = next(
        (r for r in records if r["op"] == "trigger" and r["activationId"] == activation_id),
        None,
    )
    if match is None:
        print(f"activation {activation_id} not found in {path}", file=sys.stderr)
        return 1

    print(f"--- Activation {activation_id} ---")
    print(f"{'scent:':<10}{match['targetId']}")
    print(f"{'agent:':<10}{_agent_of(match)}")
    fired_at = datetime.fromtimestamp(match["ts"] / 1000, tz=timezone.utc).isoformat()
    print(f"{'fired at:':<10}{fired_at} (ts={match['ts']})")
    print(f"{'outcome:':<10}{match['outcome']}")
    print(f"{'skipped:':<10}{match['skippedFires']}")
    print(f"{'latency:':<10}{match['latencyMs']:g}ms")
    print(f"{'journal line:':<10}seq={match['seq']}")
    print()
    print("--- nearby ops (same agent if known, ±1000ms, heuristic — not a causal link) ---")
    window_agent = match["agent"]
    nearby = [
        r
        for r in records
        if r is not match
        and abs(r["ts"] - match["ts"]) <= 1000
        and (window_agent is None or r["agent"] == window_agent)
    ]
    for rec in nearby[:20]:
        print(_fmt_row(rec))
    return 0


def _filter_agent(records: list[dict[str, Any]], agent_id: str) -> list[dict[str, Any]]:
    prefix = f"{agent_id}:"
    return [
        r
        for r in records
        if r["agent"] == agent_id
        or (r["targetId"] is not None and r["targetId"].startswith(prefix))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sbp-replay")
    parser.add_argument("path")
    parser.add_argument("--activation")
    parser.add_argument("--agent")
    args = parser.parse_args(argv)

    if args.activation is not None and args.agent is not None:
        print("--activation and --agent are mutually exclusive", file=sys.stderr)
        return 2

    path = Path(args.path)
    if not path.is_file():
        print(f"cannot read {path}", file=sys.stderr)
        return 2

    try:
        records, skipped = _parse_journal(path)
    except OSError as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        return 2

    agents = {_agent_of(r) for r in records}
    named_agents = agents - {UNATTRIBUTED}
    unattributed_note = " (+ unattributed)" if UNATTRIBUTED in agents else ""

    print(f"=== SBP replay: {path} ===")
    print(
        f"lines: {len(records)} parsed, {skipped} skipped (malformed) "
        f"| agents: {len(named_agents)}{unattributed_note}"
    )

    if args.agent is not None:
        records = _filter_agent(records, args.agent)

    if args.activation is not None:
        return _print_activation(records, args.activation, path)

    _print_timelines(records)
    _print_leaderboards(records)
    _print_trigger_fires(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
