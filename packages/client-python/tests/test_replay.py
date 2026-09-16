"""
P3.2 replay-CLI tests (12). sbp.replay.main(argv) is a pure in-process entry point:
reads an operation journal (the frozen P3.1 JSONL schema, camelCase, all 10 keys),
prints per-agent timelines / reader-writer leaderboard / trigger-fires summary, and
returns exit codes 0 ok / 1 activation not found / 2 unreadable file or usage error.

Frozen fixtures: good.jsonl (11 well-formed ops) and corrupt.jsonl (4 good lines +
one truncated mid-write line). Malformed lines are skipped with a stderr warning and
counted, never fatal. TS spellings registerScent/deregisterScent normalize to
register_scent/deregister_scent before counting; readers = sniff+read+trigger,
writers = emit+inscribe+erase+register_scent+deregister_scent+evaporate, inspect
counts toward neither.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sbp.replay import main

GOOD_LINES = [
    '{"seq":0,"ts":1700000000000,"agent":"agent-1","op":"emit","trail":"t","targetId":"p-1","outcome":"ok","latencyMs":0.5,"activationId":null,"skippedFires":null}',
    '{"seq":1,"ts":1700000000001,"agent":null,"op":"sniff","trail":"t","targetId":null,"outcome":"ok","latencyMs":0.2,"activationId":null,"skippedFires":null}',
    '{"seq":2,"ts":1700000000002,"agent":"agent-2","op":"inscribe","trail":"notes","targetId":"k1","outcome":"ok","latencyMs":0.3,"activationId":null,"skippedFires":null}',
    '{"seq":3,"ts":1700000000003,"agent":"agent-2","op":"inscribe","trail":"notes","targetId":"k1","outcome":"ok","latencyMs":0.3,"activationId":null,"skippedFires":null}',
    '{"seq":4,"ts":1700000000004,"agent":"agent-2","op":"read","trail":"notes","targetId":"k1","outcome":"ok","latencyMs":0.1,"activationId":null,"skippedFires":null}',
    '{"seq":5,"ts":1700000000005,"agent":"agent-2","op":"read","trail":"notes","targetId":"k*","outcome":"not_found","latencyMs":0.1,"activationId":null,"skippedFires":null}',
    '{"seq":6,"ts":1700000000006,"agent":null,"op":"register_scent","trail":null,"targetId":"s2","outcome":"ok","latencyMs":0.4,"activationId":null,"skippedFires":null}',
    '{"seq":7,"ts":1700000000007,"agent":null,"op":"trigger","trail":null,"targetId":"s2","outcome":"ok","latencyMs":12.5,"activationId":"s2@1700000000007","skippedFires":1}',
    '{"seq":8,"ts":1700000000008,"agent":"agent-1","op":"erase","trail":"notes","targetId":"k1","outcome":"ok","latencyMs":0.2,"activationId":null,"skippedFires":null}',
    '{"seq":9,"ts":1700000000009,"agent":"agent-1","op":"sniff","trail":"t,notes","targetId":null,"outcome":"ok","latencyMs":0.2,"activationId":null,"skippedFires":null}',
    '{"seq":10,"ts":1700000000010,"agent":"agent-2","op":"sniff","trail":"notes","targetId":null,"outcome":"ok","latencyMs":0.2,"activationId":null,"skippedFires":null}',
]

# Journal truncated mid-write (process killed); do NOT close the JSON object.
TRUNCATED_LINE = '{"seq":4,"ts":1700000000011,"agent":"agent-9","op":"emi'


@pytest.fixture
def good(tmp_path: Path) -> Path:
    p = tmp_path / "good.jsonl"
    p.write_text("\n".join(GOOD_LINES) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def corrupt(tmp_path: Path) -> Path:
    p = tmp_path / "corrupt.jsonl"
    p.write_text("\n".join(GOOD_LINES[:4]) + "\n" + TRUNCATED_LINE, encoding="utf-8")
    return p


def test_default_view_header_and_counts(good, capsys):
    rc = main([str(good)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== SBP replay:" in out
    assert "lines: 11 parsed, 0 skipped" in out
    assert "| agents: 2 (+ unattributed)" in out


def test_per_agent_timeline(good, capsys):
    rc = main([str(good)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--- Per-agent timelines ---" in out
    i_agent1 = out.index("agent-1 (3 ops)")
    i_agent2 = out.index("agent-2 (5 ops)")
    i_unattr = out.index("(unattributed) (3 ops)")
    assert i_agent1 < i_agent2 < i_unattr
    for s in ("trail=t", "target=p-1", "target=k*", "not_found"):
        assert s in out


def test_leaderboard_readers(good, capsys):
    rc = main([str(good)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "readers (sniff + read + trigger)" in out
    assert "1. agent-2         3   (read=2, sniff=1)" in out
    assert "2. (unattributed)  2   (sniff=1, trigger=1)" in out
    assert "3. agent-1         1   (sniff=1)" in out


def test_leaderboard_writers(good, capsys):
    rc = main([str(good)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "writers (emit + inscribe + erase + register_scent + deregister_scent + evaporate)" in out
    assert "1. agent-1         2   (emit=1, erase=1)" in out
    assert "2. agent-2         2   (inscribe=2)" in out
    assert "3. (unattributed)  1   (register_scent=1)" in out


def test_trigger_fires_section(good, capsys):
    rc = main([str(good)])
    out = capsys.readouterr().out
    assert rc == 0
    start = out.index("--- Trigger fires (1) ---")
    tail = out[start:]
    for s in ("s2", "s2@1700000000007", "ok", "1", "12.5ms"):
        assert s in tail


def test_activation_found(good, capsys):
    rc = main([str(good), "--activation", "s2@1700000000007"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--- Activation s2@1700000000007 ---" in out
    assert "scent:    s2" in out
    assert "outcome:  ok" in out
    assert "skipped:  1" in out
    assert "heuristic" in out


def test_activation_not_found(good, capsys):
    rc = main([str(good), "--activation", "nope@1"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "activation nope@1 not found" in captured.err
    assert "--- Activation" not in captured.out


def test_corrupt_last_line(corrupt, capsys):
    rc = main([str(corrupt)])
    captured = capsys.readouterr()
    assert rc == 0
    assert f"warning: {corrupt}:5: skipping malformed line" in captured.err
    assert "lines: 4 parsed, 1 skipped" in captured.out


def test_missing_file(tmp_path, capsys):
    rc = main([str(tmp_path / "missing.jsonl")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "cannot read" in captured.err


def test_agent_filter(good, capsys):
    rc = main([str(good), "--agent", "agent-1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "agent-2" not in out
    assert "--- Trigger fires (0) ---" in out
    assert "(none)" in out


def test_activation_and_agent_mutually_exclusive(good, capsys):
    rc = main([str(good), "--activation", "s2@1700000000007", "--agent", "agent-1"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--activation and --agent are mutually exclusive" in captured.err


def test_entrypoints(good):
    pkg_root = Path(__file__).parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pkg_root / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    r = subprocess.run(
        [sys.executable, "-m", "sbp.replay", str(good)],
        capture_output=True, text=True, cwd=str(pkg_root), env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "=== SBP replay:" in r.stdout

    exe = shutil.which("sbp-replay")
    if exe is None:
        pytest.skip("sbp-replay not installed")
    r2 = subprocess.run([exe, str(good)], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "=== SBP replay:" in r2.stdout


if __name__ == "__main__":
    sys.exit(main())
