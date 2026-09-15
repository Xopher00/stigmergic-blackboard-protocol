#!/usr/bin/env python3
"""
mutation_check.py -- prove the Python test suite (packages/client-python/tests/ and
examples/python/langchain_demo/tests/) has real teeth, by mutation testing.

Why this exists
----------------
A test suite written by the same author as the implementation, that passes on the
first run, might be vacuously green: it could pass even if the behavior it claims to
verify were broken. This script does not trust that -- it deliberately reintroduces
five real historical bugs (one at a time), runs the specific pytest target that is
supposed to catch each one, records whether the suite actually goes red and which
assertion fails, and then restores the file to its original (fixed) content before
moving to the next mutation.

Every mutation is a pure, in-memory string replacement: original file bytes are
captured before mutating, the mutated text is written, pytest runs, and then the
original bytes are written back -- so the repo is guaranteed to end up byte-identical
to how it started, regardless of test outcome or a crash mid-run (the `finally` block
below restores on any exception too).

How to run
----------
From the repo root, using the project venv:

    .venv/bin/python packages/client-python/scripts/mutation_check.py

Optional: run a single mutation by id (1-5):

    .venv/bin/python packages/client-python/scripts/mutation_check.py --only 3

Exit code is 0 if every mutation behaved as expected (went red) and every file was
restored cleanly; 1 otherwise. This is NOT a pytest file on purpose -- it deliberately
breaks working code, which pytest collection/parallelism must never be allowed to do
to the real suite.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _p(rel: str) -> Path:
    return REPO_ROOT / rel


# Each mutation's `replacements` list is applied to `file` in order; each `old` must
# match exactly once (checked) so a mutation never silently no-ops.
MUTATIONS = [
    {
        "id": 1,
        "title": "Trail-filtering all-or-nothing rejection",
        "file": "examples/python/langchain_demo/sbp_worker.py",
        "replacements": [
            (
                '    def _split_trails(self, trails: list[str]) -> tuple[list[str], list[str]]:\n'
                '        """Serve what\'s allowed instead of rejecting the whole call over one bad\n'
                '        name — returns (allowed, skipped), never silently drops the useful part."""\n'
                '        if self._allowed_trails is None:\n'
                '            return trails, []\n'
                '        allowed = [t for t in trails if t in self._allowed_trails]\n'
                '        skipped = [t for t in trails if t not in self._allowed_trails]\n'
                '        return allowed, skipped\n',

                '    def _split_trails(self, trails: list[str]) -> tuple[list[str], list[str]]:\n'
                '        """MUTATION #1 (temporary, for mutation_check.py): all-or-nothing rejection —\n'
                '        if ANY requested trail is disallowed, the whole call is rejected (nothing served)."""\n'
                '        if self._allowed_trails is None:\n'
                '            return trails, []\n'
                '        if any(t not in self._allowed_trails for t in trails):\n'
                '            return [], trails\n'
                '        return trails, []\n',
            ),
        ],
        "test_target": ["tests/test_sbp_worker.py::TestTrailFiltering"],
        "run_cwd": "examples/python/langchain_demo",
        "expect_red": True,
    },
    {
        "id": 2,
        "title": "Default cooldown allowing re-fire (targets test correctness, not impl)",
        "file": "packages/client-python/tests/test_agent.py",
        "replacements": [
            (
                '        result = await running_agent.register_scent(\n'
                '            "dynamic", ThresholdCondition(trail="t", signal_type="e", value=0.5), cooldown_ms=10_000\n'
                '        )',
                '        result = await running_agent.register_scent(\n'
                '            "dynamic", ThresholdCondition(trail="t", signal_type="e", value=0.5)\n'
                '        )',
            ),
            (
                '        @agent.when("t", "e", value=0.5, cooldown_ms=10_000)',
                '        @agent.when("t", "e", value=0.5)',
            ),
        ],
        "test_target": [
            "packages/client-python/tests/test_agent.py::TestLiveAgentOperations::"
            "test_register_and_deregister_scent_callable_while_running",
            "packages/client-python/tests/test_agent.py::TestRunLifecycle::"
            "test_staged_when_scent_fires_after_run",
        ],
        "run_cwd": ".",
        "expect_red": True,
        "note": (
            "cooldown_ms=0 is a legitimate default; this targets the TEST's own assumption "
            "that cooldown_ms=10_000 is load-bearing against the shared blackboard's polling "
            "evaluation loop re-firing within the sleep window, not a blackboard.py bug."
        ),
    },
    {
        "id": 3,
        "title": "Inscribe action-surfacing hardcoded to 'created'",
        "file": "packages/client-python/src/sbp/blackboard.py",
        "replacements": [
            (
                '            return InscribeResult(\n'
                '                trace_id=existing.id,\n'
                '                action="updated",\n'
                '                version=existing.version,\n'
                '            )',
                '            return InscribeResult(\n'
                '                trace_id=existing.id,\n'
                '                action="created",  # MUTATION #3 (temporary, for mutation_check.py): hardcoded\n'
                '                version=existing.version,\n'
                '            )',
            ),
        ],
        "test_target": [
            "packages/client-python/tests/test_blackboard.py::TestInscribeReadErase::"
            "test_inscribe_created_then_updated_with_version_increment"
        ],
        "run_cwd": ".",
        "expect_red": True,
    },
    {
        "id": 4,
        "title": "Logging lost on exception (buffer-then-log instead of log-as-you-go)",
        "file": "examples/python/langchain_demo/sbp_worker.py",
        "replacements": [
            (
                '        self.active_activations += 1\n'
                '        try:\n'
                '            # astream (not ainvoke) so each step is logged as it happens -- ainvoke only\n'
                '            # returns messages on success, losing everything if recursion_limit is hit.\n'
                '            async for chunk in self._llm_agent.astream(\n'
                '                {"messages": [{"role": "user", "content": content}]},\n'
                '                config=self._invoke_config, stream_mode="updates",\n'
                '            ):\n'
                '                self._log_step(chunk)\n'
                '        finally:\n'
                '            self.active_activations -= 1',

                '        self.active_activations += 1\n'
                '        try:\n'
                '            # MUTATION #4 (temporary, for mutation_check.py): buffer chunks and only\n'
                '            # log after the full loop completes -- an exception mid-stream loses all output.\n'
                '            chunks = []\n'
                '            async for chunk in self._llm_agent.astream(\n'
                '                {"messages": [{"role": "user", "content": content}]},\n'
                '                config=self._invoke_config, stream_mode="updates",\n'
                '            ):\n'
                '                chunks.append(chunk)\n'
                '            for chunk in chunks:\n'
                '                self._log_step(chunk)\n'
                '        finally:\n'
                '            self.active_activations -= 1',
            ),
        ],
        "test_target": [
            "tests/test_sbp_worker.py::TestActiveActivationsCounter::test_returns_to_zero_after_recursion_error"
        ],
        "run_cwd": "examples/python/langchain_demo",
        "expect_red": True,
    },
    {
        "id": 5,
        "title": "evaporate()/inspect() stubbed to no-ops, ignoring filter params",
        "file": "packages/client-python/src/sbp/blackboard.py",
        "replacements": [
            (
                '    def evaporate(self, params: EvaporateParams) -> EvaporateResult:\n'
                '        """Force evaporation of pheromones matching criteria."""\n'
                '        now = self._now()\n'
                '        to_remove: List[str] = []\n'
                '        trails_affected: set[str] = set()\n'
                '\n'
                '        for pid, p in self.pheromones.items():\n'
                '            if params.trail and p.trail != params.trail:\n'
                '                continue\n'
                '            if params.types and p.type not in params.types:\n'
                '                continue\n'
                '            if params.older_than_ms is not None and now - p.emitted_at < params.older_than_ms:\n'
                '                continue\n'
                '            if params.below_intensity is not None and compute_intensity(p, now) >= params.below_intensity:\n'
                '                continue\n'
                '            if params.tags and not match_tags(p.tags, params.tags):\n'
                '                continue\n'
                '            to_remove.append(pid)\n'
                '            trails_affected.add(p.trail)\n'
                '\n'
                '        for pid in to_remove:\n'
                '            del self.pheromones[pid]\n'
                '\n'
                '        return EvaporateResult(\n'
                '            evaporated_count=len(to_remove),\n'
                '            trails_affected=list(trails_affected),\n'
                '        )',

                '    def evaporate(self, params: EvaporateParams) -> EvaporateResult:\n'
                '        """MUTATION #5 (temporary, for mutation_check.py): stubbed no-op, ignores filters."""\n'
                '        return EvaporateResult(\n'
                '            evaporated_count=0,\n'
                '            trails_affected=[],\n'
                '        )',
            ),
            (
                '        """Inspect blackboard state."""\n'
                '        now = self._now()\n'
                '        include = params.include or ["trails", "scents", "stats"]\n'
                '        result = InspectResult(timestamp=now)\n'
                '\n'
                '        if "trails" in include:\n'
                '            trail_map: Dict[str, Dict[str, float]] = {}\n'
                '            for p in self.pheromones.values():\n'
                '                if is_evaporated(p, now):\n'
                '                    continue\n'
                '                data = trail_map.setdefault(p.trail, {"count": 0, "intensity": 0.0})\n'
                '                data["count"] += 1\n'
                '                data["intensity"] += compute_intensity(p, now)\n'
                '            result.trails = [\n'
                '                {\n'
                '                    "name": name,\n'
                '                    "pheromone_count": int(data["count"]),\n'
                '                    "total_intensity": data["intensity"],\n'
                '                    "avg_intensity": data["intensity"] / data["count"] if data["count"] else 0,\n'
                '                }\n'
                '                for name, data in trail_map.items()\n'
                '            ]\n'
                '\n'
                '        if "scents" in include:\n'
                '            result.scents = [\n'
                '                {\n'
                '                    "scent_id": s["id"],\n'
                '                    "condition_met": s["last_condition_met"],\n'
                '                    "in_cooldown": bool(s["last_triggered_at"]) and (now - s["last_triggered_at"] < s["cooldown_ms"]),\n'
                '                    "last_triggered_at": s["last_triggered_at"] or None,\n'
                '                }\n'
                '                for s in self.scents.values()\n'
                '            ]\n'
                '\n'
                '        if "stats" in include:\n'
                '            active_count = sum(1 for p in self.pheromones.values() if not is_evaporated(p, now))\n'
                '            result.stats = {\n'
                '                "total_pheromones": len(self.pheromones),\n'
                '                "active_pheromones": active_count,\n'
                '                "total_scents": len(self.scents),\n'
                '                "total_traces": len(self.traces),\n'
                '                "uptime_ms": now - self.start_time,\n'
                '            }\n'
                '\n'
                '        return result',

                '        """MUTATION #5 (temporary, for mutation_check.py): stubbed no-op, ignores include filter."""\n'
                '        now = self._now()\n'
                '        return InspectResult(timestamp=now)',
            ),
        ],
        "test_target": [
            "packages/client-python/tests/test_blackboard.py::TestEvaporate",
            "packages/client-python/tests/test_blackboard.py::TestInspect",
        ],
        "run_cwd": ".",
        "expect_red": True,
    },
]


def apply_replacements(content: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        count = content.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one occurrence of the search text, found {count}. "
                f"Update MUTATIONS in mutation_check.py to match the current source."
            )
        content = content.replace(old, new, 1)
    return content


def run_pytest(run_cwd: Path, test_targets: list[str]) -> tuple[bool, str]:
    cmd = [str(VENV_PYTHON), "-m", "pytest", "-q", *test_targets]
    proc = subprocess.run(cmd, cwd=str(run_cwd), capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, output


def extract_failure_lines(output: str) -> str:
    lines = output.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("E ") or stripped.startswith("E\t"):
            return stripped
    for line in lines:
        if "assert" in line.lower() or "error" in line.lower():
            return line.strip()
    return "(no assertion line found -- see full output)"


def run_one(mutation: dict) -> dict:
    file_path = _p(mutation["file"])
    original = file_path.read_text()

    result = {
        "id": mutation["id"],
        "title": mutation["title"],
        "file": mutation["file"],
        "restored": False,
        "went_red": None,
        "failure_line": None,
        "error": None,
    }

    try:
        mutated = apply_replacements(original, mutation["replacements"])
        file_path.write_text(mutated)

        run_cwd = _p(mutation["run_cwd"])
        passed, output = run_pytest(run_cwd, mutation["test_target"])

        result["went_red"] = not passed
        if not passed:
            result["failure_line"] = extract_failure_lines(output)
        result["full_output"] = output
    except Exception as exc:  # noqa: BLE001 -- report, then still restore below
        result["error"] = str(exc)
    finally:
        file_path.write_text(original)
        restored_ok = file_path.read_text() == original
        result["restored"] = restored_ok
        if not restored_ok:
            raise RuntimeError(
                f"FAILED TO RESTORE {mutation['file']} to its original content! "
                f"Manual intervention required (git checkout -- {mutation['file']})."
            )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=int, choices=[m["id"] for m in MUTATIONS], default=None,
                         help="Run only this mutation id (1-5).")
    args = parser.parse_args()

    if not VENV_PYTHON.exists():
        print(f"venv python not found at {VENV_PYTHON} -- run from a checkout with .venv set up.")
        return 1

    mutations = [m for m in MUTATIONS if args.only is None or m["id"] == args.only]

    results = []
    all_ok = True
    for mutation in mutations:
        print(f"\n=== Mutation #{mutation['id']}: {mutation['title']} ===")
        print(f"file: {mutation['file']}")
        if "note" in mutation:
            print(f"note: {mutation['note']}")
        result = run_one(mutation)
        results.append(result)

        if result["error"]:
            print(f"ERROR applying/running mutation: {result['error']}")
            all_ok = False
            continue

        status = "RED (test failed, as expected)" if result["went_red"] else "GREEN (test still passed -- suite missed it!)"
        print(f"result: {status}")
        if result["failure_line"]:
            print(f"failing assertion: {result['failure_line']}")
        print(f"restored cleanly: {result['restored']}")

        expected_red = mutation.get("expect_red", True)
        if result["went_red"] != expected_red:
            all_ok = False

    print("\n=== Summary ===")
    for r in results:
        mark = "PASS" if (r.get("went_red") and r["restored"] and not r["error"]) else "CHECK"
        print(f"[{mark}] #{r['id']} {r['title']}: went_red={r['went_red']} restored={r['restored']}")

    print(f"\nOverall: {'all mutations demonstrated real test coverage' if all_ok else 'see CHECK lines above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
