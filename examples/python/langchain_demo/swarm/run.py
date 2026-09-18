"""Single-command entry point for the foraging swarm experiment (TEST_IDEAS.md).

Scripted mode (default): zero API cost, deterministic ScriptedChatModel fleet, a tiny
fixed corpus. Must pass before live mode is allowed. Live mode: real OpenRouter models
against a real decompiled corpus -- point --apk at any .apk (see swarm/decompile.py).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from pathlib import Path

import sbp.blackboard as blackboard_module
from sbp.agent import SbpAgent
from sbp.blackboard import LocalBlackboard, get_shared_blackboard
from sbp.client import AsyncSbpClient
from sbp.types import EmitParams

from swarm import roles
from swarm.attention import run_attention_sampler
from swarm.board import DecayProfile
from swarm.budget import Budget, make_budget_middleware
from swarm.corpus import Corpus, load_corpus
from swarm.report import write_run_summary
from swarm.scripted import bloodhound_model, judge_model, scout_model, write_scripted_corpus


# "Nothing hot" is trivially true before anything is promoted, so the settled
# condition can fire while bloodhound is still mid-confirmation (see EXPERIMENTS.md).
FINAL_REPORT_GRACE_S = 15.0


def _all_files_covered(bb, corpus: Corpus) -> bool:
    from sbp.types import SniffParams
    covered = bb.sniff(SniffParams(trails=[roles.board.TRAIL_COVERAGE], types=[roles.board.TYPE_FILE]))
    seen = {p.payload.get("file") for p in covered.pheromones}
    return set(corpus.files) <= seen


async def _revive_stalled_scouts(
    scouts, profile: DecayProfile, corpus: Corpus, interval_s: float = 8.0,
) -> None:
    """A crashed activation (e.g. GraphRecursionError) never calls continue_watching,
    so the scout would otherwise sit dead forever. Re-poke any idle scout on an
    interval -- but never once every file is covered, since active_activations==0
    then means "legitimately done," not "stalled," and re-poking would just spend
    tokens re-discovering that repeatedly for no benefit."""
    bb = get_shared_blackboard()
    while True:
        await asyncio.sleep(interval_s)
        idle = [s for s in scouts if s.active_activations == 0]
        if not idle or _all_files_covered(bb, corpus):
            continue
        for s in idle:
            await bb.evaluate_scents()
            wake = roles.board.wake_trail(s.agent_id)
            bb.emit(EmitParams(trail=wake, type=roles.board.TYPE_WAKE, intensity=1.0,
                                decay=profile.wake, source_agent="revive-supervisor"))


async def _run_swarm(
    corpus: Corpus, profile: DecayProfile, scouts, bloodhound_worker, judge_worker,
    budget: Budget, out_dir: Path, timeout_s: float, wake_intensity: float = 1.0,
    enable_revive: bool = True,
) -> dict:
    bb = get_shared_blackboard()
    start = time.monotonic()

    done = asyncio.Event()
    observer = SbpAgent(agent_id="observer", local=True)
    end_condition = roles.end_of_run_condition(len(corpus.files), profile)

    @observer.on_scent("observer:end-of-run", end_condition)
    async def _on_end(trigger) -> None:
        done.set()

    all_workers = [*scouts, bloodhound_worker, judge_worker]
    for w in all_workers:
        await w.sbp_agent.start()
    await observer.start()
    if enable_revive:
        await roles.register_bloodhound_self_watch(bloodhound_worker)

    worker_tasks = [asyncio.create_task(w.run()) for w in all_workers]
    observer_task = asyncio.create_task(observer.run())
    heartbeat_tasks = [asyncio.create_task(roles.claim_heartbeat(s, profile)) for s in scouts]
    # Off for scripted mode: an unplanned extra wake desyncs its fixed-length script.
    revive_task = asyncio.create_task(_revive_stalled_scouts(scouts, profile, corpus)) if enable_revive else None

    sampler_stop = asyncio.Event()
    sampler_task = asyncio.create_task(run_attention_sampler(bb, 0.2, sampler_stop))

    bootstrap = AsyncSbpClient(local=True, agent_id="bootstrap")
    await bootstrap.connect()

    ended_by = "natural"
    try:
        for s in scouts:
            wake_trail = roles.board.wake_trail(s.agent_id)
            await bootstrap.emit(wake_trail, roles.board.TYPE_WAKE, intensity=wake_intensity,
                                  decay=profile.wake, payload={})
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
            # Give bloodhound one more self-watch window, then re-run the judge
            # (write_report overwrites) so its report isn't stale.
            await asyncio.sleep(FINAL_REPORT_GRACE_S)
            await judge_worker.force_trigger("supervisor:final-report")
        except asyncio.TimeoutError:
            ended_by = "timeout"
        if budget.tripped and not done.is_set():
            ended_by = "budget"
    finally:
        await bootstrap.close()
        for w in all_workers:
            w.stop()
        for t in worker_tasks:
            await t
        await observer.stop()
        await observer_task
        for t in heartbeat_tasks:
            t.cancel()
        if revive_task is not None:
            revive_task.cancel()
        sampler_stop.set()
        await sampler_task

    return {
        "files_total": len(corpus.files),
        "ended_by": "budget" if budget.tripped else ended_by,
        "budget_spent_tokens": budget.spent_tokens,
        "duration_s": round(time.monotonic() - start, 3),
    }


def _setup(out_dir: Path, seed: int, profile: DecayProfile, budget_tokens: int):
    """Installs the journaling blackboard singleton (must happen before any client
    connects) and returns the per-run state every mode needs."""
    blackboard_module._shared_blackboard = LocalBlackboard(journal_path=str(out_dir / "journal.jsonl"))
    budget = Budget(limit_tokens=budget_tokens)
    return random.Random(seed), budget, [make_budget_middleware(budget, profile)]


async def run_scripted(out_dir: Path, seed: int = 0, timeout_s: float = 30.0) -> dict:
    corpus_root = out_dir / "scripted_corpus"
    write_scripted_corpus(corpus_root)
    corpus = load_corpus(corpus_root)

    profile = DecayProfile.scripted()
    rng, budget, middleware = _setup(out_dir, seed, profile, budget_tokens=10**9)

    scouts = [
        roles.build_scout("scout-1", scout_model([
            ("app/auth/LoginActivity.java", "hardcoded_secret"),
            ("app/util/Logger.java", "insecure_logging"),
        ]), corpus, profile, rng, middleware=middleware),
        roles.build_scout("scout-2", scout_model([
            ("app/auth/SessionManager.java", "hardcoded_secret"),
            ("app/util/CryptoHelper.java", None),
        ]), corpus, profile, rng, middleware=middleware),
    ]
    bloodhound_worker = roles.build_bloodhound(bloodhound_model(), corpus, profile, middleware=middleware)
    report_path = out_dir / "report.md"
    judge_worker = roles.build_judge(judge_model(), len(corpus.files), profile, report_path,
                                      middleware=middleware)

    summary = await _run_swarm(corpus, profile, scouts, bloodhound_worker, judge_worker,
                                budget, out_dir, timeout_s, enable_revive=False)
    summary["report_path"] = str(report_path)
    summary["mode"] = "scripted"
    write_run_summary(out_dir / "run_summary.json", **summary)
    return summary


def _openrouter_model(model_slug: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_slug, base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"], timeout=60,
    )


async def run_live(
    out_dir: Path, seed: int, timeout_s: float, budget_tokens: int,
    scout_models: list[str], bloodhound_model_slug: str, judge_model_slug: str,
    apk: str | None = None, source: Path | None = None,
) -> dict:
    """Exactly one of apk (a path or http(s) URL to an .apk -- decompiled and its
    own package auto-discovered from its manifest, no hand-picked path) or source
    (an already-known corpus directory, e.g. from a prior --apk run) is required."""
    from dotenv import load_dotenv
    load_dotenv()

    if apk:
        from swarm.decompile import prepare_corpus
        corpus = prepare_corpus(apk, out_dir.parent / ".work")
    elif source:
        corpus = load_corpus(source, exclude=("R.java", "BuildConfig.java"))
    else:
        raise SystemExit("run_live needs either apk= or source=")
    if not corpus.files:
        raise SystemExit(f"no files found for apk={apk!r} source={source!r}")

    profile = DecayProfile.live()
    rng, budget, middleware = _setup(out_dir, seed, profile, budget_tokens)

    scouts = [
        roles.build_scout(f"scout-{i+1}", _openrouter_model(slug), corpus, profile, rng,
                           middleware=middleware)
        for i, slug in enumerate(scout_models)
    ]
    bloodhound_worker = roles.build_bloodhound(_openrouter_model(bloodhound_model_slug), corpus,
                                                profile, middleware=middleware)
    report_path = out_dir / "report.md"
    judge_worker = roles.build_judge(_openrouter_model(judge_model_slug), len(corpus.files), profile,
                                      report_path, middleware=middleware)

    summary = await _run_swarm(corpus, profile, scouts, bloodhound_worker, judge_worker,
                                budget, out_dir, timeout_s)
    summary["report_path"] = str(report_path)
    summary["mode"] = "live"
    summary["source"] = str(corpus.root)
    write_run_summary(out_dir / "run_summary.json", **summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Foraging swarm experiment")
    parser.add_argument("--mode", choices=["scripted", "live"], default="scripted")
    parser.add_argument("--apk", default=None, help="path or http(s) URL to an .apk -- "
                         "decompiled and its own package auto-discovered from its manifest")
    parser.add_argument("--source", default=None,
                         help="an already-decompiled corpus directory, as an alternative to --apk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--budget-tokens", type=int, default=3_000_000)
    parser.add_argument("--out", default="swarm/out")
    parser.add_argument("--scout-models", nargs="+", default=None)
    parser.add_argument("--bloodhound-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--force", action="store_true",
                         help="allow --mode live without a scripted run in this invocation")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)

    if args.mode == "scripted":
        summary = asyncio.run(run_scripted(out_dir, seed=args.seed, timeout_s=args.timeout_s or 30.0))
        print(f"scripted run: {summary}")
        return 0

    if not args.force:
        scripted_summary = asyncio.run(run_scripted(out_dir / "_preflight", seed=args.seed, timeout_s=30.0))
        if scripted_summary["ended_by"] not in ("natural", "budget"):
            print(f"scripted preflight did not finish cleanly ({scripted_summary}); refusing live run")
            return 1

    if not args.apk and not args.source:
        print("--mode live needs either --apk <path-or-url> or --source <corpus-dir>")
        return 1

    default_model = os.environ.get("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
    scout_models = args.scout_models or [default_model, default_model]
    summary = asyncio.run(run_live(
        out_dir, args.seed, args.timeout_s or 600.0, args.budget_tokens,
        scout_models, args.bloodhound_model or default_model, args.judge_model or default_model,
        apk=args.apk, source=Path(args.source) if args.source else None,
    ))
    print(f"live run: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
