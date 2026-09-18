"""Periodic sampler integrating hot-area intensity over time into a durable ledger
trace, so the judge ranks findings by sustained attention rather than peak intensity
(TEST_IDEAS.md: "how much attention an area held over the whole run ... not by how
loudly the signals peaked"). Written as traces, not pheromones, so the integral
survives the hot signal's own decay.
"""
from __future__ import annotations

import asyncio

from sbp.blackboard import LocalBlackboard
from sbp.types import InscribeParams, SniffParams

from swarm.board import TRAIL_HOT, TRAIL_LEDGER


async def run_attention_sampler(bb: LocalBlackboard, interval_s: float, stop: asyncio.Event) -> None:
    ledger: dict[str, float] = {}

    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
        hot = bb.sniff(SniffParams(trails=[TRAIL_HOT]))
        # Only areas that moved this tick: a cold area's integral is unchanged, and
        # re-inscribing every area ever seen costs a trace write per area per tick.
        for p in hot.pheromones:
            ledger[p.type] = ledger.get(p.type, 0.0) + p.current_intensity * interval_s
            bb.inscribe(InscribeParams(
                trail=TRAIL_LEDGER, key=p.type, value={"integral": ledger[p.type]},
                source_agent="attention-sampler",
            ))
