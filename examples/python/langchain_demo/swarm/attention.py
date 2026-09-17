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
        for p in hot.pheromones:
            ledger[p.type] = ledger.get(p.type, 0.0) + p.current_intensity * interval_s
        for area, integral in ledger.items():
            bb.inscribe(InscribeParams(
                trail=TRAIL_LEDGER, key=area, value={"integral": integral},
                source_agent="attention-sampler",
            ))
