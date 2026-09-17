"""Per-run token budget (TEST_IDEAS.md guard #5). This is an experiment-local cap that
never touches SDK code -- TASKS.md's "do not build: per-agent budget systems" line
rules out a server-scale protocol feature, not a local supervisor watching its own
run's spend.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware, after_model

from sbp.blackboard import get_shared_blackboard
from sbp.types import EmitParams

from swarm.board import DecayProfile, TRAIL_CONTROL, TYPE_HALT


@dataclass
class Budget:
    limit_tokens: int
    spent_tokens: int = 0
    tripped: bool = False

    def add(self, tokens: int) -> bool:
        """Adds tokens; returns True the instant the cap is newly crossed."""
        self.spent_tokens += tokens
        if not self.tripped and self.spent_tokens >= self.limit_tokens:
            self.tripped = True
            return True
        return False


def make_budget_middleware(budget: Budget, profile: DecayProfile) -> AgentMiddleware:
    """Sums each model call's usage into the shared budget; on first crossing, emits
    swarm.control/halt so every role's own scent condition stops it -- the halt
    propagates through the board, not through a direct call into each worker."""

    @after_model
    def _meter(state, runtime):
        total = sum(
            (getattr(m, "usage_metadata", None) or {}).get("total_tokens", 0)
            for m in state["messages"]
        )
        if total and budget.add(total):
            get_shared_blackboard().emit(EmitParams(
                trail=TRAIL_CONTROL, type=TYPE_HALT, intensity=1.0, decay=profile.control,
                payload={"reason": "token budget exceeded", "spent": budget.spent_tokens},
                source_agent="budget-supervisor",
            ))
        # {} not None: _log_step's node_update.get(...) has no None guard.
        return {}

    return _meter
