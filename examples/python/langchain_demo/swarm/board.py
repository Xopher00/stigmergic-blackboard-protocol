"""Board vocabulary: trail names and decay profiles shared by every role, plus a
BoardSnapshot reader for the scorer. Half-lives are minutes in live mode
(TEST_IDEAS.md: "think minutes, not hours"); scripted mode uses a scaled-down profile
so a deterministic run finishes in seconds of real wall-clock time without changing the
mechanism under test -- decay math is scale-invariant, only the constants differ.

sbp_emit's decay model always falls back to the emitting worker's single
SbpAgent.default_decay (confirmed in sbp/agent.py: `decay or self.default_decay`,
never per-call) -- a generic sbp_emit tool call can't choose it. So each trail here
gets its own purpose-built tool in roles.py calling SbpAgent.emit(..., decay=...)
directly, instead of relying on the generic sbp_emit tool for everything.
"""
from __future__ import annotations

from dataclasses import dataclass

from sbp.blackboard import LocalBlackboard
from sbp.types import DecayModel, ExponentialDecay, ImmortalDecay, SniffParams

from swarm.scoring import BoardSnapshot

TRAIL_CLAIMS = "swarm.claims"
TRAIL_VISITED = "swarm.visited"
TRAIL_EVIDENCE = "swarm.evidence"
TRAIL_HOT = "swarm.hot"
TRAIL_QUESTIONS = "swarm.questions"
TRAIL_COVERAGE = "swarm.coverage"
TRAIL_CONTROL = "swarm.control"
TRAIL_DOSSIER = "dossier"
TRAIL_LEDGER = "ledger"

TYPE_CLAIM = "claim"
TYPE_VISITED = "visited"
TYPE_WAKE = "wake"
TYPE_FILE = "file"
TYPE_HALT = "halt"

SCOUT_TRAILS = (TRAIL_CLAIMS, TRAIL_VISITED, TRAIL_EVIDENCE, TRAIL_COVERAGE, TRAIL_QUESTIONS)
BLOODHOUND_TRAILS = (TRAIL_EVIDENCE, TRAIL_HOT, TRAIL_QUESTIONS, TRAIL_DOSSIER)


def wake_trail(scout_id: str) -> str:
    return f"swarm.wake.{scout_id}"


@dataclass(frozen=True)
class DecayProfile:
    claim: DecayModel
    visited: DecayModel
    evidence: DecayModel
    hot: DecayModel
    question: DecayModel
    wake: DecayModel
    coverage: DecayModel
    control: DecayModel
    claim_heartbeat_s: float
    claim_live_threshold: float = 0.05
    hot_quiet_threshold: float = 0.1

    @staticmethod
    def live() -> "DecayProfile":
        return DecayProfile(
            claim=ExponentialDecay(half_life_ms=30_000),
            visited=ExponentialDecay(half_life_ms=300_000),
            evidence=ExponentialDecay(half_life_ms=180_000),
            hot=ExponentialDecay(half_life_ms=240_000),
            question=ExponentialDecay(half_life_ms=120_000),
            # on_scent fixes cooldown_ms=1000 regardless of profile; this half-life
            # must clear 0.5 well before that, so it isn't scaled by scripted().
            wake=ExponentialDecay(half_life_ms=300),
            coverage=ExponentialDecay(half_life_ms=300_000),
            control=ImmortalDecay(),
            claim_heartbeat_s=5.0,
        )

    @staticmethod
    def scripted(scale: float = 0.02) -> "DecayProfile":
        # Scaled-down live() so a deterministic run finishes in seconds, not minutes.
        live = DecayProfile.live()
        return DecayProfile(
            claim=ExponentialDecay(half_life_ms=max(300, int(live.claim.half_life_ms * scale))),
            visited=ExponentialDecay(half_life_ms=max(300, int(live.visited.half_life_ms * scale))),
            evidence=ExponentialDecay(half_life_ms=max(300, int(live.evidence.half_life_ms * scale))),
            hot=ExponentialDecay(half_life_ms=max(300, int(live.hot.half_life_ms * scale))),
            question=ExponentialDecay(half_life_ms=max(300, int(live.question.half_life_ms * scale))),
            wake=live.wake,
            coverage=ExponentialDecay(half_life_ms=max(300, int(live.coverage.half_life_ms * scale))),
            control=ImmortalDecay(),
            claim_heartbeat_s=0.5,
        )


def dir_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def read_board_snapshot(bb: LocalBlackboard, profile: DecayProfile) -> BoardSnapshot:
    """Direct-from-blackboard view used by the run supervisor and by tests; scouts get
    the same information through the suggest_files tool built on top of this."""
    claims = bb.sniff(SniffParams(trails=[TRAIL_CLAIMS], types=[TYPE_CLAIM],
                                   min_intensity=profile.claim_live_threshold))
    claimed = frozenset(
        f for p in claims.pheromones if (f := p.payload.get("file"))
    )

    visited = bb.sniff(SniffParams(trails=[TRAIL_VISITED], types=[TYPE_VISITED]))
    visited_intensity: dict[str, float] = {}
    for p in visited.pheromones:
        f = p.payload.get("file")
        if f:
            visited_intensity[f] = max(visited_intensity.get(f, 0.0), p.current_intensity)

    hot = bb.sniff(SniffParams(trails=[TRAIL_HOT]))
    hot_by_dir: dict[str, float] = {}
    for p in hot.pheromones:
        hot_by_dir[p.type] = max(hot_by_dir.get(p.type, 0.0), p.current_intensity)

    return BoardSnapshot(claimed=claimed, visited_intensity=visited_intensity,
                          hot_intensity_by_dir=hot_by_dir)
