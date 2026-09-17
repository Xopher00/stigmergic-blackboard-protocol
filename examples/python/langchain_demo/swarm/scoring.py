"""Deterministic file-choice scorer -- the scout's own sensing of board state.

Combines heat attraction, a recency penalty, and claim exclusion into one score per
candidate file, with a seeded epsilon chance of ignoring the score entirely so the
swarm keeps exploring even while one area sits at maximum heat (TEST_IDEAS.md guard
#3). Pure and offline: reads only a snapshot it's handed, never a live blackboard --
that's what makes the guard tests fast, seeded, and LLM-free.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoardSnapshot:
    """What the scorer can see: claimed files, per-file recency penalty (the decayed
    intensity of that file's `visited` pheromone, 0 if never visited), and hot
    intensity per directory."""

    claimed: frozenset[str] = field(default_factory=frozenset)
    visited_intensity: dict[str, float] = field(default_factory=dict)
    hot_intensity_by_dir: dict[str, float] = field(default_factory=dict)


def dir_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def score_file(path: str, board: BoardSnapshot) -> float:
    """Higher is more attractive. Claimed files are never chosen deliberately --
    guarded separately by choose_next_file/suggest_next_files, not by score alone,
    so a scorer bug can't silently steal a claimed file."""
    heat = board.hot_intensity_by_dir.get(dir_of(path), 0.0)
    recency_penalty = board.visited_intensity.get(path, 0.0)
    return heat - recency_penalty


def choose_next_file(
    candidates: list[str],
    board: BoardSnapshot,
    rng: random.Random,
    epsilon: float = 0.15,
) -> str | None:
    """One scout's next-file decision. Ties are broken via rng, not sort order, to
    avoid a lexical bias toward early-alphabet files."""
    unclaimed = [c for c in candidates if c not in board.claimed]
    if not unclaimed:
        return None
    if rng.random() < epsilon:
        return rng.choice(unclaimed)
    scored = [(score_file(c, board), c) for c in unclaimed]
    best = max(s for s, _ in scored)
    return rng.choice([c for s, c in scored if s == best])


def suggest_next_files(
    candidates: list[str],
    board: BoardSnapshot,
    rng: random.Random,
    epsilon: float = 0.15,
    k: int = 3,
) -> list[str]:
    """A shortlist for the scout's tool call. This is sensing, not assignment -- the
    scout still chooses which (if any) of these to read."""
    unclaimed = [c for c in candidates if c not in board.claimed]
    if not unclaimed:
        return []
    if rng.random() < epsilon:
        shuffled = list(unclaimed)
        rng.shuffle(shuffled)
        return shuffled[:k]
    ranked = sorted(unclaimed, key=lambda c: score_file(c, board), reverse=True)
    return ranked[:k]
