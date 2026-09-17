"""Offline, seeded tests for the pure file-choice scorer (TEST_IDEAS.md guard #3:
exploration keeps happening even at maximum heat). No LLM, no blackboard."""
import random

from swarm.scoring import BoardSnapshot, choose_next_file, score_file, suggest_next_files


def test_unclaimed_hot_file_scores_higher_than_a_cold_one():
    board = BoardSnapshot(hot_intensity_by_dir={"a": 1.0, "b": 0.0})
    assert score_file("a/x.java", board) > score_file("b/y.java", board)


def test_recently_visited_file_scores_lower_than_an_unvisited_one():
    board = BoardSnapshot(visited_intensity={"a/x.java": 0.9})
    assert score_file("a/x.java", board) < score_file("a/y.java", board)


def test_choose_next_file_never_returns_a_claimed_file():
    candidates = ["a/x.java", "a/y.java"]
    board = BoardSnapshot(claimed=frozenset({"a/x.java"}))
    rng = random.Random(0)
    for _ in range(50):
        assert choose_next_file(candidates, board, rng, epsilon=0.5) == "a/y.java"


def test_choose_next_file_returns_none_when_everything_is_claimed():
    board = BoardSnapshot(claimed=frozenset({"a/x.java"}))
    assert choose_next_file(["a/x.java"], BoardSnapshot(claimed=board.claimed), random.Random(0)) is None


def test_exploration_still_happens_at_maximum_heat():
    """Guard #3: one area pinned at max heat must not stop other areas being picked."""
    candidates = [f"hot/{i}.java" for i in range(20)] + ["cold/only.java"]
    board = BoardSnapshot(hot_intensity_by_dir={"hot": 1.0, "cold": 0.0})
    rng = random.Random(1)
    picks = {choose_next_file(candidates, board, rng, epsilon=0.15) for _ in range(500)}
    assert "cold/only.java" in picks


def test_with_epsilon_zero_the_top_scored_file_always_wins():
    candidates = ["cold/a.java", "hot/b.java"]
    board = BoardSnapshot(hot_intensity_by_dir={"hot": 1.0, "cold": 0.0})
    rng = random.Random(0)
    for _ in range(20):
        assert choose_next_file(candidates, board, rng, epsilon=0.0) == "hot/b.java"


def test_suggest_next_files_excludes_claimed_and_respects_k():
    candidates = [f"a/{i}.java" for i in range(10)]
    board = BoardSnapshot(claimed=frozenset({"a/0.java"}))
    rng = random.Random(0)
    picks = suggest_next_files(candidates, board, rng, epsilon=0.0, k=3)
    assert len(picks) == 3
    assert "a/0.java" not in picks


def test_suggest_next_files_empty_when_all_claimed():
    board = BoardSnapshot(claimed=frozenset({"a/0.java"}))
    assert suggest_next_files(["a/0.java"], board, random.Random(0)) == []
