"""
Property-based tests for LocalBlackboard.emit()'s merge strategies.

Generates sequences of emit() calls with random intensities (fixed matching
trail/type/payload so they merge, per SPECIFICATION.md 5.1: "Pheromones MUST
match if trail + type + payload_hash are identical") and checks the invariants
each merge strategy promises, rather than the fixed examples in test_blackboard.py.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sbp.blackboard import LocalBlackboard
from sbp.types import EmitParams, ExponentialDecay

FROZEN_NOW = 1_700_000_000_000

# Merge-matching only considers a pheromone "existing" while it's not evaporated
# (default ttl_floor is 0.01), so merge tests keep intensities safely above that floor.
merge_intensities = st.floats(min_value=0.02, max_value=1, allow_nan=False, allow_infinity=False)
any_intensities = st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)


def _frozen_blackboard() -> LocalBlackboard:
    bb = LocalBlackboard()
    bb._now = lambda: FROZEN_NOW  # pin the clock so emit() sequences see elapsed=0, not real decay
    return bb


def _emit(bb, intensity, merge_strategy, decay=None):
    return bb.emit(
        EmitParams(
            trail="t",
            type="e",
            intensity=intensity,
            decay=decay,
            payload={"k": "v"},
            merge_strategy=merge_strategy,
        )
    )


# --- reinforce: tracks the running max, never decreases ---


@given(first=merge_intensities, second=merge_intensities)
def test_reinforce_never_decreases_intensity(first, second):
    bb = _frozen_blackboard()
    before = _emit(bb, first, "reinforce")
    after = _emit(bb, second, "reinforce")
    assert after.new_intensity >= before.new_intensity
    assert after.new_intensity == max(first, second)


@given(intensities_seq=st.lists(merge_intensities, min_size=1, max_size=8))
def test_reinforce_over_sequence_equals_running_max(intensities_seq):
    bb = _frozen_blackboard()
    running_max = -1.0
    for i in intensities_seq:
        result = _emit(bb, i, "reinforce")
        running_max = max(running_max, i)
        assert result.new_intensity == running_max


# --- reinforce: floors at the decayed current intensity, not the stored initial ---


def test_reinforce_floors_at_decayed_current_intensity():
    bb = _frozen_blackboard()
    first = _emit(bb, 0.9, "reinforce", decay=ExponentialDecay(half_life_ms=10000))
    pheromone = bb.pheromones[first.pheromone_id]
    pheromone.last_reinforced_at = FROZEN_NOW - 7400  # 0.9 * 0.5^0.74 ≈ 0.53 current now
    after = _emit(bb, 0.3, "reinforce")
    assert after.new_intensity == pytest.approx(0.53, abs=0.02)
    assert after.new_intensity > 0.3


# --- add: capped at 1.0, at least as large as either individual intensity ---


@given(first=merge_intensities, second=merge_intensities)
def test_add_capped_at_one_and_at_least_either_operand(first, second):
    bb = _frozen_blackboard()
    _emit(bb, first, "add")
    after = _emit(bb, second, "add")
    assert after.new_intensity <= 1.0
    assert after.new_intensity >= first
    assert after.new_intensity >= second
    assert after.new_intensity == min(1.0, first + second)


@given(intensities_seq=st.lists(merge_intensities, min_size=1, max_size=8))
def test_add_over_sequence_never_exceeds_one(intensities_seq):
    bb = _frozen_blackboard()
    for i in intensities_seq:
        result = _emit(bb, i, "add")
        assert 0.0 <= result.new_intensity <= 1.0


# --- new: never merges, always creates a distinct pheromone ---


@given(intensities_seq=st.lists(any_intensities, min_size=1, max_size=8))
def test_new_strategy_always_creates_distinct_pheromone(intensities_seq):
    bb = _frozen_blackboard()
    for n, i in enumerate(intensities_seq, start=1):
        result = _emit(bb, i, "new")
        assert result.action == "created"
        assert len(bb.pheromones) == n


# --- reinforce floors at current; replace equals the newly emitted intensity ---


@given(first=merge_intensities, second=merge_intensities)
def test_reinforce_result_equals_max_of_current_and_emitted(first, second):
    bb = _frozen_blackboard()
    _emit(bb, first, "reinforce")
    after = _emit(bb, second, "reinforce")
    assert after.new_intensity == max(first, second)
    assert after.previous_intensity == first


@given(first=merge_intensities, second=merge_intensities)
def test_replace_result_equals_newly_emitted_intensity(first, second):
    bb = _frozen_blackboard()
    _emit(bb, first, "replace")
    after = _emit(bb, second, "replace")
    assert after.new_intensity == second


@given(intensities_seq=st.lists(merge_intensities, min_size=2, max_size=8))
@settings(max_examples=50)
def test_reinforce_and_replace_never_blend_across_sequence(intensities_seq):
    reinforce_bb = _frozen_blackboard()
    replace_bb = _frozen_blackboard()
    running_max = -1.0
    for i in intensities_seq:
        r_result = _emit(reinforce_bb, i, "reinforce")
        p_result = _emit(replace_bb, i, "replace")
        running_max = max(running_max, i)
        assert r_result.new_intensity == running_max
        assert p_result.new_intensity == i
    assert len(reinforce_bb.pheromones) == 1
    assert len(replace_bb.pheromones) == 1


# --- merging strategies keep a single pheromone for a matching trail/type/payload ---


@given(
    strategy=st.sampled_from(["reinforce", "replace", "add"]),
    intensities_seq=st.lists(merge_intensities, min_size=1, max_size=8),
)
def test_merging_strategies_keep_single_pheromone(strategy, intensities_seq):
    bb = _frozen_blackboard()
    for i in intensities_seq:
        _emit(bb, i, strategy)
    assert len(bb.pheromones) == 1
