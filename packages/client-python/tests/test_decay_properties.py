"""
Property-based tests for sbp.decay — compute_intensity() and is_evaporated().

Generates a wide range of pheromone/elapsed-time inputs with `hypothesis` to check
invariants from SPECIFICATION.md section 4.2 ("initial_intensity MUST be 0.0-1.0
normalized") and 4.3 (computeIntensity), rather than the hand-picked examples in
test_blackboard.py.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sbp.decay import compute_intensity, is_evaporated
from sbp.types import ExponentialDecay, ImmortalDecay, LinearDecay, Pheromone, StepDecay

# Shared strategies
intensities = st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
elapsed_times = st.integers(min_value=0, max_value=10**12)
half_lives = st.integers(min_value=1, max_value=10**9)
linear_rates = st.floats(
    min_value=1e-9, max_value=1.0, allow_nan=False, allow_infinity=False
)
ttl_floors = st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)


def _make_pheromone(intensity: float, decay_model, ttl_floor: float = 0.01) -> Pheromone:
    return Pheromone(
        id="p1",
        trail="t",
        type="e",
        emitted_at=0,
        last_reinforced_at=0,
        initial_intensity=intensity,
        decay_model=decay_model,
        ttl_floor=ttl_floor,
    )


# --- Exponential decay ---


@given(intensity=intensities, half_life_ms=half_lives, elapsed=elapsed_times)
def test_exponential_bounded_between_zero_and_initial(intensity, half_life_ms, elapsed):
    p = _make_pheromone(intensity, ExponentialDecay(half_life_ms=half_life_ms))
    result = compute_intensity(p, elapsed)
    assert 0.0 <= result <= intensity + 1e-12


@given(
    intensity=intensities,
    half_life_ms=half_lives,
    t1=elapsed_times,
    t2=elapsed_times,
)
def test_exponential_monotonically_non_increasing(intensity, half_life_ms, t1, t2):
    lo, hi = sorted((t1, t2))
    p = _make_pheromone(intensity, ExponentialDecay(half_life_ms=half_life_ms))
    i_lo = compute_intensity(p, lo)
    i_hi = compute_intensity(p, hi)
    assert i_hi <= i_lo + 1e-12


@given(intensity=intensities, half_life_ms=half_lives)
def test_exponential_at_zero_elapsed_equals_initial(intensity, half_life_ms):
    p = _make_pheromone(intensity, ExponentialDecay(half_life_ms=half_life_ms))
    assert compute_intensity(p, 0) == intensity


@given(intensity=intensities, half_life_ms=half_lives)
def test_exponential_at_exactly_one_half_life_halves(intensity, half_life_ms):
    p = _make_pheromone(intensity, ExponentialDecay(half_life_ms=half_life_ms))
    result = compute_intensity(p, half_life_ms)
    assert result == pytest.approx(intensity / 2, abs=1e-9)


# --- Linear decay ---


@given(intensity=intensities, rate_per_ms=linear_rates, elapsed=elapsed_times)
def test_linear_bounded_between_zero_and_initial(intensity, rate_per_ms, elapsed):
    p = _make_pheromone(intensity, LinearDecay(rate_per_ms=rate_per_ms))
    result = compute_intensity(p, elapsed)
    assert 0.0 <= result <= intensity + 1e-12


@given(
    intensity=intensities,
    rate_per_ms=linear_rates,
    t1=elapsed_times,
    t2=elapsed_times,
)
def test_linear_monotonically_non_increasing(intensity, rate_per_ms, t1, t2):
    lo, hi = sorted((t1, t2))
    p = _make_pheromone(intensity, LinearDecay(rate_per_ms=rate_per_ms))
    i_lo = compute_intensity(p, lo)
    i_hi = compute_intensity(p, hi)
    assert i_hi <= i_lo + 1e-12


@given(intensity=intensities, rate_per_ms=linear_rates)
def test_linear_at_zero_elapsed_equals_initial(intensity, rate_per_ms):
    p = _make_pheromone(intensity, LinearDecay(rate_per_ms=rate_per_ms))
    assert compute_intensity(p, 0) == intensity


# --- Immortal decay ---


@given(intensity=intensities, elapsed=elapsed_times)
def test_immortal_never_decreases(intensity, elapsed):
    p = _make_pheromone(intensity, ImmortalDecay())
    assert compute_intensity(p, elapsed) == intensity


@given(intensity=intensities, t1=elapsed_times, t2=elapsed_times)
def test_immortal_constant_regardless_of_order(intensity, t1, t2):
    p = _make_pheromone(intensity, ImmortalDecay())
    assert compute_intensity(p, t1) == compute_intensity(p, t2) == intensity


# --- Step decay, well-formed steps: sorted by at_ms, intensities in [0, 1],
# non-increasing (a decay curve, per SPECIFICATION.md's "Historical markers" use case) ---


@st.composite
def wellformed_step_lists(draw, initial: float, max_steps=5):
    # Steps capped at `initial` -- a decay curve must not exceed its own starting value.
    n = draw(st.integers(min_value=1, max_value=max_steps))
    at_ms_values = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=10**9),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    step_intensities = st.floats(
        min_value=0, max_value=initial, allow_nan=False, allow_infinity=False
    )
    intensities_desc = sorted(
        draw(st.lists(step_intensities, min_size=n, max_size=n)), reverse=True
    )
    return [
        {"at_ms": float(at), "intensity": float(i)}
        for at, i in zip(at_ms_values, intensities_desc)
    ]


@given(data=st.data())
def test_step_wellformed_bounded(data):
    initial = data.draw(intensities)
    steps = data.draw(wellformed_step_lists(initial))
    elapsed = data.draw(elapsed_times)
    p = _make_pheromone(initial, StepDecay(steps=steps))
    result = compute_intensity(p, elapsed)
    assert 0.0 <= result <= initial + 1e-12


@given(data=st.data())
def test_step_wellformed_monotonically_non_increasing(data):
    initial = data.draw(intensities)
    steps = data.draw(wellformed_step_lists(initial))
    t1 = data.draw(elapsed_times)
    t2 = data.draw(elapsed_times)
    lo, hi = sorted((t1, t2))
    p = _make_pheromone(initial, StepDecay(steps=steps))
    i_lo = compute_intensity(p, lo)
    i_hi = compute_intensity(p, hi)
    assert i_hi <= i_lo + 1e-12


@given(intensity=intensities, elapsed=elapsed_times)
def test_step_empty_steps_returns_initial(intensity, elapsed):
    # reversed([]) yields nothing, falls through to initial_intensity.
    p = _make_pheromone(intensity, StepDecay(steps=[]))
    assert compute_intensity(p, elapsed) == intensity


@given(
    out_of_range_intensity=st.one_of(
        st.floats(min_value=1.0000001, max_value=1e6, allow_nan=False),
        st.floats(min_value=-1e6, max_value=-0.0000001, allow_nan=False),
    ),
    elapsed=st.integers(min_value=0, max_value=10**9),
)
@settings(max_examples=25)
def test_step_intensity_out_of_range_is_clamped(out_of_range_intensity, elapsed):
    steps = [{"at_ms": 0.0, "intensity": out_of_range_intensity}]
    p = _make_pheromone(0.5, StepDecay(steps=steps))
    result = compute_intensity(p, elapsed)
    assert 0.0 <= result <= 1.0


@given(
    initial=intensities,
    step_intensity=intensities,
    elapsed=st.integers(min_value=1, max_value=10**9),
)
@settings(max_examples=50)
def test_step_value_bounded_by_initial_intensity(initial, step_intensity, elapsed):
    steps = [{"at_ms": 0.0, "intensity": step_intensity}]
    p = _make_pheromone(initial, StepDecay(steps=steps))
    result = compute_intensity(p, elapsed)
    assert result <= initial + 1e-12


# --- is_evaporated ---


@given(intensity=intensities, half_life_ms=half_lives, elapsed=elapsed_times, ttl_floor=ttl_floors)
def test_is_evaporated_consistent_with_ttl_floor(intensity, half_life_ms, elapsed, ttl_floor):
    p = _make_pheromone(intensity, ExponentialDecay(half_life_ms=half_life_ms), ttl_floor=ttl_floor)
    assert is_evaporated(p, elapsed) == (compute_intensity(p, elapsed) < ttl_floor)


@given(intensity=intensities, half_life_ms=half_lives, ttl_floor=ttl_floors)
def test_is_evaporated_at_zero_elapsed(intensity, half_life_ms, ttl_floor):
    p = _make_pheromone(intensity, ExponentialDecay(half_life_ms=half_life_ms), ttl_floor=ttl_floor)
    assert is_evaporated(p, 0) == (intensity < ttl_floor)


@given(intensity=intensities, elapsed=elapsed_times, ttl_floor=ttl_floors)
def test_immortal_evaporates_only_if_initial_below_floor(intensity, elapsed, ttl_floor):
    p = _make_pheromone(intensity, ImmortalDecay(), ttl_floor=ttl_floor)
    assert is_evaporated(p, elapsed) == (intensity < ttl_floor)
