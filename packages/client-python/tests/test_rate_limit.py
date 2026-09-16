"""
P3.3 item 2 — local-mode rate limiting (RL1-RL7). Ports the TS server's
token-bucket limiter (packages/server/src/rate-limiter.ts) onto LocalBlackboard.

Contract these tests pin for the implementer:
  - Constructor: LocalBlackboard(rate_limit_max_requests=<int>,
    rate_limit_window_ms=<int>). Flat kwargs, consistent with
    journal_path/max_pheromones. OPT-IN: with no rate-limit kwargs there is no
    limiting at all. The TS server likewise only installs its hook when
    options.rateLimit is provided (server.ts:91); the 1000/60000 figures are
    defaults inside the option, not an always-on behavior.
  - Exception: sbp.rate_limiter.RateLimitError with attributes retry_after_ms,
    limit, window_ms — the TS error's data payload (rate-limiter.ts:91-94,
    code -32004). Custom class, not PermissionError: rate limiting is transient
    congestion, not an ownership/freeze refusal.
  - Algorithm per agent id: bucket starts full (tokens=max_requests); each op
    does tokens = min(max_requests, tokens + elapsed_ms * (max_requests /
    window_ms)); tokens < 1 rejects, else tokens -= 1.
  - One bucket per agent, shared across op types, keyed by the agent id each
    op already carries: emit(source_agent=), register_scent(agent_id=),
    deregister_scent(agent_id=), inscribe(source_agent=). sniff() carries no
    agent identity and is never limited (RL6). register_scent(agent_id=None)
    is the legacy unprefixed path — out of scope here.

Exact retry/refill values below rely on float-exact arithmetic
(ceil((1-0)/(5/60000)) == 12000; 30000 * (10/60000) == 5.0), verified this
session — an implementer seeing off-by-one there should suspect a formula
deviation, not test flakiness.
"""
import pytest

from sbp.blackboard import LocalBlackboard
from sbp.rate_limiter import RateLimitError
from sbp.types import (
    EmitParams, InscribeParams, RegisterScentParams, SniffParams,
    ThresholdCondition,
)

FROZEN_NOW = 1_700_000_000_000


def _limited_bb(max_requests, window_ms=60_000):
    bb = LocalBlackboard(
        rate_limit_max_requests=max_requests, rate_limit_window_ms=window_ms
    )
    clock = {"now": FROZEN_NOW}
    bb._now = lambda: clock["now"]
    return bb, clock


def _emit(bb, agent, trail="t", type="e"):
    return bb.emit(EmitParams(trail=trail, type=type, intensity=0.8, source_agent=agent))


def _register(bb, agent, scent_id="s1"):
    return bb.register_scent(
        RegisterScentParams(
            scent_id=scent_id,
            agent_endpoint="test://a",
            condition=ThresholdCondition(trail="t", signal_type="e", operator=">=", value=0.5),
        ),
        agent_id=agent,
    )


def _inscribe(bb, agent, key="k1"):
    return bb.inscribe(
        InscribeParams(trail="notes", key=key, value={"v": 1}, source_agent=agent)
    )


class TestRateLimit:
    def test_rl1_burst_exceeding_limit_rejects_with_retry_after(self):
        bb, clock = _limited_bb(5)
        for _ in range(5):
            _emit(bb, "agent-1")  # bucket starts full: exactly 5 requests allowed
        with pytest.raises(RateLimitError) as exc:
            _emit(bb, "agent-1")
        err = exc.value
        # tokens == 0 at reject time -> ceil((1 - 0) / (5 / 60000)) == 12000
        assert err.retry_after_ms == 12000
        assert err.limit == 5
        assert err.window_ms == 60_000

    def test_rl2_buckets_are_per_agent(self):
        bb, clock = _limited_bb(5)
        for _ in range(5):
            _emit(bb, "agent-1")
        with pytest.raises(RateLimitError):
            _emit(bb, "agent-1")
        # agent-2 starts with its own full bucket: 5 ok, 6th rejected
        for _ in range(5):
            _emit(bb, "agent-2")
        with pytest.raises(RateLimitError):
            _emit(bb, "agent-2")
        # agent-1's exhausted bucket is untouched by agent-2's activity
        with pytest.raises(RateLimitError):
            _emit(bb, "agent-1")

    def test_rl3_full_window_refills_the_bucket(self):
        bb, clock = _limited_bb(5)
        for _ in range(5):
            _emit(bb, "agent-1")
        clock["now"] += 60_000
        # full window -> refill = 60000 * (5/60000), capped at max_requests = 5:
        # 5 ok, then rejected again (proves full, not partial, refill)
        for _ in range(5):
            _emit(bb, "agent-1")
        with pytest.raises(RateLimitError):
            _emit(bb, "agent-1")

    def test_rl4_partial_refill_after_half_the_window(self):
        bb, clock = _limited_bb(10)
        for _ in range(10):
            _emit(bb, "agent-1")
        clock["now"] += 30_000
        # refill = 30000 * (10 / 60000) = 5.0 tokens exactly: 5 ok, 6th rejected
        for _ in range(5):
            _emit(bb, "agent-1")
        with pytest.raises(RateLimitError):
            _emit(bb, "agent-1")

    def test_rl5_no_rate_limit_kwargs_means_no_limiting(self):
        bb = LocalBlackboard()  # opt-in feature: absent option -> disabled
        clock = {"now": FROZEN_NOW}
        bb._now = lambda: clock["now"]
        for _ in range(50):
            _emit(bb, "agent-1")
        for _ in range(50):
            bb.sniff(SniffParams(trails=["t"]))

    def test_rl6_sniff_has_no_agent_id_so_is_never_limited(self):
        bb, clock = _limited_bb(2)
        for _ in range(2):
            _emit(bb, "agent-1")
        with pytest.raises(RateLimitError):
            _emit(bb, "agent-1")  # sanity: the limiter is active and agent-1 is out
        for _ in range(25):
            bb.sniff(SniffParams(trails=["t"]))

    def test_rl7_bucket_is_shared_across_agent_keyed_ops(self):
        bb, clock = _limited_bb(4)
        _emit(bb, "agent-1")
        _register(bb, "agent-1")
        _inscribe(bb, "agent-1")
        _emit(bb, "agent-1", trail="t2")  # 4th op for agent-1
        # every agent-keyed op draws from the same per-agent bucket, so the next
        # agent-1 op is rejected regardless of its type
        with pytest.raises(RateLimitError):
            bb.deregister_scent("s1", agent_id="agent-1")
        # ...while agent-2 still owns a full bucket for the same op type
        bb.deregister_scent("s1", agent_id="agent-2")
