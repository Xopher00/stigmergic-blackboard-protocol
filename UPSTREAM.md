# Upstream relationship

This repo is a fork of [`AdviceNXT/sbp`](https://github.com/AdviceNXT/sbp), dormant since
April 2026. This fork continues regardless of upstream's activity; this file tracks the delta
and the small set of fixes sent back upstream.

## Sent upstream

Four fixes, each ported to upstream's current `main` as a standalone PR with its own test
(verified to fail against the unfixed code, pass against the fix):

| Fix | PR |
|---|---|
| Step decay could make intensity rise over time instead of decay | [#2](https://github.com/AdviceNXT/sbp/pull/2) |
| `hysteresis` field was accepted but never wired into trigger evaluation | [#3](https://github.com/AdviceNXT/sbp/pull/3) |
| Sequential trigger dispatch — Python local blackboard | [#4](https://github.com/AdviceNXT/sbp/pull/4) |
| Sequential trigger dispatch — TypeScript server | [#5](https://github.com/AdviceNXT/sbp/pull/5) |

Plus [issue #6](https://github.com/AdviceNXT/sbp/issues/6) proposing a spec amendment
(SPECIFICATION.md §7.1) making concurrent trigger dispatch an explicit MUST, since the spec's
silence on it is plausibly why both implementations independently got it wrong.

Filed 2026-09-16. Per plan: wait 3-4 weeks; no response means this fork is the
implementation going forward, no further upstream action needed.

## Not sent upstream (fork-only)

Everything else in this fork is deliberately fork-only — either too large/opinionated to be a
small upstream PR, or specific to a security-hardening direction upstream never asked for:

- **Trigger storm protection** — cap one in-flight activation per trigger, default edge-triggering
  with a minimum re-fire interval, an activation timeout.
- **Zombie worker fix** — `stop()` now deregisters every scent an agent registered, not just the
  ones staged at construction.
- **Ownership and namespacing** — scent IDs auto-prefixed per agent, ownership enforced in the
  blackboard itself (not just client-side), an operator freeze switch.
- **Permission-list closure** — `evaporate(trail=None)` and `inspect` no longer bypass a worker's
  `allowed_trails`.
- **Untrusted-data labeling** — another agent's payload is wrapped and labeled wherever it enters
  an LLM's context.
- **Failure visibility** — structured JSON logging and a reserved-trail stall signal on handler
  exceptions/timeouts.
- **Protocol semantics cleanup** — `reinforce` merge strategy fixed to never weaken a signal,
  duplicate spec section numbers, unified protocol version, removed two declared-but-unimplemented
  condition types.
- **Operation journal + `sbp-replay`** — an append-only audit log (both languages) and an offline
  CLI dashboard over it.
- **Python parity work** — pheromone GC, rate limiting, an injectable clock, and a real
  start()/ready signal for agents, bringing local mode up to par with the TS server.
- **Differential test suite** — proves Python and TypeScript compute identical decay, merge, and
  trigger-evaluation results.
- **README rewrite and a deterministic, API-key-free rogue-agent demo** exercising the hardening
  above end to end.

None of this is proposed upstream; it reflects a fork-specific decision to hardened this protocol
for a scenario upstream's own README never addressed (untrusted multi-agent coordination), not a
disagreement with upstream's direction on anything narrower.
