# Differential fixtures

Shared JSON fixtures asserting Python and TypeScript compute identical outputs for decay math,
merge strategies, and threshold-condition evaluation. Consumed by
`packages/server/src/differential.test.ts` (vitest) and
`packages/client-python/tests/test_differential.py` (pytest) — each loads these files directly and
computes outputs using its own language's real `decay`/`emit`/`conditions` code, asserting against
the fixture's `expected_*` fields with an absolute tolerance of `1e-9`.

- `fixtures/decay.json` — one entry per `(decay_model, initial_intensity, elapsed_ms) ->
  expected_intensity`.
- `fixtures/merge.json` — one entry per `(existing_intensity, existing_decay,
  elapsed_before_merge_ms, emitted_intensity, merge_strategy) -> expected_intensity`.
- `fixtures/trigger.json` — one entry per `(threshold condition, pheromones) -> (expected_met,
  expected_value)`.

Fixtures are hand-written against the formulas in `packages/server/src/decay.ts` /
`packages/server/src/conditions.ts` / the `emit()` merge branch in `packages/server/src/blackboard.ts`
— that TypeScript reference implementation is the spec source of truth. If protocol semantics
change, update the fixtures here (not just one language's test file) so both consumer suites stay
in sync.
