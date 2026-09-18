Read TASKS.md in the repo root, then continue Phase 6 (the foraging swarm experiment,
`examples/python/langchain_demo/swarm/`).

Both modes work end to end. Scripted: `python -m swarm.run --mode scripted` from
`examples/python/langchain_demo`, plus the guard test suite
(`tests/test_swarm_guards.py`, `test_swarm_scoring.py`, `test_swarm_dynamic_scent.py`).
Live: `python -m swarm.run --mode live --apk <path-or-url> ...` -- point it at any
.apk and it decompiles with jadx and auto-discovers the app's own package from its
manifest, no hand-picked path required (`swarm/decompile.py`). Four live runs against
Amaze File Manager's `fileoperations/` (18 files) all independently confirmed the same
real vulnerability; see `swarm/EXPERIMENTS.md` for the full comparison and every
earlier failed attempt with its root cause.

Rules:
- Read `examples/python/langchain_demo/swarm/EXPERIMENTS.md` before running anything
  live — it's the log of what's been tried and what it cost. Add an entry for every
  run, including failed ones.
- State expected cost before a live run and get it confirmed — this is real API
  spend, not a free/local/reversible action.
- Don't touch `packages/client-python` or `packages/server` for this phase unless a
  real bug in the SDK blocks the experiment (as `sbp_worker.py`'s `_log_step` did) —
  say so explicitly if that happens, don't fix it silently.

Open item: only 18 of 521 files in Amaze File Manager's own package have been
covered (`fileoperations/` only, chosen to keep debugging cost down). The pipeline is
now proven reliable across 4 configurations -- scaling to the full package (or a
larger real subsystem) is the natural next step, but size it and state the cost
before running rather than assuming the same budget shape holds at 25-30x the file
count.
