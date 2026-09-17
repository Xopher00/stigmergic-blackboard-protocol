Read TASKS.md in the repo root, then continue Phase 6 (the foraging swarm experiment,
`examples/python/langchain_demo/swarm/`).

Scripted mode is implemented and green: `python -m swarm.run --mode scripted` from
`examples/python/langchain_demo`, plus the guard test suite
(`tests/test_swarm_guards.py`, `test_swarm_scoring.py`, `test_swarm_dynamic_scent.py`).
Live mode is implemented (`--mode live`) but has not yet been run against a real
corpus or real models.

Rules:
- Read `examples/python/langchain_demo/swarm/EXPERIMENTS.md` before running anything
  live — it's the log of what's been tried and what it cost. Add an entry for every
  run, including failed ones.
- Run `setup_corpus.py` once to fetch and decompile InsecureBankv2 before a live run.
- State expected cost before a live run and get it confirmed — this is real API
  spend, not a free/local/reversible action.
- Don't touch `packages/client-python` or `packages/server` for this phase unless a
  real bug in the SDK blocks the experiment (as `sbp_worker.py`'s `_log_step` did) —
  say so explicitly if that happens, don't fix it silently.
