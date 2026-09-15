---
name: sbp-worker
description: Add a new SBP-reactive LangChain agent to examples/python/langchain_demo/. Use when asked to create, add, or register a new agent that reacts to the SBP blackboard (emit/sniff/inscribe/read/erase/evaporate/inspect/register_scent/deregister_scent) in this repo — points at the one existing primitive (SbpWorker) instead of re-deriving or duplicating it.
---

# Adding an SBP agent

There is exactly one way to create a new SBP-reactive LangChain agent in this repo:
`SbpWorker`, in `examples/python/langchain_demo/sbp_worker.py`. Do not build a separate
toolkit, a manual `SbpAgent.when()` wiring, or a parallel directory — that was tried
once in this project's history and scrapped for exactly that reason.

Read `sbp_worker.py` and `examples/python/langchain_demo/complex_demo.py` (a worked
example: researcher/writer agents) before writing anything.

## Checklist

1. Pick an `agent_id`, write its `system_prompt`.
2. Decide `listens_for` — the scent condition that wakes it up. Two forms:
   - Simple dict, e.g. `{"trail": "science.space", "signal_type": "finding", "value": 0.5}`
     — single-threshold, routed through `SbpAgent.when()`.
   - A full `sbp.types` `ScentCondition` object (`ThresholdCondition`,
     `CompositeCondition`, `RateCondition`, or `TraceCondition` for a cross-layer
     pheromone+trace trigger) — routed through `SbpAgent.on_scent()` instead. Use this
     when the agent should only wake on multiple independent signals holding at once
     (e.g. `CompositeCondition(operator="and", conditions=[...])`) — this is SBP's own
     "signal accumulation/consensus" mechanism, not something to build separately.
3. Decide the **minimal** `sbp_ops` it actually needs — not everything "just in case."
   Giving an agent a tool it has no instructed reason to use is a confirmed way to make
   it loop/stall (this happened once in this project's history). Default (`sbp_ops`
   unset) is `emit`/`sniff`/`inscribe`/`read`. Opt-in only, when genuinely needed:
   `erase` (destructive), `evaporate` (force-remove stale signals now instead of waiting
   for decay), `inspect` (global blackboard snapshot — not trail-scoped, unlike every
   other op), `register_scent`/`deregister_scent` (start/stop watching a *new* condition
   live while already running — pairs internally with `subscribe`/`unsubscribe`, which
   are not exposed as tools directly, only used by these two). Also set
   `allowed_trails` to scope which trails it may touch.
4. Add any non-SBP `tools` it needs (file I/O, web search, etc.) via the plain `tools=`
   parameter — same list `create_agent(tools=...)` already expects, no separate "extra
   tools" concept. Set `default_decay` (an `sbp.types` `DecayModel`) only if this
   worker's emissions need something other than the default 5-minute exponential decay
   — this is construction-time only, not a per-emission tool argument (a nested typed
   decay-model union is a bad shape for an LLM tool call to construct itself).
5. Only if the agent needs to remember past activations, pass a `checkpointer`
   (e.g. `MemorySaver()`). Context-length trimming is then automatic — do not build or
   pass a separate trimmer middleware.
6. Construct it: `SbpWorker(agent_id, model, system_prompt, listens_for=..., sbp_ops=...)`.
   Add its `.run()` to the fleet's background asyncio tasks, and `.stop()` to teardown.

## Model

`SbpWorker` takes a plain `model: BaseChatModel` instance — it knows nothing about
which provider or key is in use. Build the model at the call site (see `complex_demo.py`'s
`_model()`), reading `OPENROUTER_API_KEY`/`OPENROUTER_MODEL` from `.env` via
`python-dotenv` — this directory's convention (see its README).

## What NOT to do

- Don't build a second `AsyncSbpClient` or a separate tool-wrapping toolkit —
  `SbpWorker` already builds its tools from `SbpAgent`'s own bound methods.
- Don't hand-write `create_agent()` + manual `SbpAgent.when()` wiring — that's exactly
  what `SbpWorker` replaces.
- Don't add a new directory for a new scenario — add new workers to `complex_demo.py`,
  or a new script that imports `SbpWorker` from this same file.
- Don't guess at a `sleep(N)` to wait for a reactive agent to finish — use an
  `asyncio.Event` set by an observer/trigger, as `complex_demo.py` does.
