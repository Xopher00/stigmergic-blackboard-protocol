# SBP + LangChain Integration Demo

This directory contains examples of how to integrate Stigmergic Blackboard Protocol (SBP) with LangChain agents.

## 1. Minimal Comparison

See how easy it is to upgrade a standard LangChain agent to an SBP-enabled agent.

*   **`standard_agent.py`**: A vanilla LangChain agent that reports findings to stdout.
*   **`sbp_agent.py`**: The same agent, but "enhanced" to emit findings to the shared blackboard.

**Key Differences:**
1.  **Import**: `from sbp.client import AsyncSbpClient`
2.  **Initialize**: `sbp = AsyncSbpClient()`
3.  **Emit**: Calls `await sbp.emit(...)` inside the tool.

## 2. Multi-Agent Orchestration

*   **`sbp_worker.py`**: `SbpWorker` — the one way to build a new SBP-reactive LangChain
    agent in this repo. Fuses `SbpAgent`'s own registration (`.when()`/`.run()`/`.stop()`)
    with a `create_agent()` agent permissioned to a subset of SBP operations, instead of
    wiring a toolkit and a trigger handler together by hand each time. Its tool coverage
    is a complete mirror of `SbpAgent` itself — every parameter those methods accept is
    reachable, gated per-worker by `sbp_ops`: `emit` (with `tags`/`merge_strategy`,
    free-form `payload`), `sniff` (with `types`/`limit`/`include_evaporated`), `inscribe`
    (with `tags`), `read` (with `keys`/`prefix`/`limit`), `erase` (with `older_than_ms`),
    and three opt-in extras not on by default — `evaporate` (force-remove stale signals
    now instead of waiting for decay), `inspect` (a snapshot of overall blackboard
    state), and `register_scent`/`deregister_scent` (start/stop watching a *new*
    condition live while already running, not just what was declared at construction).
*   **`complex_demo.py`**: A fully scent-driven architecture — neither agent is ever
    called directly:
    1.  A **Researcher** (`SbpWorker`) wakes on `research/requested`, finds a fact, and
        emits it to `science.space`.
    2.  A **Writer** (`SbpWorker`) wakes on `science.space/finding`, writes a tweet, and
        emits it to `social.twitter`.
    3.  The script's only non-reactive step is one bootstrap `emit` that kicks things
        off — everything downstream is agents reacting to what's on the blackboard.

**Local Mode:**
This demo uses `local=True` which runs an **in-memory blackboard**. You do **NOT** need to run the SBP server separately. This makes it perfect for testing and single-process simulations.

### Adding a new agent

Creating a new SBP-reactive agent is one `SbpWorker(...)` call — nothing to rewrite from
scratch:

1.  Pick an `agent_id`, write its `system_prompt`.
2.  Decide `listens_for` — either the simple dict shortcut (single
    trail/signal_type/threshold), or a full `sbp.types` `ScentCondition` (e.g.
    `CompositeCondition` to wake only when multiple independent signals hold at once —
    see `examples/python/sbp_reference.py`'s crisis-detector) when one condition isn't
    enough.
3.  Decide the *minimal* `sbp_ops`/`allowed_trails` it actually needs — not everything
    "just in case"; an unused tool with no clear purpose is a real way to confuse an
    agent into looping. Default is `emit`/`sniff`/`inscribe`/`read`; add `erase`,
    `evaporate`, `inspect`, `register_scent`/`deregister_scent` only when the agent
    genuinely needs them.
4.  Add any non-SBP `tools` it needs (file I/O, web search, etc. — plain LangChain
    tools; `SbpWorker` merges them with its SBP tools into one list, same as
    `create_agent(tools=...)` already expects). Optionally set `default_decay` if this
    worker's emissions shouldn't use the default 5-minute exponential decay.
5.  Only if it needs to remember past activations, set `checkpointer` — context-length
    trimming is then automatic, no extra step.
6.  Construct it, add its `.run()` to the fleet's background tasks and `.stop()` to
    teardown.

See `.claude/skills/sbp-worker/SKILL.md` for the same checklist aimed at a coding agent,
and `complex_demo.py` for a worked example.

## Setup

1.  **Install Python Dependencies** — two separate sets, since `standard_agent.py`/
    `sbp_agent.py`'s API is incompatible with `sbp_worker.py`/`complex_demo.py`'s:
    ```bash
    # For sbp_worker.py / complex_demo.py:
    pip install -r requirements.txt
    # For standard_agent.py / sbp_agent.py instead:
    pip install -r requirements-legacy.txt
    ```

2.  **Install SBP Client**:
    ```bash
    pip install -e ../../../packages/client-python
    ```

3.  **Set API Key**:
    ```bash
    cp .env.example .env
    # sbp_worker.py/complex_demo.py read OPENROUTER_API_KEY/OPENROUTER_MODEL;
    # standard_agent.py/sbp_agent.py read OPENAI_API_KEY instead — set whichever you need.
    ```

## Running

```bash
# Run standard agent (older API, OPENAI_API_KEY — not yet migrated)
python standard_agent.py

# Run SBP-enabled agent (older API, OPENAI_API_KEY — not yet migrated)
python sbp_agent.py

# Run full multi-agent demo (SbpWorker, OpenRouter, no server required!)
python complex_demo.py
```

`standard_agent.py`/`sbp_agent.py` intentionally still use the deprecated
`create_openai_functions_agent`/`AgentExecutor` API and `OPENAI_API_KEY` — they're a
narrower before/after comparison (one tool, one agent) and haven't been migrated to
`SbpWorker`/OpenRouter yet.
