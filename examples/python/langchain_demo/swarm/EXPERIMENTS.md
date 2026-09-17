# Swarm experiment log

Every live run (`python -m swarm.run --mode live ...`) gets an entry here, including
failed ones. Scripted-mode development runs don't need an entry.

## Template

```
### <date> — <short label>

- **Variables**: scout models, bloodhound model, judge model, seed, budget cap,
  corpus/source.
- **Command**: exact CLI invocation.
- **Outcome**: ended_by, files covered, duration, actual token spend / cost.
- **Findings**: what the swarm reported, checked against InsecureBankv2's known
  ground truth (hardcoded credentials, weak crypto, exported components, insecure
  logging, cleartext traffic, allowBackup).
- **What worked / what didn't**: concrete, one or two sentences each.
- **VERDICT**: worked / failed — do not reuse / unknown.
```

## Runs

### 2026-09-18 — InsecureBankv2, control config (killed before spend)

- **Variables**: 2 scouts (glm-5.3-flash x2), bloodhound/judge glm-5.3-flash, corpus =
  InsecureBankv2's own package (12 files, ~62KB).
- **Outcome**: killed before any tokens spent -- user correctly flagged the target as
  too trivial (12 known files from a deliberately-vulnerable teaching app with a
  public answer key) to be a real measurement.
- **VERDICT**: unknown -- never ran. Superseded by the Amaze File Manager target below.

### 2026-09-18 — Amaze File Manager, 12 scouts, full 521-file package, attempt 1

- **Variables**: 12 scouts (6x glm-5.3-flash, 6x qwen3-30b-a3b), bloodhound
  gemini-2.5-flash, judge claude-haiku-4.5, corpus = full `com/amaze/filemanager`
  (521 files), budget cap 3M tokens, recursion_limit=30 (scout).
- **Outcome**: killed manually -- 7+ of 12 scouts hit GraphRecursionError on their
  FIRST activation, before making any real progress.
- **Root cause**: no per-activation gate. `continue_watching`'s docstring said "call
  this once at the end" but nothing enforced it -- real models chained through
  multiple files in one activation (unlike the scripted model, which only ever
  proposes exactly the scripted sequence) until recursion_limit crashed them.
- **What didn't work**: relying on prompt compliance ("always end your turn by
  calling continue_watching") to bound activation length. Real models do not reliably
  honor this.
- **VERDICT**: failed -- do not reuse this design (no claim gate) at any scale above
  a handful of files.

### 2026-09-18 — Amaze File Manager, 12 scouts, full 521-file package, attempt 2

- **Variables**: same as attempt 1, plus a mechanical claim gate (claim_file refuses
  a second claim while a live claim already exists on the board, checked via
  `_has_live_claim`, not local per-turn state) and recursion_limit raised 30->40 to
  account for the budget middleware's after_model hook adding a second graph node per
  model turn (missed in the original sizing).
- **Outcome**: ended_by=budget at 100s, 3,453,005 tokens spent (over the 3M cap),
  ~19 files touched across the whole fleet. Judge correctly woke on the halt-OR
  branch of end_of_run_condition and wrote an honest "INCOMPLETE, 0 confirmed
  findings" report (bloodhound hadn't gotten independent confirmation on anything
  yet) -- the no-false-fires guarantee held even under time pressure.
- **Root cause of the token blowup**: journal showed only 15 total activations but
  802 sniff calls. Most spend was NOT reading file content -- it was activations
  looping unproductively (repeated suggest_files/sniff calls without settling),
  and each additional tool call in one activation resends the ENTIRE accumulated
  conversation as prompt tokens, so a loop's cost grows close to quadratically.
  At this rate (~180k tokens/file touched), covering the full 521-file corpus once
  would cost on the order of 90M+ tokens -- not viable.
- **What worked**: the claim gate. scout-1's real trace showed clean one-file-per-
  activation cycles across 5 files, including a genuine, credible finding (a Zip
  Slip / path-traversal pattern in `CompressedExplorerAdapter.java`, not yet
  independently confirmed when the budget cut off).
- **What didn't work**: recursion_limit=40 was still enough headroom for an
  unproductive loop to burn huge context before failing. Full corpus coverage in one
  run is not viable at this per-file cost regardless of crash handling.
- **Fixes applied afterward** (not yet run): recursion_limit lowered to 22 (caps
  loop damage cheaply instead of absorbing it), added grep_file/read_lines tools so
  triage doesn't require a full read_file, tightened the scout prompt to call
  suggest_files once and triage cheaply first, added a revive task that re-pokes any
  scout with 0 active_activations every 8s so a crash is a setback, not a permanent
  loss (live mode only -- it desyncs scripted mode's fixed-length message lists).
- **VERDICT**: partial -- proves the mechanism (evidence independence, honest
  incomplete reporting, real finding) but the token economics don't scale to the
  full package in one run without the follow-up fixes. Re-test before trusting.
