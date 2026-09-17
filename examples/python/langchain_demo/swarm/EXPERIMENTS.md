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

### 2026-09-18 — Amaze File Manager, fileoperations/ (18 files), validation attempt 1

- **Variables**: 4 scouts (2x glm-5.3-flash, 2x qwen3-30b-a3b), bloodhound
  gemini-2.5-flash, judge claude-haiku-4.5, corpus = `fileoperations/` (18 files),
  budget 600k tokens, recursion_limit=22, grep_file/read_lines added, revive task
  enabled.
- **Outcome**: killed manually -- multiple scouts (not model-specific: glm and qwen
  both affected) got permanently stuck in an "already hold a claim" refusal loop
  they could never escape, repeatedly hitting recursion_limit on every revival.
- **Root cause -- a real bug, not a model-capability issue**: `mark_covered(path)`
  trusted whatever `path` argument the model supplied and released THAT pheromone,
  never verifying it matched the file actually held. After a crash (which wipes the
  model's own turn memory) or simple confusion, a model would call
  `mark_covered` with the WRONG path -- e.g. holding a claim on
  `filesystem/cloud/CloudStreamServer.java` but calling
  `mark_covered("StreamNotFoundException.java")`. Since pheromone matching is by
  exact trail+type+payload, this created an unrelated zero-intensity pheromone and
  left the REAL claim live forever. Every subsequent claim_file call for that scout
  was refused, and the refusal message didn't say which file was actually held, so
  the model had no way to self-correct.
- **Fix**: `mark_visited`, `mark_covered`, and `report_evidence` no longer take a
  `path` argument at all -- they look up whichever file the scout's own board claim
  currently names (`_my_claimed_file`) and act on that. Only `claim_file` takes a
  path now. This removes the whole class of path-mismatch bugs rather than patching
  one instance of it.
- **What worked**: the revive task DID keep re-triggering the stuck scouts every 8s
  as designed -- confirms that mechanism functions. It just kept re-triggering into
  the same permanent jam until the underlying bug was fixed.
- **VERDICT**: failed -- do not reuse mark_covered/report_evidence with a
  model-supplied path argument. Re-test with the no-argument versions.

### 2026-09-18 — Amaze File Manager, fileoperations/ (18 files), validation attempt 2

- **Variables**: same as attempt 1, with the claim-consistency fix applied.
- **Outcome**: ended_by=budget at 870,067 tokens (600k cap + a little overrun before
  the halt propagated), 119s, all 18 files touched. No crash was permanent -- every
  scout that hit recursion_limit=22 recovered cleanly on its next activation
  (confirmed in the trace: scout-4 got refused with an ACCURATE "you already hold a
  claim on filesystem/StorageNaming.java" message and released it correctly next
  turn, instead of jamming forever).
- **Findings**: scout-1 found two credible, correctly-attributed issues in
  `filesystem/cloud/CloudStreamServer.java` -- an unauthenticated embedded
  NanoHTTPD-style network server, and insecure logging. Not yet promoted to the
  dossier when the budget cut off (bloodhound hadn't caught up), so the judge
  correctly wrote an honest "no verified findings" report rather than overclaiming.
- **Token economics**: ~48k tokens/file for the scan phase, vs. ~180k/file in the
  broken attempt -- roughly a 4x improvement from the claim-consistency fix +
  grep_file/read_lines + tighter recursion_limit together. Full 521-file corpus
  scan-only at this rate would be ~25M tokens -- large but no longer absurd.
- **What worked**: recovery, claim correctness, evidence attribution, honest
  incomplete reporting under budget pressure.
- **What didn't**: 600k tokens covers the scan but not enough headroom left for
  bloodhound to confirm + judge to synthesize. Needs a larger budget, not a
  mechanism fix.
- **VERDICT**: worked -- mechanism is sound. Re-run same corpus with a larger budget
  to get a full confirm-and-report cycle before scaling to the comparison matrix.

### 2026-09-18 — Amaze File Manager, fileoperations/ (18 files), 2.5M budget

- **Variables**: same 4-scout mix, budget raised 600k -> 2.5M, timeout 900s.
- **Outcome**: STILL ended_by=budget, 2,531,482 tokens, 380s. 15/18 distinct files
  claimed (genuine progress, not a re-processing loop -- verified by extracting
  distinct claim_file paths from the trace). ~28+ recoverable crashes over the run.
  Judge again wrote an honest "no verified findings" report.
- **Root cause -- the real bottleneck, not budget**: `report_evidence`'s kind
  argument is free text. Across the two fileoperations runs, the SAME real
  finding (an unauthenticated embedded HTTP server) got reported as
  "unauthenticated_network_server" in one run and "unauthenticated_network_service"
  in the other -- different scouts, different sessions, both plausible labels for
  the same thing. Bloodhound's confirmation requires an EXACT string match across
  two independent files' evidence. Label drift silently makes confirmation
  unreachable regardless of how much budget or coverage the swarm gets. 4x the
  budget didn't fix this because it was never the bottleneck.
- **Fix**: `report_evidence` now enforces a fixed 10-value vocabulary
  (EVIDENCE_KINDS in roles.py) -- any other string is rejected with the exact
  allowed list, not silently accepted. Also fixed the revive task blindly re-poking
  scouts forever even after all files were covered (it could not distinguish
  "stalled" from "legitimately done") -- it now checks real coverage first via
  `_all_files_covered` and stops reviving once nothing is left to do.
- **VERDICT**: failed to reach a confirm-and-report cycle, but correctly diagnosed
  why -- not a budget problem. Re-test with the vocabulary fix before spending more.

### 2026-09-18 — Amaze File Manager, fileoperations/ (18 files), vocabulary fix live

- **Variables**: same 4-scout mix, 2.5M budget, with the evidence vocabulary fix.
- **Outcome**: STILL ended_by=budget, 2,669,496 tokens, 380s. Judge again wrote "no
  verified findings."
- **Root cause -- the actual structural bug, found by reading the full trace**: the
  vocabulary fix worked (both an unauthenticated-server report on
  `filesystem/smbstreamer/StreamServer.java` AND a second, independently-labeled
  match on `filesystem/cloud/CloudStreamServer.java` used the EXACT SAME kind,
  `unauthenticated_network_service`) -- real, independent confirmation genuinely
  existed on the board. But bloodhound never saw it: its `listens_for` was
  `threshold(TRAIL_EVIDENCE, "*", ">=", 0.5)` with edge_rising and no hysteresis.
  Once the FIRST piece of evidence ever appears, the trail-wide aggregate stays
  above 0.5 continuously (evidence keeps arriving faster than any one report
  decays), so the aggregate never dips back down for edge_rising to see a SECOND
  edge. Bloodhound woke exactly once, saw 1 report, correctly declined, and then
  slept for the rest of the run no matter how much later, independently-confirming
  evidence arrived. This is why raising the budget never helped in any prior
  attempt -- bloodhound was asleep, not starved.
- **Fix**: gave bloodhound the same self-looping wake pattern already proven to
  work for scouts -- its own decaying wake pheromone, a `keep_watching` tool it
  calls once at the end of every activation, and `listens_for` changed to
  `or_(evidence-edge, own-wake-edge)` so the first evidence event still wakes it
  immediately, but it now also re-checks the evidence trail periodically
  afterward instead of going permanently dormant.
- **Known follow-on cost**: bloodhound now polls indefinitely for the rest of the
  run (there is no signal telling it "nothing left to discover"), which spends
  some budget on empty re-checks after coverage is done. Not fixed yet -- worth a
  cheap follow-up (e.g. stop re-arming once coverage is complete and evidence has
  been stable for N checks), but not blocking.
- **VERDICT**: failed -- the diagnosis (bloodhound structurally asleep after its
  first wake) was right, but the fix (OR the evidence edge with its own wake edge
  in one composite) was wrong. See the next entry for why and the real fix.

### 2026-09-18 — bloodhound self-wake fix, take 2: separate scent, not OR

- **What was wrong with take 1**: OR'ing the evidence-edge condition with a
  wake-edge condition in one composite doesn't work either. Edge tracking
  (`last_condition_met`) is per SCENT, evaluated on the whole composite's met/
  not-met -- not per branch. Once evidence exists, it stays present long enough
  that the OR's overall value is continuously true, which means the wake branch's
  own rise-and-fall is invisible to edge detection: the composite was already
  "true" before the wake pheromone even rose, so there is no edge to see. Verified
  live: bloodhound called keep_watching correctly, its own wake pheromone was
  created, and it still never triggered again.
- **Real fix**: bloodhound's `listens_for` stays exactly as originally designed
  (evidence-edge only, unchanged) for its first wake. It now also calls
  `sbp_register_scent` to register a SEPARATE, independently-tracked dynamic scent
  watching its own wake trail (the same mechanism scouts already use for dynamic
  questions) -- a genuinely different scent_id has its own last_condition_met, so
  its edges are never masked by the construction-time scent's state. Added
  `register_scent` to BLOODHOUND_SBP_OPS; the prompt now instructs registering (or
  idempotently refreshing) the self-watch scent, then keep_watching, at the end of
  every activation.
- **Scripted-mode caveat found while fixing this**: scripted mode's bloodhound
  does NOT need this mechanism -- its one evidence-triggered activation already
  fully exercises confirmation, and forcing scripted.py to also call
  register_scent/keep_watching caused its fixed message list to keep cycling back
  and re-marking hot forever (ScriptedChatModel wraps to message 0 once
  exhausted), which stalled the natural end condition. Reverted -- scripted mode's
  script does not need to mirror everything the live prompt instructs.
- **Open risk, not yet hardened**: a real model that keeps re-confirming an
  already-dossier'd kind on every self-watch cycle (instead of skipping it, as
  instructed) could keep `swarm.hot`'s intensity refreshed indefinitely via
  reinforce, which would prevent the natural end condition (`not hot`) from ever
  being satisfied. The prompt says to skip already-inscribed kinds; this is not
  independently enforced by a tool yet.
- **VERDICT**: worked mechanically (verified: scripted mode green, regression
  green) but not yet proven live. Re-test before trusting.
