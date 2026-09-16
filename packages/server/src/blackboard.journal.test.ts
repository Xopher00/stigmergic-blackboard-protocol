// Red by design until `BlackboardOptions.journal` exists — every test reads the journal
// file, so the missing option surfaces as ENOENT; J6 also needs the TS storm-skip counter.

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Blackboard } from "./blackboard.js";

// Alphabetically sorted so Object.keys(line).sort() compares directly.
const JOURNAL_KEYS = [
  "activationId",
  "agent",
  "latencyMs",
  "op",
  "outcome",
  "seq",
  "skippedFires",
  "targetId",
  "trail",
  "ts",
];

interface JournalLine {
  seq: number;
  ts: number;
  agent: string | null;
  op: string;
  trail: string | null;
  targetId: string | null;
  outcome: "ok" | "not_found" | "error";
  latencyMs: number;
  activationId: string | null;
  skippedFires: number | null;
}

function readJournalAt(path: string): JournalLine[] {
  return readFileSync(path, "utf-8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as JournalLine);
}

function expectValidJournalLine(line: JournalLine, expectedSeq: number): void {
  expect(Object.keys(line).sort()).toEqual(JOURNAL_KEYS);
  // seq is assumed 1-based and gapless per journal file.
  expect(Number.isInteger(line.seq)).toBe(true);
  expect(line.seq).toBe(expectedSeq);
  expect(Number.isInteger(line.ts)).toBe(true);
  expect(line.agent === null || typeof line.agent === "string").toBe(true);
  expect(typeof line.op).toBe("string");
  expect(line.trail === null || typeof line.trail === "string").toBe(true);
  expect(line.targetId === null || typeof line.targetId === "string").toBe(true);
  expect(["ok", "not_found", "error"]).toContain(line.outcome);
  expect(typeof line.latencyMs).toBe("number");
  expect(line.latencyMs).toBeGreaterThanOrEqual(0);
  expect(line.activationId === null || typeof line.activationId === "string").toBe(true);
  expect(line.skippedFires === null || typeof line.skippedFires === "number").toBe(true);
}

describe("Operation Journal", () => {
  let dir: string;
  let journalPath: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "sbp-journal-"));
    journalPath = join(dir, "journal.jsonl");
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  // Same fire-and-forget dispatch settle used by the evaluateScents tests.
  const settle = () => new Promise((resolve) => setTimeout(resolve, 10));

  const journalBoard = () =>
    new Blackboard({ trackEmissionHistory: false, journal: { path: journalPath } });

  const threshold = (trail: string, signal_type = "signal") => ({
    type: "threshold" as const,
    trail,
    signal_type,
    aggregation: "sum" as const,
    operator: ">=" as const,
    value: 1,
  });

  it("J1: emit(created) logs op/trail/targetId/agent/outcome/latencyMs", () => {
    const bb = journalBoard();
    const result = bb.emit({
      trail: "test.signals",
      type: "event",
      intensity: 0.8,
      source_agent: "agent-a",
    });

    const lines = readJournalAt(journalPath);
    expect(lines).toHaveLength(1);
    expect(lines[0].op).toBe("emit");
    expect(lines[0].trail).toBe("test.signals");
    expect(lines[0].targetId).toBe(result.pheromone_id);
    expect(lines[0].agent).toBe("agent-a");
    expect(lines[0].outcome).toBe("ok");
    expect(typeof lines[0].latencyMs).toBe("number");
    expect(lines[0].latencyMs).toBeGreaterThanOrEqual(0);
  });

  it("J2: emit(reinforce) targets the existing pheromone id", () => {
    const bb = journalBoard();
    const first = bb.emit({ trail: "test.signals", type: "event", intensity: 0.8 });
    bb.emit({ trail: "test.signals", type: "event", intensity: 0.8 });

    const lines = readJournalAt(journalPath);
    expect(lines).toHaveLength(2);
    expect(lines[1].op).toBe("emit");
    expect(lines[1].targetId).toBe(first.pheromone_id);
  });

  it("J3: sniff logs joined trails and null targetId", () => {
    const bb = journalBoard();
    bb.emit({ trail: "test.a", type: "event", intensity: 0.5 });
    bb.sniff({ trails: ["test.a", "test.b"] });

    const lines = readJournalAt(journalPath);
    expect(lines).toHaveLength(2);
    expect(lines[1].op).toBe("sniff");
    expect(lines[1].trail).toBe("test.a,test.b");
    expect(lines[1].targetId).toBeNull();
    expect(lines[1].outcome).toBe("ok");
  });

  it("J4: registerScent/deregisterScent log verbatim scent_id; unknown deregister is not_found", () => {
    const bb = journalBoard();
    bb.registerScent({
      scent_id: "scent-1",
      agent_endpoint: "http://localhost:8080",
      condition: threshold("test.a"),
    });
    bb.deregisterScent({ scent_id: "scent-1" });
    bb.deregisterScent({ scent_id: "ghost-scent" });

    const lines = readJournalAt(journalPath);
    const registered = lines.find((l) => l.op === "registerScent");
    const deregistered = lines.find(
      (l) => l.op === "deregisterScent" && l.targetId === "scent-1",
    );
    const ghost = lines.find(
      (l) => l.op === "deregisterScent" && l.targetId === "ghost-scent",
    );

    expect(registered?.trail).toBeNull();
    expect(registered?.targetId).toBe("scent-1");
    expect(registered?.outcome).toBe("ok");
    expect(deregistered?.targetId).toBe("scent-1");
    expect(deregistered?.outcome).toBe("ok");
    expect(ghost?.targetId).toBe("ghost-scent");
    expect(ghost?.outcome).toBe("not_found");
  });

  it("J5: fired trigger logs exactly one line with activationId and skippedFires=0", async () => {
    const bb = journalBoard();
    let fired = 0;
    bb.onTrigger("scent-1", async () => {
      fired++;
    });
    bb.registerScent({
      scent_id: "scent-1",
      agent_endpoint: "http://localhost:8080",
      condition: threshold("test.a"),
      cooldown_ms: 0,
      trigger_mode: "level",
    });
    bb.emit({ trail: "test.a", type: "signal", intensity: 5 });

    await bb.evaluateScents();
    await settle();

    expect(fired).toBe(1);
    const lines = readJournalAt(journalPath).filter((l) => l.op === "trigger");
    expect(lines).toHaveLength(1);
    expect(lines[0].targetId).toBe("scent-1");
    expect(lines[0].outcome).toBe("ok");
    expect(lines[0].activationId).toMatch(/^scent-1@\d+$/);
    expect(lines[0].skippedFires).toBe(0);
    expect(typeof lines[0].latencyMs).toBe("number");
  });

  it("J6: skip-fires while a dispatch is already running are counted (skippedFires=1)", async () => {
    const bb = journalBoard();
    bb.onTrigger("scent-1", () => new Promise(() => {})); // never settles
    bb.registerScent({
      scent_id: "scent-1",
      agent_endpoint: "http://localhost:8080",
      condition: threshold("test.a"),
      cooldown_ms: 0,
      trigger_mode: "level",
      max_execution_ms: 100,
    });
    bb.emit({ trail: "test.a", type: "signal", intensity: 5 });

    // First evaluate starts the dispatch; the second must be counted as
    // skipped, not executed, while the handler is still "running".
    await bb.evaluateScents();
    await bb.evaluateScents();
    await settle();

    const lines = readJournalAt(journalPath).filter((l) => l.op === "trigger");
    expect(lines.length).toBeGreaterThan(0);
    expect(lines.some((l) => l.skippedFires === 1)).toBe(true);
  });

  it("J7: evaporate logs the filter trail with null targetId", () => {
    const bb = journalBoard();
    bb.emit({ trail: "test.a", type: "event", intensity: 0.5 });
    bb.evaporate({ trail: "test.a" });

    const lines = readJournalAt(journalPath);
    expect(lines[1].op).toBe("evaporate");
    expect(lines[1].trail).toBe("test.a");
    expect(lines[1].targetId).toBeNull();
    expect(lines[1].outcome).toBe("ok");
  });

  it("J8: inspect logs with null trail and null targetId", () => {
    const bb = journalBoard();
    bb.inspect({ include: ["stats"] });

    const lines = readJournalAt(journalPath);
    expect(lines).toHaveLength(1);
    expect(lines[0].op).toBe("inspect");
    expect(lines[0].trail).toBeNull();
    expect(lines[0].targetId).toBeNull();
    expect(lines[0].outcome).toBe("ok");
  });

  it("J9: inscribe logs trail+key and is never suppressed for repeat keys", () => {
    const bb = journalBoard();
    bb.inscribe({ trail: "test.traces", key: "k1", value: { v: 1 }, source_agent: "agent-a" });
    bb.inscribe({ trail: "test.traces", key: "k1", value: { v: 2 }, source_agent: "agent-a" });

    const lines = readJournalAt(journalPath).filter((l) => l.op === "inscribe");
    expect(lines).toHaveLength(2);
    expect(lines[0].trail).toBe("test.traces");
    expect(lines[0].targetId).toBe("k1");
    expect(lines[0].agent).toBe("agent-a");
    expect(lines[0].outcome).toBe("ok");
    expect(lines[1].targetId).toBe("k1");
    expect(lines[1].seq).not.toBe(lines[0].seq);
  });

  it("J10: read logs ok on hit and not_found (line still appended) on miss", () => {
    const bb = journalBoard();
    bb.inscribe({ trail: "test.traces", key: "k1", value: { v: 1 } });
    bb.read({ trails: ["test.traces"], keys: ["k1"] });
    bb.read({ keys: ["missing"] });

    const lines = readJournalAt(journalPath).filter((l) => l.op === "read");
    expect(lines).toHaveLength(2);
    expect(lines[0].outcome).toBe("ok");
    expect(lines[0].targetId).toBe("k1");
    expect(lines[1].outcome).toBe("not_found");
    expect(lines[1].targetId).toBe("missing");
  });

  it("J11: erase appends a tombstone line and the file stays append-only", () => {
    const bb = journalBoard();
    bb.inscribe({ trail: "test.traces", key: "k1", value: { v: 1 } });
    bb.inscribe({ trail: "test.traces", key: "k2", value: { v: 2 } });
    bb.erase({ trail: "test.traces", keys: ["k1"] });

    const afterErase = readJournalAt(journalPath);
    expect(afterErase[0]).toMatchObject({
      op: "inscribe",
      trail: "test.traces",
      targetId: "k1",
      outcome: "ok",
    });
    expect(afterErase[1]).toMatchObject({ op: "inscribe", targetId: "k2" });
    expect(afterErase[2]).toMatchObject({
      op: "erase",
      trail: "test.traces",
      targetId: "k1",
      outcome: "ok",
    });

    // k1 is gone from the store: the read misses and journals not_found...
    const miss = bb.read({ keys: ["k1"] });
    expect(miss.traces).toHaveLength(0);

    // ...but the file still contains every earlier line, in order.
    const file = readJournalAt(journalPath);
    expect(file).toHaveLength(4);
    expect(file[3]).toMatchObject({ op: "read", outcome: "not_found", targetId: "k1" });
    expect(file.some((l) => l.op === "inscribe" && l.targetId === "k1")).toBe(true);
    expect(file.some((l) => l.op === "inscribe" && l.targetId === "k2")).toBe(true);
    expect(file.some((l) => l.op === "erase" && l.targetId === "k1")).toBe(true);
  });

  it("J12: explicit journal path is honored; no journal option means no file", () => {
    const explicitPath = join(dir, "j.jsonl");
    const withJournal = new Blackboard({
      trackEmissionHistory: false,
      journal: { path: explicitPath },
    });
    withJournal.emit({ trail: "test.a", type: "event", intensity: 0.5 });
    withJournal.sniff({ trails: ["test.a"] });
    expect(existsSync(explicitPath)).toBe(true);
    expect(readJournalAt(explicitPath)).toHaveLength(2);

    const withoutJournal = new Blackboard({ trackEmissionHistory: false });
    withoutJournal.emit({ trail: "test.b", type: "event", intensity: 0.5 });
    withoutJournal.sniff({ trails: ["test.b"] });
    // Only the explicit board's journal exists; the default board created no file.
    expect(readdirSync(dir)).toEqual(["j.jsonl"]);
  });

  it("J13: journal ts/latencyMs follow the injected clock, never real timers", () => {
    const FROZEN = 1_720_000_000_000;
    const frozenPath = join(dir, "frozen.jsonl");
    const frozen = new Blackboard({
      trackEmissionHistory: false,
      clock: () => FROZEN,
      journal: { path: frozenPath },
    });
    frozen.emit({ trail: "test.a", type: "event", intensity: 0.5 });
    frozen.sniff({ trails: ["test.a"] });
    frozen.evaporate({ trail: "test.a" });

    const frozenLines = readJournalAt(frozenPath);
    expect(frozenLines).toHaveLength(3);
    for (const line of frozenLines) {
      expect(line.ts).toBe(FROZEN);
      expect(line.latencyMs).toBe(0);
    }

    const counterPath = join(dir, "counter.jsonl");
    let tick = 0;
    const counter = new Blackboard({
      trackEmissionHistory: false,
      clock: () => tick++,
      journal: { path: counterPath },
    });
    counter.emit({ trail: "test.a", type: "event", intensity: 0.5 });
    counter.sniff({ trails: ["test.a"] });
    counter.evaporate({ trail: "test.a" });

    const counterLines = readJournalAt(counterPath);
    expect(counterLines).toHaveLength(3);
    for (let i = 1; i < counterLines.length; i++) {
      expect(counterLines[i].ts).toBeGreaterThanOrEqual(counterLines[i - 1].ts);
    }
    expect(counterLines.some((l) => l.latencyMs > 0)).toBe(true);
  });

  it("J14: every line parses and carries exactly the 10 schema keys with valid values", async () => {
    const bb = journalBoard();
    bb.emit({ trail: "test.a", type: "signal", intensity: 1, source_agent: "agent-a" });
    bb.sniff({ trails: ["test.a"] });
    bb.registerScent({
      scent_id: "scent-1",
      agent_endpoint: "http://localhost:8080",
      condition: threshold("test.a"),
    });
    bb.deregisterScent({ scent_id: "scent-1" });
    bb.deregisterScent({ scent_id: "ghost-scent" }); // not_found
    bb.inscribe({ trail: "test.traces", key: "k1", value: { v: 1 }, source_agent: "agent-a" });
    bb.read({ trails: ["test.traces"], keys: ["k1"] });
    bb.read({ keys: ["missing"] }); // not_found
    bb.erase({ trail: "test.traces", keys: ["k1"] });
    bb.evaporate({ trail: "test.a" });
    bb.inspect({ include: ["stats"] });

    // Drive a trigger too, so trigger lines join the sweep.
    bb.onTrigger("scent-2", async () => {});
    bb.registerScent({
      scent_id: "scent-2",
      agent_endpoint: "http://localhost:8080",
      condition: threshold("test.a"),
      cooldown_ms: 0,
      trigger_mode: "level",
    });
    bb.emit({ trail: "test.a", type: "signal", intensity: 5 });
    await bb.evaluateScents();
    await settle();

    const lines = readJournalAt(journalPath);
    const ops = new Set(lines.map((l) => l.op));
    for (const op of [
      "emit",
      "sniff",
      "registerScent",
      "deregisterScent",
      "trigger",
      "evaporate",
      "inspect",
      "inscribe",
      "read",
      "erase",
    ]) {
      expect(ops.has(op), `no journal line for op "${op}"`).toBe(true);
    }
    lines.forEach((line, i) => expectValidJournalLine(line, i + 1));
  });
});
