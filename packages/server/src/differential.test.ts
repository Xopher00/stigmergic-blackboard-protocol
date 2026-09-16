/**
 * Differential tests: TS reference implementation vs the shared JSON fixtures
 * in differential/fixtures/ (frozen; also consumed by the Python test suite).
 */

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { computeIntensity } from "./decay.js";
import { evaluateCondition } from "./conditions.js";
import { Blackboard } from "./blackboard.js";
import type { DecayModel, Pheromone, ScentCondition } from "./types.js";

// src/ -> server/ -> packages/ -> repo root
const fixturePath = (name: string) =>
  fileURLToPath(new URL(`../../../differential/fixtures/${name}`, import.meta.url));

const loadFixture = (name: string): any[] =>
  JSON.parse(readFileSync(fixturePath(name), "utf-8"));

// Deterministic take on conformance.test.ts's makePheromone; negative timestamps
// bake each pheromone's own elapsed_ms in, so the context evaluates at now = 0.
function makePheromone(overrides: Partial<Pheromone> = {}): Pheromone {
  return {
    id: `diff-${Math.random().toString(36).slice(2)}`,
    trail: "t",
    type: "e",
    emitted_at: 0,
    last_reinforced_at: 0,
    initial_intensity: 1.0,
    decay_model: { type: "immortal" },
    payload: {},
    tags: [],
    ttl_floor: 0.01,
    ...overrides,
  };
}

// toBeCloseTo(x, 9) tolerates 0.5e-9 — tighter than the fixtures' 1e-9 contract.
const expectClose = (actual: number, expected: number) =>
  expect(actual).toBeCloseTo(expected, 9);

describe("Differential: decay", () => {
  for (const entry of loadFixture("decay.json")) {
    it(entry.name, () => {
      const p = makePheromone({
        initial_intensity: entry.initial_intensity,
        decay_model: entry.decay_model as DecayModel,
      });
      expectClose(computeIntensity(p, entry.elapsed_ms), entry.expected_intensity);
    });
  }
});

describe("Differential: merge strategies", () => {
  for (const entry of loadFixture("merge.json")) {
    it(entry.name, () => {
      let t = 0;
      const bb = new Blackboard({ clock: () => t, trackEmissionHistory: false });
      bb.emit({
        trail: "t",
        type: "e",
        intensity: entry.existing_intensity,
        decay: entry.existing_decay as DecayModel,
        merge_strategy: "new",
      });
      t = entry.elapsed_before_merge_ms;
      const result = bb.emit({
        trail: "t",
        type: "e",
        intensity: entry.emitted_intensity,
        merge_strategy: entry.merge_strategy,
      });
      expectClose(result.new_intensity, entry.expected_intensity);
    });
  }
});

describe("Differential: threshold conditions", () => {
  for (const entry of loadFixture("trigger.json")) {
    it(entry.name, () => {
      const pheromones: Pheromone[] = entry.pheromones.map((fp: any) =>
        makePheromone({
          trail: fp.trail,
          type: fp.type,
          initial_intensity: fp.initial_intensity,
          decay_model: fp.decay_model as DecayModel,
          tags: fp.tags,
          ttl_floor: fp.ttl_floor,
          emitted_at: -fp.elapsed_ms,
          last_reinforced_at: -fp.elapsed_ms,
        }),
      );
      const result = evaluateCondition(entry.condition as ScentCondition, {
        pheromones,
        now: 0,
      });
      expect(result.met).toBe(entry.expected_met);
      expectClose(result.value, entry.expected_value);
    });
  }
});
