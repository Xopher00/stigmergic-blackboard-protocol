/**
 * SBP Blackboard - Core State Management
 */

import { v7 as uuidv7 } from "uuid";
import type { PheromoneStore } from "./store.js";
import { MemoryStore } from "./store.js";
import type { TraceStore } from "./trace-store.js";
import { MemoryTraceStore } from "./trace-store.js";
import {
  type Pheromone,
  type PheromoneSnapshot,
  type Scent,
  type DecayModel,
  type EmitParams,
  type EmitResult,
  type SniffParams,
  type SniffResult,
  type AggregateStats,
  type RegisterScentParams,
  type RegisterScentResult,
  type DeregisterScentParams,
  type DeregisterScentResult,
  type EvaporateParams,
  type EvaporateResult,
  type InspectParams,
  type InspectResult,
  type TriggerPayload,
  type TagFilter,
  type Trace,
  type InscribeParams,
  type InscribeResult,
  type ReadParams,
  type ReadResult,
  type EraseParams,
  type EraseResult,
  type TraceInfo,
  TRACE_MAX_VALUE_SIZE,
  SbpError,
} from "./types.js";
import { computeIntensity, isEvaporated, defaultDecay } from "./decay.js";
import {
  evaluateCondition,
  createSnapshot,
  shouldRearm,
  type EvaluationResult,
} from "./conditions.js";
import { createHash } from "crypto";
import { Journal } from "./journal.js";

export interface BlackboardOptions {
  /** Interval for scent evaluation in ms (default: 100) */
  evaluationInterval?: number;
  /** Default decay model for new pheromones */
  defaultDecay?: DecayModel;
  /** Default TTL floor for evaporation */
  defaultTtlFloor?: number;
  /** Maximum pheromones before GC triggers */
  maxPheromones?: number;
  /** Enable emission history tracking for rate conditions */
  trackEmissionHistory?: boolean;
  /** How long to keep emission history (ms) */
  emissionHistoryWindow?: number;
  /** Pluggable pheromone storage backend (default: MemoryStore) */
  store?: PheromoneStore;
  /** Pluggable trace storage backend (default: MemoryTraceStore) */
  traceStore?: TraceStore;
}

// Clock lives off BlackboardOptions: ServerOptions extends the latter and its
// options map is Required<Omit<...>>, which would force a clock onto every server.
export interface BlackboardRuntimeOptions extends BlackboardOptions {
  /** Injectable clock (epoch ms); default Date.now. Lets tests pin time. */
  clock?: () => number;
  /** Append-only JSONL operation journal; absent → journaling disabled. */
  journal?: { path: string };
}

export interface TriggerHandler {
  (payload: TriggerPayload): Promise<void>;
}

export class Blackboard {
  private store: PheromoneStore;
  private traceStore: TraceStore;
  private scents = new Map<string, Scent>();
  private triggerHandlers = new Map<string, TriggerHandler>();
  private emissionHistory: Array<{ trail: string; type: string; timestamp: number }> = [];
  private evaluationTimer: ReturnType<typeof setInterval> | null = null;
  private startTime: number;
  private pendingDispatches: Set<Promise<void>> = new Set();
  // Scent ids disarmed by hysteresis (§7.4); absence from the set means armed.
  private disarmedScentIds = new Set<string>();

  private readonly clock: () => number;
  private readonly journal: Journal | undefined;
  // §7.1 trigger-storm skip tracking, keyed by scent id (runtime-only; not on Scent).
  private scentRunState = new Map<string, { running: boolean; skippedFires: number }>();

  private options: Omit<
    Required<BlackboardRuntimeOptions>,
    "store" | "traceStore" | "clock" | "journal"
  >;

  constructor(options: BlackboardRuntimeOptions = {}) {
    this.clock = options.clock ?? (() => Date.now());
    this.journal = options.journal ? new Journal(options.journal.path) : undefined;
    // startTime can't be a field initializer: it must read this.clock, which
    // only exists once the constructor body runs.
    this.startTime = this.clock();
    this.store = options.store ?? new MemoryStore();
    this.traceStore = options.traceStore ?? new MemoryTraceStore();
    this.options = {
      evaluationInterval: options.evaluationInterval ?? 100,
      defaultDecay: options.defaultDecay ?? defaultDecay(),
      defaultTtlFloor: options.defaultTtlFloor ?? 0.01,
      maxPheromones: options.maxPheromones ?? 100000,
      trackEmissionHistory: options.trackEmissionHistory ?? true,
      emissionHistoryWindow: options.emissionHistoryWindow ?? 60000,
    };
  }

  // ==========================================================================
  // EMIT
  // ==========================================================================

  emit(params: EmitParams): EmitResult {
    const t0 = this.clock();
    try {
      const result = this.emitOp(params);
      this.journalLine(t0, "emit", params.trail, result.pheromone_id, params.source_agent ?? null, "ok");
      return result;
    } catch (err) {
      this.journalLine(t0, "emit", params.trail, null, params.source_agent ?? null, "error");
      throw err;
    }
  }

  private emitOp(params: EmitParams): EmitResult {
    const now = this.clock();
    const {
      trail,
      type,
      intensity,
      decay = this.options.defaultDecay,
      payload = {},
      tags = [],
      merge_strategy = "reinforce",
      source_agent,
    } = params;

    // Validate intensity
    const clampedIntensity = Math.max(0, Math.min(1, intensity));

    // Track emission for rate conditions
    if (this.options.trackEmissionHistory) {
      this.emissionHistory.push({ trail, type, timestamp: now });
      this.pruneEmissionHistory(now);
    }

    // Generate payload hash for matching
    const payloadHash = this.hashPayload(payload);

    // Find existing pheromone for merge
    let existing: Pheromone | undefined;
    if (merge_strategy !== "new") {
      for (const p of this.store.values()) {
        if (
          p.trail === trail &&
          p.type === type &&
          this.hashPayload(p.payload) === payloadHash &&
          !isEvaporated(p, now)
        ) {
          existing = p;
          break;
        }
      }
    }

    if (existing) {
      const previousIntensity = computeIntensity(existing, now);
      let action: EmitResult["action"] = "reinforced";

      switch (merge_strategy) {
        case "reinforce":
          existing.initial_intensity = Math.max(previousIntensity, clampedIntensity);
          existing.last_reinforced_at = now;
          action = "reinforced";
          break;

        case "replace":
          existing.initial_intensity = clampedIntensity;
          existing.last_reinforced_at = now;
          existing.payload = payload;
          existing.tags = tags;
          if (source_agent) existing.source_agent = source_agent;
          action = "replaced";
          break;

        case "add":
          existing.initial_intensity = Math.min(1, previousIntensity + clampedIntensity);
          existing.last_reinforced_at = now;
          action = "merged";
          break;
      }

      return {
        pheromone_id: existing.id,
        action,
        previous_intensity: previousIntensity,
        new_intensity: computeIntensity(existing, now),
      };
    }

    // Create new pheromone
    const id = uuidv7();
    const pheromone: Pheromone = {
      id,
      trail,
      type,
      emitted_at: now,
      last_reinforced_at: now,
      initial_intensity: clampedIntensity,
      decay_model: decay,
      payload,
      source_agent,
      tags,
      ttl_floor: this.options.defaultTtlFloor,
    };

    this.store.set(id, pheromone);

    // Trigger GC if needed
    if (this.store.size > this.options.maxPheromones) {
      this.gc();
    }

    return {
      pheromone_id: id,
      action: "created",
      new_intensity: clampedIntensity,
    };
  }

  // ==========================================================================
  // SNIFF
  // ==========================================================================

  sniff(params: SniffParams = {}): SniffResult {
    const t0 = this.clock();
    const trail = params.trails?.join(",") ?? null;
    try {
      const result = this.sniffOp(params);
      this.journalLine(t0, "sniff", trail, null, null, "ok");
      return result;
    } catch (err) {
      this.journalLine(t0, "sniff", trail, null, null, "error");
      throw err;
    }
  }

  private sniffOp(params: SniffParams = {}): SniffResult {
    const now = this.clock();
    const {
      trails,
      types,
      min_intensity = 0,
      max_age_ms,
      tags,
      limit = 100,
      include_evaporated = false,
    } = params;

    const results: PheromoneSnapshot[] = [];
    const aggregates = new Map<string, { count: number; sum: number; max: number }>();

    for (const p of this.store.values()) {
      // Filter by trail
      if (trails && trails.length > 0 && !trails.includes(p.trail)) continue;

      // Filter by type
      if (types && types.length > 0 && !types.includes(p.type)) continue;

      const intensity = computeIntensity(p, now);

      // Filter evaporated
      if (!include_evaporated && intensity < p.ttl_floor) continue;

      // Filter by min intensity
      if (intensity < min_intensity) continue;

      // Filter by max age
      if (max_age_ms !== undefined && now - p.emitted_at > max_age_ms) continue;

      // Filter by tags
      if (tags && !this.matchTags(p.tags, tags)) continue;

      results.push(createSnapshot(p, now));

      // Aggregate
      const key = `${p.trail}/${p.type}`;
      const agg = aggregates.get(key) || { count: 0, sum: 0, max: 0 };
      agg.count++;
      agg.sum += intensity;
      agg.max = Math.max(agg.max, intensity);
      aggregates.set(key, agg);
    }

    // Sort by intensity descending
    results.sort((a, b) => b.current_intensity - a.current_intensity);

    // Build aggregates result
    const aggregatesResult: Record<string, AggregateStats> = {};
    for (const [key, agg] of aggregates) {
      aggregatesResult[key] = {
        count: agg.count,
        sum_intensity: agg.sum,
        max_intensity: agg.max,
        avg_intensity: agg.count > 0 ? agg.sum / agg.count : 0,
      };
    }

    return {
      timestamp: now,
      pheromones: results.slice(0, limit),
      aggregates: aggregatesResult,
    };
  }

  // ==========================================================================
  // REGISTER_SCENT
  // ==========================================================================

  registerScent(params: RegisterScentParams): RegisterScentResult {
    const t0 = this.clock();
    try {
      const result = this.registerScentOp(params);
      this.journalLine(t0, "registerScent", null, params.scent_id, null, "ok");
      return result;
    } catch (err) {
      this.journalLine(t0, "registerScent", null, params.scent_id, null, "error");
      throw err;
    }
  }

  private registerScentOp(params: RegisterScentParams): RegisterScentResult {
    const {
      scent_id,
      agent_endpoint,
      condition,
      cooldown_ms = 0,
      activation_payload = {},
      trigger_mode = "level",
      hysteresis = 0,
      max_execution_ms = 30000,
      context_trails,
    } = params;

    const isUpdate = this.scents.has(scent_id);

    const scent: Scent = {
      scent_id,
      agent_endpoint,
      condition,
      cooldown_ms,
      activation_payload,
      trigger_mode,
      hysteresis,
      max_execution_ms,
      last_triggered_at: null,
      last_condition_met: false,
      context_trails,
    };

    this.scents.set(scent_id, scent);
    this.disarmedScentIds.delete(scent_id); // re-registration re-arms

    // Evaluate current state
    const now = this.clock();
    const evalResult = evaluateCondition(condition, {
      pheromones: [...this.store.values()],
      now,
      emissionHistory: this.emissionHistory,
      traces: [...this.traceStore.values()],
    });

    return {
      scent_id,
      status: isUpdate ? "updated" : "registered",
      current_condition_state: {
        met: evalResult.met,
      },
    };
  }

  // ==========================================================================
  // DEREGISTER_SCENT
  // ==========================================================================

  deregisterScent(params: DeregisterScentParams): DeregisterScentResult {
    const t0 = this.clock();
    try {
      const result = this.deregisterScentOp(params);
      this.journalLine(
        t0,
        "deregisterScent",
        null,
        params.scent_id,
        null,
        result.status === "not_found" ? "not_found" : "ok"
      );
      return result;
    } catch (err) {
      this.journalLine(t0, "deregisterScent", null, params.scent_id, null, "error");
      throw err;
    }
  }

  private deregisterScentOp(params: DeregisterScentParams): DeregisterScentResult {
    const { scent_id } = params;

    if (this.scents.has(scent_id)) {
      this.scents.delete(scent_id);
      this.triggerHandlers.delete(scent_id);
      this.disarmedScentIds.delete(scent_id);
      return { scent_id, status: "deregistered" };
    }

    return { scent_id, status: "not_found" };
  }

  // ==========================================================================
  // EVAPORATE
  // ==========================================================================

  evaporate(params: EvaporateParams = {}): EvaporateResult {
    const t0 = this.clock();
    const trail = params.trail ?? null;
    try {
      const result = this.evaporateOp(params);
      this.journalLine(t0, "evaporate", trail, null, null, "ok");
      return result;
    } catch (err) {
      this.journalLine(t0, "evaporate", trail, null, null, "error");
      throw err;
    }
  }

  private evaporateOp(params: EvaporateParams = {}): EvaporateResult {
    const now = this.clock();
    const { trail, types, older_than_ms, below_intensity, tags } = params;

    const toRemove: string[] = [];
    const trailsAffected = new Set<string>();

    for (const [id, p] of this.store.entries()) {
      if (trail && p.trail !== trail) continue;
      if (types && types.length > 0 && !types.includes(p.type)) continue;
      if (older_than_ms !== undefined && now - p.emitted_at < older_than_ms) continue;
      if (below_intensity !== undefined && computeIntensity(p, now) >= below_intensity) continue;
      if (tags && !this.matchTags(p.tags, tags)) continue;

      toRemove.push(id);
      trailsAffected.add(p.trail);
    }

    for (const id of toRemove) {
      this.store.delete(id);
    }

    return {
      evaporated_count: toRemove.length,
      trails_affected: [...trailsAffected],
    };
  }

  // ==========================================================================
  // INSPECT
  // ==========================================================================

  inspect(params: InspectParams = {}): InspectResult {
    const t0 = this.clock();
    try {
      const result = this.inspectOp(params);
      this.journalLine(t0, "inspect", null, null, null, "ok");
      return result;
    } catch (err) {
      this.journalLine(t0, "inspect", null, null, null, "error");
      throw err;
    }
  }

  private inspectOp(params: InspectParams = {}): InspectResult {
    const now = this.clock();
    const include = params.include ?? ["trails", "scents", "stats"];
    const result: InspectResult = { timestamp: now };

    if (include.includes("trails")) {
      const trailMap = new Map<string, { count: number; intensity: number }>();

      for (const p of this.store.values()) {
        if (isEvaporated(p, now)) continue;

        const current = trailMap.get(p.trail) || { count: 0, intensity: 0 };
        current.count++;
        current.intensity += computeIntensity(p, now);
        trailMap.set(p.trail, current);
      }

      result.trails = [...trailMap.entries()].map(([name, data]) => ({
        name,
        pheromone_count: data.count,
        total_intensity: data.intensity,
        avg_intensity: data.count > 0 ? data.intensity / data.count : 0,
      }));
    }

    if (include.includes("scents")) {
      result.scents = [...this.scents.values()].map((s) => ({
        scent_id: s.scent_id,
        agent_endpoint: s.agent_endpoint,
        condition_met: s.last_condition_met,
        in_cooldown: s.last_triggered_at
          ? now - s.last_triggered_at < s.cooldown_ms
          : false,
        last_triggered_at: s.last_triggered_at,
      }));
    }

    if (include.includes("traces")) {
      const traceInfos: TraceInfo[] = [];
      for (const t of this.traceStore.values()) {
        traceInfos.push({
          trail: t.trail,
          key: t.key,
          version: t.version,
          updated_at: t.updated_at,
        });
      }
      result.traces = traceInfos;
    }

    if (include.includes("stats")) {
      let activeCount = 0;
      for (const p of this.store.values()) {
        if (!isEvaporated(p, now)) activeCount++;
      }

      result.stats = {
        total_pheromones: this.store.size,
        active_pheromones: activeCount,
        total_scents: this.scents.size,
        total_traces: this.traceStore.size,
        uptime_ms: now - this.startTime,
      };
    }

    return result;
  }

  // ==========================================================================
  // TRIGGER HANDLING
  // ==========================================================================

  /**
   * Register a local handler for a scent
   */
  onTrigger(scentId: string, handler: TriggerHandler): void {
    this.triggerHandlers.set(scentId, handler);
  }

  /**
   * Remove a local trigger handler
   */
  offTrigger(scentId: string): void {
    this.triggerHandlers.delete(scentId);
  }

  // ==========================================================================
  // EVALUATION LOOP
  // ==========================================================================

  /**
   * Start the background scent evaluation loop
   */
  start(): void {
    if (this.evaluationTimer) return;

    this.evaluationTimer = setInterval(() => {
      this.evaluateScents().catch((err) => {
        console.error("[SBP] Evaluation error:", err);
      });
    }, this.options.evaluationInterval);
  }

  /**
   * Stop the background evaluation loop
   */
  async stop(): Promise<void> {
    if (this.evaluationTimer) {
      clearInterval(this.evaluationTimer);
      this.evaluationTimer = null;
    }

    // Drain in-flight, fire-and-forget trigger dispatches so they don't
    // outlive the Blackboard instance (mirrors the Python client's stop()).
    if (this.pendingDispatches.size > 0) {
      await Promise.allSettled([...this.pendingDispatches]);
    }
  }

  /**
   * Evaluate all scents and trigger as needed
   */
  async evaluateScents(): Promise<void> {
    const now = this.clock();
    const pheromones = [...this.store.values()];

    for (const scent of this.scents.values()) {
      // Check cooldown
      if (scent.last_triggered_at && now - scent.last_triggered_at < scent.cooldown_ms) {
        continue;
      }

      const evalResult = evaluateCondition(scent.condition, {
        pheromones,
        now,
        emissionHistory: this.emissionHistory,
        traces: [...this.traceStore.values()],
      });

      const hysteresis = scent.hysteresis ?? 0;
      const edgeMode =
        scent.trigger_mode === "edge_rising" || scent.trigger_mode === "edge_falling";

      // Hysteresis re-arm (§7.4): a fired scent stays disarmed until the
      // value moves hysteresis beyond the threshold away from the trigger side.
      if (
        hysteresis > 0 &&
        edgeMode &&
        this.disarmedScentIds.has(scent.scent_id) &&
        shouldRearm(scent.condition, evalResult.value, evalResult.met, scent.trigger_mode, hysteresis)
      ) {
        this.disarmedScentIds.delete(scent.scent_id);
      }

      let shouldTrigger = this.shouldTrigger(scent, evalResult);
      scent.last_condition_met = evalResult.met;

      // §7.1: while an activation is in flight, further qualifying evaluations
      // are skipped — counted, not dispatched, and journaled as their own line.
      const runState = this.scentRunState.get(scent.scent_id);
      if (shouldTrigger && runState?.running) {
        runState.skippedFires += 1;
        this.journalLine(
          this.clock(),
          "trigger",
          null,
          scent.scent_id,
          null,
          "ok",
          `${scent.scent_id}@${scent.last_triggered_at ?? now}`,
          runState.skippedFires,
        );
        shouldTrigger = false;
      }

      if (shouldTrigger) {
        scent.last_triggered_at = now;
        if (hysteresis > 0 && edgeMode) {
          this.disarmedScentIds.add(scent.scent_id);
        }
        // Fire-and-forget: a slow/blocked handler for this scent must not
        // delay trigger delivery to other scents in this loop (SPECIFICATION.md §7.1).
        const t0 = this.clock();
        const activationId = `${scent.scent_id}@${now}`;
        const state = runState ?? { running: false, skippedFires: 0 };
        if (!runState) this.scentRunState.set(scent.scent_id, state);
        state.running = true;
        const dispatchPromise = this.dispatchTrigger(scent, evalResult, now).finally(() => {
          this.pendingDispatches.delete(dispatchPromise);
          state.running = false;
          const skipped = state.skippedFires;
          state.skippedFires = 0;
          this.journalLine(t0, "trigger", null, scent.scent_id, null, "ok", activationId, skipped);
        });
        this.pendingDispatches.add(dispatchPromise);
      }
    }
  }

  /**
   * Determine if a scent should trigger based on mode; with hysteresis > 0, edge modes gate on armed (see evaluateScents) instead of last_condition_met
   */
  private shouldTrigger(scent: Scent, evalResult: EvaluationResult): boolean {
    const conditionMet = evalResult.met;
    const hysteresis = scent.hysteresis ?? 0;
    const armed = !this.disarmedScentIds.has(scent.scent_id);

    switch (scent.trigger_mode) {
      case "level":
        return conditionMet;

      case "edge_rising":
        if (hysteresis > 0) return conditionMet && armed;
        return conditionMet && !scent.last_condition_met;

      case "edge_falling":
        if (hysteresis > 0) return !conditionMet && armed;
        return !conditionMet && scent.last_condition_met;

      default:
        return conditionMet;
    }
  }

  /**
   * Dispatch a trigger to the agent
   */
  private async dispatchTrigger(
    scent: Scent,
    evalResult: { met: boolean; value: number; matchingPheromoneIds: string[] },
    now: number
  ): Promise<void> {
    // Build context pheromones
    const contextTrails = scent.context_trails || [];
    const contextPheromones: PheromoneSnapshot[] = [];

    if (contextTrails.length > 0) {
      for (const p of this.store.values()) {
        if (contextTrails.includes(p.trail) && !isEvaporated(p, now)) {
          contextPheromones.push(createSnapshot(p, now));
        }
      }
    } else {
      // Include matching pheromones
      for (const id of evalResult.matchingPheromoneIds) {
        const p = this.store.get(id);
        if (p) {
          contextPheromones.push(createSnapshot(p, now));
        }
      }
    }

    const payload: TriggerPayload = {
      scent_id: scent.scent_id,
      triggered_at: now,
      condition_snapshot: {
        [scent.scent_id]: {
          value: evalResult.value,
          pheromone_ids: evalResult.matchingPheromoneIds,
        },
      },
      context_pheromones: contextPheromones,
      activation_payload: scent.activation_payload,
    };

    // Try local handler first
    const localHandler = this.triggerHandlers.get(scent.scent_id);
    if (localHandler) {
      try {
        await localHandler(payload);
      } catch (err) {
        console.error(`[SBP] Trigger handler error for ${scent.scent_id}:`, err);
      }
      return;
    }

    // Otherwise, HTTP POST to endpoint
    try {
      const response = await fetch(scent.agent_endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-SBP-Version": "0.1",
        },
        body: JSON.stringify({
          jsonrpc: "2.0",
          method: "sbp/trigger",
          params: payload,
        }),
        signal: AbortSignal.timeout(scent.max_execution_ms),
      });

      if (!response.ok) {
        console.error(`[SBP] Trigger failed for ${scent.scent_id}: ${response.status}`);
      }
    } catch (err) {
      console.error(`[SBP] Trigger dispatch error for ${scent.scent_id}:`, err);
    }
  }

  // ==========================================================================
  // UTILITIES
  // ==========================================================================

  /**
   * Garbage collect evaporated pheromones
   */
  gc(): number {
    const now = this.clock();
    const toRemove: string[] = [];

    for (const [id, p] of this.store.entries()) {
      if (isEvaporated(p, now)) {
        toRemove.push(id);
      }
    }

    for (const id of toRemove) {
      this.store.delete(id);
    }

    return toRemove.length;
  }

  /**
   * Get raw pheromone count (for testing)
   */
  get size(): number {
    return this.store.size;
  }

  /**
   * Get scent count
   */
  get scentCount(): number {
    return this.scents.size;
  }

  private hashPayload(payload: Record<string, unknown>): string {
    const content = JSON.stringify(payload, Object.keys(payload).sort());
    return createHash("sha256").update(content).digest("hex").slice(0, 16);
  }

  private matchTags(tags: string[], filter: TagFilter): boolean {
    if (filter.any && filter.any.length > 0) {
      if (!filter.any.some((t) => tags.includes(t))) return false;
    }
    if (filter.all && filter.all.length > 0) {
      if (!filter.all.every((t) => tags.includes(t))) return false;
    }
    if (filter.none && filter.none.length > 0) {
      if (filter.none.some((t) => tags.includes(t))) return false;
    }
    return true;
  }

  private pruneEmissionHistory(now: number): void {
    const cutoff = now - this.options.emissionHistoryWindow;
    this.emissionHistory = this.emissionHistory.filter((e) => e.timestamp >= cutoff);
  }

  private journalLine(
    t0: number,
    op: string,
    trail: string | null,
    targetId: string | null,
    agent: string | null,
    outcome: "ok" | "not_found" | "error",
    activationId: string | null = null,
    skippedFires: number | null = null,
  ): void {
    if (!this.journal) return;
    this.journal.append({
      ts: t0,
      agent,
      op,
      trail,
      targetId,
      outcome,
      latencyMs: this.clock() - t0,
      activationId,
      skippedFires,
    });
  }

  // ==========================================================================
  // TRACE OPERATIONS — Durable Knowledge Layer
  // ==========================================================================

  /**
   * Inscribe a trace — create or update a durable knowledge record.
   * Matching by trail + key. If exists, bumps version.
   */
  inscribe(params: InscribeParams): InscribeResult {
    const t0 = this.clock();
    try {
      const result = this.inscribeOp(params);
      this.journalLine(t0, "inscribe", params.trail, params.key, params.source_agent ?? null, "ok");
      return result;
    } catch (err) {
      this.journalLine(t0, "inscribe", params.trail, params.key, params.source_agent ?? null, "error");
      throw err;
    }
  }

  private inscribeOp(params: InscribeParams): InscribeResult {
    const now = this.clock();
    const { trail, key, value, tags = [], source_agent } = params;

    // Size check
    const serialized = JSON.stringify(value);
    if (serialized.length > TRACE_MAX_VALUE_SIZE) {
      throw new SbpError(
        -32008,
        `Trace value exceeds maximum size (${serialized.length} > ${TRACE_MAX_VALUE_SIZE} bytes)`
      );
    }

    const existing = this.traceStore.getByKey(trail, key);

    if (existing) {
      existing.value = value;
      existing.updated_at = now;
      existing.version += 1;
      existing.tags = tags;
      if (source_agent) existing.source_agent = source_agent;
      this.traceStore.set(existing.id, existing);

      return {
        trace_id: existing.id,
        action: "updated",
        version: existing.version,
      };
    }

    const id = uuidv7();
    const trace: Trace = {
      id,
      trail,
      key,
      value,
      created_at: now,
      updated_at: now,
      version: 1,
      source_agent,
      tags,
    };

    this.traceStore.set(id, trace);

    return {
      trace_id: id,
      action: "created",
      version: 1,
    };
  }

  /**
   * Read traces from the blackboard.
   */
  read(params: ReadParams = {}): ReadResult {
    const t0 = this.clock();
    const trail = params.trails?.join(",") ?? null;
    const targetId = params.keys?.length
      ? params.keys.join(",")
      : params.prefix
        ? `${params.prefix}*`
        : null;
    try {
      const result = this.readOp(params);
      this.journalLine(
        t0,
        "read",
        trail,
        targetId,
        null,
        result.traces.length === 0 ? "not_found" : "ok"
      );
      return result;
    } catch (err) {
      this.journalLine(t0, "read", trail, targetId, null, "error");
      throw err;
    }
  }

  private readOp(params: ReadParams = {}): ReadResult {
    const now = this.clock();
    const { trails, keys, tags, prefix, limit = 100 } = params;

    const results: Trace[] = [];

    for (const t of this.traceStore.values()) {
      // Filter by trail
      if (trails && trails.length > 0 && !trails.includes(t.trail)) continue;

      // Filter by key
      if (keys && keys.length > 0 && !keys.includes(t.key)) continue;

      // Filter by prefix
      if (prefix && !t.key.startsWith(prefix)) continue;

      // Filter by tags
      if (tags && !this.matchTags(t.tags, tags)) continue;

      results.push(t);
    }

    // Sort by updated_at descending (most recent first)
    results.sort((a, b) => b.updated_at - a.updated_at);

    return {
      timestamp: now,
      traces: results.slice(0, limit),
    };
  }

  /**
   * Erase traces matching criteria.
   */
  erase(params: EraseParams = {}): EraseResult {
    const t0 = this.clock();
    const trail = params.trail ?? null;
    const targetId = params.keys?.join(",") ?? null;
    try {
      const result = this.eraseOp(params);
      this.journalLine(t0, "erase", trail, targetId, null, "ok");
      return result;
    } catch (err) {
      this.journalLine(t0, "erase", trail, targetId, null, "error");
      throw err;
    }
  }

  private eraseOp(params: EraseParams = {}): EraseResult {
    const now = this.clock();
    const { trail, keys, tags, older_than_ms } = params;

    const toRemove: string[] = [];
    const trailsAffected = new Set<string>();

    for (const [id, t] of this.traceStore.entries()) {
      if (trail && t.trail !== trail) continue;
      if (keys && keys.length > 0 && !keys.includes(t.key)) continue;
      if (older_than_ms !== undefined && now - t.updated_at < older_than_ms) continue;
      if (tags && !this.matchTags(t.tags, tags)) continue;

      toRemove.push(id);
      trailsAffected.add(t.trail);
    }

    for (const id of toRemove) {
      this.traceStore.delete(id);
    }

    return {
      erased_count: toRemove.length,
      trails_affected: [...trailsAffected],
    };
  }

  /**
   * Get trace count
   */
  get traceCount(): number {
    return this.traceStore.size;
  }
}
