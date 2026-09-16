"""
SBP Local Blackboard Implementation
"""
import time
import asyncio
import uuid
import hashlib
import json
import os
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set

from sbp.types import (
    Pheromone, PheromoneSnapshot,
    ScentCondition, CompositeCondition, DecayModel, ExponentialDecay,
    EmitParams, EmitResult,
    SniffParams, SniffResult, AggregateStats,
    RegisterScentParams, RegisterScentResult,
    DeregisterScentResult,
    EvaporateParams, EvaporateResult,
    InspectParams, InspectResult, TriggerPayload, TagFilter,
    Trace, InscribeParams, InscribeResult,
    ReadParams, ReadResult,
    EraseParams, EraseResult,
    TRACE_MAX_VALUE_SIZE,
)
from sbp.decay import compute_intensity, is_evaporated
from sbp.evaluator import evaluate_condition, EvaluationContext, match_tags, should_rearm
from sbp.journal import Journal
from sbp.rate_limiter import RateLimiter

class LocalBlackboard:
    def __init__(
        self,
        journal_path: str | os.PathLike[str] | None = None,
        max_pheromones: int | None = None,
        rate_limit_max_requests: int | None = None,
        rate_limit_window_ms: int = 60_000,
    ):
        self.pheromones: Dict[str, Pheromone] = {}
        self.scents: Dict[str, Any] = {} # Storing internal scent dicts
        self.handlers: Dict[str, Callable[[TriggerPayload], Awaitable[None]]] = {}
        self.emission_history: List[Dict[str, Any]] = []
        self.start_time = self._now()

        # Trace storage: composite key "trail\0key" -> Trace
        self.traces: Dict[str, Trace] = {}
        self.traces_by_id: Dict[str, str] = {}  # id -> composite key

        # Options
        self.emission_history_window = 60000
        self.default_ttl_floor = 0.01

        # Background task
        self._running = False
        self._task = None
        self._dispatch_tasks: Set[asyncio.Task] = set()

        # Ownership prefixes whose scents refuse mutation and skip evaluation/dispatch
        self._frozen_prefixes: Set[str] = set()

        self._journal: Journal | None = Journal(str(journal_path)) if journal_path else None
        self._max_pheromones = max_pheromones
        self._rate_limiter = (
            RateLimiter(rate_limit_max_requests, rate_limit_window_ms)
            if rate_limit_max_requests is not None
            else None
        )

    def _trace_key(self, trail: str, key: str) -> str:
        return f"{trail}\0{key}"

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._dispatch_tasks:
            await asyncio.gather(*self._dispatch_tasks, return_exceptions=True)

    async def _loop(self):
        while self._running:
            try:
                await self.evaluate_scents()
            except Exception as e:
                print(f"[SBP Local] Error in loop: {e}")
            await asyncio.sleep(0.1)

    def _now(self) -> int:
        return int(time.time() * 1000)

    def _check_rate_limit(self, agent_id: str | None) -> None:
        # One shared bucket per agent across all agent-keyed ops; calls without
        # an agent identifier are never limited.
        if self._rate_limiter and agent_id:
            self._rate_limiter.check(agent_id, self._now())

    def _journal_line(
        self,
        t0: int,
        op: str,
        trail: str | None,
        target_id: str | None,
        agent: str | None,
        outcome: str,
        activation_id: str | None = None,
        skipped_fires: int | None = None,
    ) -> None:
        if not self._journal:
            return
        self._journal.append({
            "ts": t0, "agent": agent, "op": op, "trail": trail, "targetId": target_id,
            "outcome": outcome, "latencyMs": self._now() - t0,
            "activationId": activation_id, "skippedFires": skipped_fires,
        })

    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        content = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def emit(self, params: EmitParams) -> EmitResult:
        self._check_rate_limit(params.source_agent)
        t0 = self._now()
        try:
            result = self._emit(params)
        except Exception:
            self._journal_line(t0, "emit", params.trail, None, params.source_agent, "error")
            raise
        self._journal_line(
            t0, "emit", params.trail, result.pheromone_id, params.source_agent, "ok"
        )
        return result

    def _emit(self, params: EmitParams) -> EmitResult:
        now = self._now()

        # Record history
        self.emission_history.append({
            "trail": params.trail,
            "type": params.type,
            "timestamp": now
        })
        self._prune_history(now)

        payload_hash = self._hash_payload(params.payload)

        # Find existing
        existing = None
        if params.merge_strategy != "new":
            for p in self.pheromones.values():
                if (p.trail == params.trail and
                    p.type == params.type and
                    self._hash_payload(p.payload) == payload_hash and
                    not is_evaporated(p, now)):
                    existing = p
                    break

        clamped_intensity = max(0.0, min(1.0, params.intensity))

        if existing:
            prev_intensity = compute_intensity(existing, now)
            action = "reinforced"

            if params.merge_strategy == "reinforce":
                existing.initial_intensity = max(prev_intensity, clamped_intensity)
                existing.last_reinforced_at = now
            elif params.merge_strategy == "replace":
                existing.initial_intensity = clamped_intensity
                existing.last_reinforced_at = now
                existing.payload = params.payload
                existing.tags = params.tags
                action = "replaced"
            elif params.merge_strategy == "add":
                existing.initial_intensity = min(1.0, prev_intensity + clamped_intensity)
                existing.last_reinforced_at = now
                action = "merged"

            return EmitResult(
                pheromone_id=existing.id,
                action=action, # type: ignore
                previous_intensity=prev_intensity,
                new_intensity=compute_intensity(existing, now)
            )

        # Create new
        pid = str(uuid.uuid4())
        pheromone = Pheromone(
            id=pid,
            trail=params.trail,
            type=params.type,
            emitted_at=now,
            last_reinforced_at=now,
            initial_intensity=clamped_intensity,
            decay_model=params.decay or {"type": "exponential", "half_life_ms": 300000}, # type: ignore
            payload=params.payload,
            source_agent=params.source_agent,
            tags=params.tags,
            ttl_floor=self.default_ttl_floor
        )
        self.pheromones[pid] = pheromone

        # Checked after insertion so a store already at the limit is swept by the
        # next emit; `>` (not `>=`) matches the TS reference's emit-time GC.
        if self._max_pheromones is not None and len(self.pheromones) > self._max_pheromones:
            self.gc()

        return EmitResult(
            pheromone_id=pid,
            action="created",
            new_intensity=clamped_intensity
        )

    def sniff(self, params: SniffParams) -> SniffResult:
        t0 = self._now()
        trail = ",".join(params.trails) if params.trails else None
        try:
            result = self._sniff(params)
        except Exception:
            self._journal_line(t0, "sniff", trail, None, None, "error")
            raise
        self._journal_line(t0, "sniff", trail, None, None, "ok")
        return result

    def _sniff(self, params: SniffParams) -> SniffResult:
        now = self._now()
        results = []
        aggs: Dict[str, AggregateStats] = {}

        # Temp agg storage: key -> [sum, count, max]
        temp_aggs: Dict[str, List[float]] = {}

        for p in self.pheromones.values():
            if params.trails and p.trail not in params.trails: continue
            if params.types and p.type not in params.types: continue

            intensity = compute_intensity(p, now)

            if not params.include_evaporated and intensity < p.ttl_floor: continue
            if intensity < params.min_intensity: continue
            if params.max_age_ms and (now - p.emitted_at > params.max_age_ms): continue
            if params.tags and not match_tags(p.tags, params.tags): continue

            # Add to results
            results.append(self._create_snapshot(p, now))

            # Aggregate
            key = f"{p.trail}/{p.type}"
            if key not in temp_aggs:
                temp_aggs[key] = [0.0, 0.0, 0.0] # sum, count, max

            temp_aggs[key][0] += intensity
            temp_aggs[key][1] += 1
            temp_aggs[key][2] = max(temp_aggs[key][2], intensity)

        # Sort
        results.sort(key=lambda x: x.current_intensity, reverse=True)

        # Finalize aggs
        for k, v in temp_aggs.items():
            aggs[k] = AggregateStats(
                count=int(v[1]),
                sum_intensity=v[0],
                max_intensity=v[2],
                avg_intensity=v[0]/v[1] if v[1] > 0 else 0
            )

        return SniffResult(
            timestamp=now,
            pheromones=results[:params.limit],
            aggregates=aggs
        )

    def register_scent(
        self, params: RegisterScentParams, agent_id: str | None = None
    ) -> RegisterScentResult:
        self._check_rate_limit(agent_id)
        t0 = self._now()
        try:
            result = self._register_scent(params, agent_id)
        except Exception:
            self._journal_line(t0, "register_scent", None, params.scent_id, agent_id, "error")
            raise
        self._journal_line(t0, "register_scent", None, result.scent_id, agent_id, "ok")
        return result

    def _register_scent(
        self, params: RegisterScentParams, agent_id: str | None = None
    ) -> RegisterScentResult:
        resolved = self._resolve_scent_id(params.scent_id, agent_id)
        if self._scent_frozen(resolved):
            raise PermissionError("refused register_scent: prefix is frozen")

        for t in self._condition_trails(params.condition):
            if t == "system" or t.startswith("system."):
                raise PermissionError(
                    f"refused register_scent: condition watches reserved trail '{t}'; "
                    "system namespace is reserved so stall signals cannot cascade"
                )

        now = self._now()
        is_update = resolved in self.scents

        scent = {
            "id": resolved,
            "condition": params.condition,
            "cooldown_ms": params.cooldown_ms,
            "activation_payload": params.activation_payload,
            "context_trails": params.context_trails,
            "trigger_mode": params.trigger_mode,
            "hysteresis": params.hysteresis,
            "max_execution_ms": params.max_execution_ms,
            "armed": True,
            "last_triggered_at": 0,
            "last_condition_met": False,
            "running": False,
            "skipped_fires": 0
        }
        self.scents[resolved] = scent

        # Evaluate immediately to return state
        ctx = EvaluationContext(
            list(self.pheromones.values()), now, self.emission_history,
            traces=list(self.traces.values())
        )
        result = evaluate_condition(params.condition, ctx)

        return RegisterScentResult(
            scent_id=resolved,
            status="updated" if is_update else "registered",
            current_condition_state={"met": result.met}
        )

    def deregister_scent(
        self, scent_id: str, agent_id: str | None = None
    ) -> DeregisterScentResult:
        self._check_rate_limit(agent_id)
        t0 = self._now()
        try:
            result = self._deregister_scent(scent_id, agent_id)
        except Exception:
            self._journal_line(t0, "deregister_scent", None, scent_id, agent_id, "error")
            raise
        outcome = "ok" if result.status == "deregistered" else "not_found"
        self._journal_line(t0, "deregister_scent", None, result.scent_id, agent_id, outcome)
        return result

    def _deregister_scent(
        self, scent_id: str, agent_id: str | None = None
    ) -> DeregisterScentResult:
        resolved = self._resolve_scent_id(scent_id, agent_id)
        if self._scent_frozen(resolved):
            raise PermissionError("refused deregister_scent: prefix is frozen")
        if resolved in self.scents:
            del self.scents[resolved]
            if resolved in self.handlers:
                del self.handlers[resolved]
            return DeregisterScentResult(scent_id=resolved, status="deregistered")
        return DeregisterScentResult(scent_id=resolved, status="not_found")

    def subscribe(
        self,
        scent_id: str,
        handler: Callable[[TriggerPayload], Awaitable[None]],
        agent_id: str | None = None,
    ) -> None:
        resolved = self._resolve_scent_id(scent_id, agent_id)
        if self._scent_frozen(resolved):
            raise PermissionError("refused subscribe: prefix is frozen")
        if resolved in self.handlers:
            raise ValueError(
                f"scent '{resolved}' is already subscribed; "
                f"unsubscribe('{resolved}') first to replace it"
            )
        self.handlers[resolved] = handler

    def unsubscribe(self, scent_id: str, agent_id: str | None = None) -> None:
        resolved = self._resolve_scent_id(scent_id, agent_id)
        if self._scent_frozen(resolved):
            raise PermissionError("refused unsubscribe: prefix is frozen")
        if resolved in self.handlers:
            del self.handlers[resolved]

    def _resolve_scent_id(self, scent_id: str, agent_id: str | None) -> str:
        # agent_id=None is the legacy path: no prefixing, no ownership checks
        if agent_id is None:
            return scent_id
        if ":" not in scent_id:
            return f"{agent_id}:{scent_id}"
        claimed = scent_id.split(":", 1)[0]
        if claimed == agent_id:
            return scent_id
        raise PermissionError(
            f"scent_id '{scent_id}' claims foreign prefix '{claimed}:'; "
            f"agent '{agent_id}' may only own '{agent_id}:'"
        )

    def freeze(self, agent_id_prefix: str) -> None:
        self._frozen_prefixes.add(agent_id_prefix)

    def unfreeze(self, agent_id_prefix: str) -> None:
        self._frozen_prefixes.discard(agent_id_prefix)

    def _scent_frozen(self, scent_id: str) -> bool:
        return any(scent_id.startswith(p) for p in self._frozen_prefixes)

    def _condition_trails(self, condition: ScentCondition) -> list[str]:
        # Composites nest arbitrarily deep; every leaf watches a trail.
        if isinstance(condition, CompositeCondition):
            return [t for sub in condition.conditions for t in self._condition_trails(sub)]
        return [condition.trail]

    def _log_event(self, now: int, agent: str, event: str, detail: str) -> None:
        print(json.dumps({"time": now, "agent": agent, "event": event, "detail": detail}))

    def _emit_stall(
        self, scent_id: str, cause: str, extra_payload: dict[str, Any], now: int
    ) -> None:
        # Default "reinforce" merge: repeated identical-shape stalls reinforce
        # one signal instead of piling up new pheromones.
        self.emit(EmitParams(
            trail="system.errors",
            type="stalled",
            intensity=1.0,
            decay=ExponentialDecay(half_life_ms=60000),
            payload={"scent_id": scent_id, "cause": cause, **extra_payload},
            source_agent="blackboard",
        ))

    async def evaluate_scents(self):
        now = self._now()
        pheromones = list(self.pheromones.values())
        ctx = EvaluationContext(
            pheromones, now, self.emission_history,
            traces=list(self.traces.values())
        )

        for scent in self.scents.values():
            # Frozen scents are skipped wholesale: no condition work, no state churn
            if self._scent_frozen(scent["id"]):
                continue

            # Cooldown check
            if now - scent["last_triggered_at"] < scent["cooldown_ms"]:
                continue

            result = evaluate_condition(scent["condition"], ctx)
            met = result.met
            last_met = scent["last_condition_met"]
            mode = scent["trigger_mode"]
            hysteresis = scent.get("hysteresis", 0) or 0
            edge_mode = mode in ("edge_rising", "edge_falling")

            # Hysteresis re-arm (§7.4): a fired scent stays disarmed until the
            # value moves hysteresis beyond the threshold away from the trigger side.
            armed = scent.get("armed", True)
            if hysteresis > 0 and edge_mode and not armed and should_rearm(
                scent["condition"], result.value, met, mode, hysteresis
            ):
                scent["armed"] = True
                armed = True

            should_trigger = False

            if mode == "level":
                should_trigger = met
            elif mode == "edge_rising":
                if hysteresis > 0:
                    should_trigger = met and armed
                else:
                    should_trigger = met and not last_met
            elif mode == "edge_falling":
                if hysteresis > 0:
                    should_trigger = not met and armed
                else:
                    should_trigger = not met and last_met

            scent["last_condition_met"] = met

            # §7.1: while an activation is in flight, further qualifying evaluations
            # are skipped — no cooldown update, no hysteresis disarm on a skipped tick.
            if should_trigger and scent.get("running"):
                scent["skipped_fires"] += 1
                should_trigger = False

            if should_trigger:
                scent["last_triggered_at"] = now
                if hysteresis > 0 and edge_mode:
                    scent["armed"] = False
                # Fire-and-forget: a slow/blocked handler for this scent must not
                # delay trigger delivery to other scents in this loop (SPECIFICATION.md §7.1).
                scent["running"] = True
                task = asyncio.create_task(self._dispatch_trigger(scent, result, now))
                self._dispatch_tasks.add(task)
                task.add_done_callback(self._dispatch_tasks.discard)

    async def _dispatch_trigger(self, scent: Dict[str, Any], result, now: int):
        t0 = self._now()
        dispatch_started = False  # frozen scents return early and get no trigger line
        try:
            # Silent skip, never a raise — a dispatch in flight when a freeze lands
            # must still clear `running` via the finally below.
            if self._scent_frozen(scent["id"]):
                return
            dispatch_started = True

            context = []
            matching_ids = set(result.matching_pheromone_ids)

            if scent.get("context_trails"):
                for p in self.pheromones.values():
                    if p.trail in scent["context_trails"] and not is_evaporated(p, now):
                        context.append(self._create_snapshot(p, now))
            else:
                for pid in matching_ids:
                    if pid in self.pheromones:
                        context.append(self._create_snapshot(self.pheromones[pid], now))

            payload = TriggerPayload(
                scent_id=scent["id"],
                triggered_at=now,
                condition_snapshot={
                    scent["id"]: {
                        "value": result.value,
                        "pheromone_ids": list(matching_ids)
                    }
                },
                context_pheromones=context,
                activation_payload=scent["activation_payload"]
            )

            handler = self.handlers.get(scent["id"])
            if handler:
                try:
                    await asyncio.wait_for(
                        handler(payload), timeout=scent["max_execution_ms"] / 1000
                    )
                except asyncio.TimeoutError:
                    self._log_event(
                        now, scent["id"], "activation_timeout",
                        f"activation timed out after {scent['max_execution_ms']}ms",
                    )
                    self._emit_stall(
                        scent["id"], "timeout", {"timeout_ms": scent["max_execution_ms"]}, now
                    )
                # asyncio.TimeoutError subclasses Exception on Python 3.10, so it is caught first
                except Exception as e:
                    self._log_event(now, scent["id"], "handler_error", f"{type(e).__name__}: {e}")
                    self._emit_stall(
                        scent["id"], "exception", {"error": f"{type(e).__name__}: {e}"}, now
                    )
        finally:
            scent["running"] = False
            skipped = scent.get("skipped_fires", 0)
            if dispatch_started:
                activation_id = f"{scent['id']}@{now}"
                self._journal_line(
                    t0, "trigger", None, scent["id"], None, "ok",
                    activation_id=activation_id, skipped_fires=skipped,
                )
            if skipped > 0:
                print(
                    f"[SBP Local] Scent {scent['id']}: {skipped} fires were skipped "
                    "while an activation was running."
                )
                scent["skipped_fires"] = 0

    def _create_snapshot(self, p: Pheromone, now: int) -> PheromoneSnapshot:
        return PheromoneSnapshot(
            id=p.id,
            trail=p.trail,
            type=p.type,
            current_intensity=compute_intensity(p, now),
            payload=p.payload,
            age_ms=now - p.emitted_at,
            tags=p.tags,
            source_agent=p.source_agent
        )

    def _prune_history(self, now: int):
        cutoff = now - self.emission_history_window
        self.emission_history = [e for e in self.emission_history if e["timestamp"] >= cutoff]

    # =========================================================================
    # TRACE OPERATIONS — Durable Knowledge Layer
    # =========================================================================

    def inscribe(self, params: dict | InscribeParams) -> InscribeResult:
        """Inscribe a trace — create or update a durable knowledge record."""
        trail, key, agent = self._inscribe_fields(params)
        self._check_rate_limit(agent)
        t0 = self._now()
        try:
            result = self._inscribe(params)
        except Exception:
            self._journal_line(t0, "inscribe", trail, key, agent, "error")
            raise
        self._journal_line(t0, "inscribe", trail, key, agent, "ok")
        return result

    @staticmethod
    def _inscribe_fields(
        params: dict[str, Any] | InscribeParams,
    ) -> tuple[Any, Any, Any]:
        if isinstance(params, dict):
            return params.get("trail"), params.get("key"), params.get("source_agent")
        return params.trail, params.key, params.source_agent

    def _inscribe(self, params: dict | InscribeParams) -> InscribeResult:
        now = self._now()

        # Accept both dict and InscribeParams
        if isinstance(params, dict):
            trail = params["trail"]
            key = params["key"]
            value = params["value"]
            tags = params.get("tags", [])
            source_agent = params.get("source_agent")
        else:
            trail = params.trail
            key = params.key
            value = params.value
            tags = params.tags
            source_agent = params.source_agent

        # Size check
        serialized = json.dumps(value)
        if len(serialized) > TRACE_MAX_VALUE_SIZE:
            raise ValueError(
                f"Trace value exceeds maximum size ({len(serialized)} > {TRACE_MAX_VALUE_SIZE} bytes)"
            )

        ck = self._trace_key(trail, key)
        existing = self.traces.get(ck)

        if existing:
            existing.value = value
            existing.updated_at = now
            existing.version += 1
            existing.tags = tags
            if source_agent:
                existing.source_agent = source_agent
            return InscribeResult(
                trace_id=existing.id,
                action="updated",
                version=existing.version,
            )

        trace_id = str(uuid.uuid4())
        trace = Trace(
            id=trace_id,
            trail=trail,
            key=key,
            value=value,
            created_at=now,
            updated_at=now,
            version=1,
            source_agent=source_agent,
            tags=tags,
        )
        self.traces[ck] = trace
        self.traces_by_id[trace_id] = ck

        return InscribeResult(
            trace_id=trace_id,
            action="created",
            version=1,
        )

    def read(self, params: ReadParams) -> ReadResult:
        """Read traces from the blackboard."""
        t0 = self._now()
        trail = ",".join(params.trails) if params.trails else None
        if params.keys:
            target = ",".join(params.keys)
        elif params.prefix:
            target = f"{params.prefix}*"
        else:
            target = None
        try:
            result = self._read(params)
        except Exception:
            self._journal_line(t0, "read", trail, target, None, "error")
            raise
        outcome = "ok" if result.traces else "not_found"
        self._journal_line(t0, "read", trail, target, None, outcome)
        return result

    def _read(self, params: ReadParams) -> ReadResult:
        now = self._now()
        results: List[Trace] = []

        for t in self.traces.values():
            if params.trails and t.trail not in params.trails:
                continue
            if params.keys and t.key not in params.keys:
                continue
            if params.prefix and not t.key.startswith(params.prefix):
                continue
            if params.tags and not match_tags(t.tags, params.tags):
                continue
            results.append(t)

        results.sort(key=lambda x: x.updated_at, reverse=True)

        return ReadResult(
            timestamp=now,
            traces=results[:params.limit],
        )

    def erase(self, params: EraseParams) -> EraseResult:
        """Erase traces matching criteria."""
        t0 = self._now()
        target = ",".join(params.keys) if params.keys else None
        try:
            result = self._erase(params)
        except Exception:
            self._journal_line(t0, "erase", params.trail, target, None, "error")
            raise
        self._journal_line(t0, "erase", params.trail, target, None, "ok")
        return result

    def _erase(self, params: EraseParams) -> EraseResult:
        now = self._now()
        to_remove: List[str] = []
        trails_affected: set[str] = set()

        for ck, t in self.traces.items():
            if params.trail and t.trail != params.trail:
                continue
            if params.keys and t.key not in params.keys:
                continue
            if params.older_than_ms is not None and now - t.updated_at < params.older_than_ms:
                continue
            if params.tags and not match_tags(t.tags, params.tags):
                continue
            to_remove.append(ck)
            trails_affected.add(t.trail)

        for ck in to_remove:
            trace = self.traces.pop(ck)
            self.traces_by_id.pop(trace.id, None)

        return EraseResult(
            erased_count=len(to_remove),
            trails_affected=list(trails_affected),
        )

    def evaporate(self, params: EvaporateParams) -> EvaporateResult:
        """Force evaporation of pheromones matching criteria."""
        t0 = self._now()
        try:
            result = self._evaporate(params)
        except Exception:
            self._journal_line(t0, "evaporate", params.trail, None, None, "error")
            raise
        self._journal_line(t0, "evaporate", params.trail, None, None, "ok")
        return result

    def _evaporate(self, params: EvaporateParams) -> EvaporateResult:
        now = self._now()
        to_remove: List[str] = []
        trails_affected: set[str] = set()

        for pid, p in self.pheromones.items():
            if params.trail and p.trail != params.trail:
                continue
            if params.types and p.type not in params.types:
                continue
            if params.older_than_ms is not None and now - p.emitted_at < params.older_than_ms:
                continue
            if params.below_intensity is not None and compute_intensity(p, now) >= params.below_intensity:
                continue
            if params.tags and not match_tags(p.tags, params.tags):
                continue
            to_remove.append(pid)
            trails_affected.add(p.trail)

        for pid in to_remove:
            del self.pheromones[pid]

        return EvaporateResult(
            evaporated_count=len(to_remove),
            trails_affected=list(trails_affected),
        )

    def gc(self) -> int:
        """Delete evaporated pheromones from the in-memory store. Returns the count removed."""
        now = self._now()
        expired = [pid for pid, p in self.pheromones.items() if is_evaporated(p, now)]
        for pid in expired:
            del self.pheromones[pid]
        return len(expired)

    def inspect(self, params: InspectParams) -> InspectResult:
        """Inspect blackboard state."""
        t0 = self._now()
        try:
            result = self._inspect(params)
        except Exception:
            self._journal_line(t0, "inspect", None, None, None, "error")
            raise
        self._journal_line(t0, "inspect", None, None, None, "ok")
        return result

    def _inspect(self, params: InspectParams) -> InspectResult:
        now = self._now()
        include = params.include or ["trails", "scents", "stats"]
        result = InspectResult(timestamp=now)

        if "trails" in include:
            trail_map: Dict[str, Dict[str, float]] = {}
            for p in self.pheromones.values():
                if is_evaporated(p, now):
                    continue
                data = trail_map.setdefault(p.trail, {"count": 0, "intensity": 0.0})
                data["count"] += 1
                data["intensity"] += compute_intensity(p, now)
            result.trails = [
                {
                    "name": name,
                    "pheromone_count": int(data["count"]),
                    "total_intensity": data["intensity"],
                    "avg_intensity": data["intensity"] / data["count"] if data["count"] else 0,
                }
                for name, data in trail_map.items()
            ]

        if "scents" in include:
            result.scents = [
                {
                    "scent_id": s["id"],
                    "condition_met": s["last_condition_met"],
                    "in_cooldown": bool(s["last_triggered_at"]) and (now - s["last_triggered_at"] < s["cooldown_ms"]),
                    "last_triggered_at": s["last_triggered_at"] or None,
                }
                for s in self.scents.values()
            ]

        if "stats" in include:
            active_count = sum(1 for p in self.pheromones.values() if not is_evaporated(p, now))
            result.stats = {
                "total_pheromones": len(self.pheromones),
                "active_pheromones": active_count,
                "total_scents": len(self.scents),
                "total_traces": len(self.traces),
                "uptime_ms": now - self.start_time,
            }

        return result


# Singleton instance for shared local mode
_shared_blackboard: Optional[LocalBlackboard] = None

def get_shared_blackboard() -> LocalBlackboard:
    global _shared_blackboard
    if _shared_blackboard is None:
        _shared_blackboard = LocalBlackboard()
    return _shared_blackboard
