import math


class RateLimitError(Exception):
    def __init__(self, retry_after_ms: int, limit: int, window_ms: int):
        self.retry_after_ms = retry_after_ms
        self.limit = limit
        self.window_ms = window_ms
        super().__init__(f"Rate limited: retry after {retry_after_ms}ms")


class RateLimiter:
    def __init__(self, max_requests: int, window_ms: int):
        self.max_requests = max_requests
        self.window_ms = window_ms
        self._buckets: dict[str, dict[str, float]] = {}

    def check(self, agent_id: str, now: int) -> None:
        bucket = self._buckets.get(agent_id)
        if bucket is None:
            bucket = {"tokens": float(self.max_requests), "last_refill": float(now)}
            self._buckets[agent_id] = bucket

        elapsed = now - bucket["last_refill"]
        refill_rate = self.max_requests / self.window_ms
        bucket["tokens"] = min(self.max_requests, bucket["tokens"] + elapsed * refill_rate)
        bucket["last_refill"] = now

        if bucket["tokens"] < 1:
            retry_after_ms = math.ceil((1 - bucket["tokens"]) / refill_rate)
            raise RateLimitError(retry_after_ms, self.max_requests, self.window_ms)

        bucket["tokens"] -= 1
