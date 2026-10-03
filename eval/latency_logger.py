"""Latency logger for pipeline stages.

Tracks wall-clock time per stage, supporting p50/p95 summaries.
"""
import statistics
import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List


class LatencyLogger:
    """Logs wall-clock latency for named pipeline stages."""

    STAGES = ["retrieve", "generate", "extract", "verify", "score", "decide"]

    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        """Context manager that records wall-clock time for a stage (in ms)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings[name].append(elapsed_ms)

    def summary(self) -> dict:
        """Return {stage_name: {count, p50_ms, p95_ms}} for each recorded stage."""
        result = {}
        for name, times in self._timings.items():
            if not times:
                continue
            sorted_times = sorted(times)
            n = len(sorted_times)
            p50 = sorted_times[int(n * 0.50)]
            p95_idx = min(int(n * 0.95), n - 1)
            p95 = sorted_times[p95_idx]
            result[name] = {"count": n, "p50_ms": p50, "p95_ms": p95}
        return result

    def log(self, query_id: str) -> None:
        """Print per-query timings to stdout."""
        print(f"[LatencyLogger] query_id={query_id}")
        for name, times in self._timings.items():
            if times:
                last = times[-1]
                print(f"  {name}: {last:.2f} ms (last)")
