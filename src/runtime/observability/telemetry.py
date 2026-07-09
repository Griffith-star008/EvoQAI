"""
AQIP Runtime Observability: Telemetry Collector.
Provides structured logging, metrics collection, and health monitoring
for the autonomous runtime components.
"""
import time
import json
import os
from datetime import datetime
from collections import defaultdict


class MetricsCollector:
    """Collects and aggregates runtime metrics (simulating Prometheus-style gauges/counters)."""

    def __init__(self):
        self._counters = defaultdict(float)
        self._gauges = defaultdict(float)
        self._histograms = defaultdict(list)
        self._start_time = time.time()

    def inc_counter(self, name: str, value: float = 1.0, labels: dict = None):
        """Increment a monotonic counter (e.g., total_evolutions, total_verifications)."""
        key = self._make_key(name, labels)
        self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: dict = None):
        """Set a gauge to a specific value (e.g., current_memory_usage, cpu_utilization)."""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: dict = None):
        """Record an observation in a histogram (e.g., evolution_latency_ms)."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)

    def _make_key(self, name: str, labels: dict = None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def export_prometheus_format(self) -> str:
        """Export all metrics in Prometheus exposition format."""
        lines = [f"# AQIP Telemetry Export @ {datetime.now().isoformat()}\n"]
        for key, val in self._counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {val}")
        for key, val in self._gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {val}")
        for key, vals in self._histograms.items():
            base = key.split('{')[0]
            lines.append(f"# TYPE {base} histogram")
            lines.append(f"{base}_count {len(vals)}")
            lines.append(f"{base}_sum {sum(vals):.4f}")
            if vals:
                lines.append(f"{base}_avg {sum(vals)/len(vals):.4f}")
        return "\n".join(lines)


class StructuredLogger:
    """JSON-structured logging for distributed runtime observability."""

    LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

    def __init__(self, component: str, log_dir: str = "logs", min_level: str = "INFO"):
        self.component = component
        self.min_level = self.LEVELS.get(min_level, 20)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{component}_{datetime.now().strftime('%Y%m%d')}.jsonl")

    def _log(self, level: str, message: str, **kwargs):
        if self.LEVELS.get(level, 0) < self.min_level:
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "component": self.component,
            "message": message,
            **kwargs,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if level in ("ERROR", "CRITICAL"):
            print(f"[{level}] [{self.component}] {message}")

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)


class HealthMonitor:
    """Runtime health checker with self-healing capabilities."""

    def __init__(self, metrics: MetricsCollector, logger: StructuredLogger):
        self.metrics = metrics
        self.logger = logger
        self.checks = []

    def register_check(self, name: str, check_fn, threshold: float):
        """Register a health check function."""
        self.checks.append({"name": name, "fn": check_fn, "threshold": threshold})

    def run_health_checks(self) -> dict:
        """Execute all registered health checks and return status."""
        results = {"timestamp": datetime.now().isoformat(), "status": "HEALTHY", "checks": []}

        for check in self.checks:
            value = check["fn"]()
            passed = value <= check["threshold"]
            result = {
                "name": check["name"],
                "value": value,
                "threshold": check["threshold"],
                "passed": passed,
            }
            results["checks"].append(result)

            if not passed:
                results["status"] = "DEGRADED"
                self.logger.warning(
                    f"Health check FAILED: {check['name']}",
                    value=value,
                    threshold=check["threshold"],
                )
                self.metrics.inc_counter("health_check_failures", labels={"check": check["name"]})

        self.logger.info(f"Health check completed: {results['status']}")
        return results


if __name__ == "__main__":
    metrics = MetricsCollector()
    logger = StructuredLogger("runtime_evolution", min_level="DEBUG")
    health = HealthMonitor(metrics, logger)

    # Simulate runtime telemetry
    metrics.inc_counter("total_evolution_cycles")
    metrics.set_gauge("memory_usage_mb", 1024.5)
    metrics.observe_histogram("evolution_latency_ms", 45.2)
    metrics.observe_histogram("evolution_latency_ms", 52.1)
    metrics.observe_histogram("evolution_latency_ms", 38.7)

    logger.info("Runtime evolution cycle completed", cycle=1, loss_delta=-0.042)

    # Health check
    health.register_check("memory_mb", lambda: 1024.5, threshold=4096.0)
    health.register_check("cpu_temp_c", lambda: 72.0, threshold=90.0)
    status = health.run_health_checks()

    print("\n" + metrics.export_prometheus_format())
    print(f"\nHealth Status: {status['status']}")
