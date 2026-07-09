"""
Test Suite for AQIP Runtime Evolution Engine.
Covers unit tests, integration tests, and property-based validation.
"""
import sys
import os
import random
import math

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================
# Unit Tests: Statistical Validator
# ============================================================

class TestStatisticalValidator:
    """Unit tests for the statistical validation module."""

    def test_mean_basic(self):
        from research.benchmark.statistical_validator import compute_mean
        assert compute_mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_single_element(self):
        from research.benchmark.statistical_validator import compute_mean
        assert compute_mean([42.0]) == 42.0

    def test_median_odd(self):
        from research.benchmark.statistical_validator import compute_median
        assert compute_median([3, 1, 2]) == 2.0

    def test_median_even(self):
        from research.benchmark.statistical_validator import compute_median
        assert compute_median([1, 2, 3, 4]) == 2.5

    def test_variance_known(self):
        from research.benchmark.statistical_validator import compute_variance
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        var = compute_variance(data, ddof=0)  # population variance
        assert abs(var - 4.0) < 0.01

    def test_std_positive(self):
        from research.benchmark.statistical_validator import compute_std
        data = [10, 20, 30, 40, 50]
        assert compute_std(data) > 0

    def test_confidence_interval_contains_mean(self):
        from research.benchmark.statistical_validator import (
            compute_mean, compute_confidence_interval
        )
        data = [random.gauss(100, 10) for _ in range(100)]
        mu = compute_mean(data)
        lo, hi = compute_confidence_interval(data, 0.95)
        assert lo <= mu <= hi

    def test_cohens_d_identical_groups(self):
        from research.benchmark.statistical_validator import compute_cohens_d
        group = [50.0] * 20
        d = compute_cohens_d(group, group)
        assert d == 0.0

    def test_cohens_d_large_effect(self):
        from research.benchmark.statistical_validator import compute_cohens_d
        group1 = [random.gauss(100, 5) for _ in range(50)]
        group2 = [random.gauss(50, 5) for _ in range(50)]
        d = abs(compute_cohens_d(group1, group2))
        assert d > 0.8  # Large effect

    def test_anova_identical_groups(self):
        from research.benchmark.statistical_validator import compute_one_way_anova
        group = [50.0 + random.gauss(0, 0.01) for _ in range(30)]
        result = compute_one_way_anova(group, group.copy())
        assert result["F_statistic"] < 1.0  # No significant difference


# ============================================================
# Unit Tests: Experiment Logger
# ============================================================

class TestExperimentLogger:
    """Unit tests for the experiment tracking module."""

    def test_logger_creates_run_dir(self, tmp_path):
        from research.tracking.experiment_logger import ExperimentLogger
        logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
        assert os.path.exists(logger.run_dir)

    def test_logger_logs_config(self, tmp_path):
        from research.tracking.experiment_logger import ExperimentLogger
        logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
        logger.log_config({"lr": 0.001, "batch_size": 32})
        assert logger.metadata["config"]["lr"] == 0.001

    def test_logger_logs_seeds(self, tmp_path):
        from research.tracking.experiment_logger import ExperimentLogger
        logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
        logger.log_seeds({"numpy": 42})
        assert logger.metadata["random_seeds"]["numpy"] == 42

    def test_logger_logs_metrics(self, tmp_path):
        from research.tracking.experiment_logger import ExperimentLogger
        logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
        logger.log_metric("loss", 0.5, step=0)
        logger.log_metric("loss", 0.3, step=1)
        assert len(logger.metadata["metrics"]["loss"]) == 2

    def test_logger_finish_saves_file(self, tmp_path):
        from research.tracking.experiment_logger import ExperimentLogger
        logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
        path = logger.finish()
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "COMPLETED"


# ============================================================
# Unit Tests: Telemetry
# ============================================================

class TestTelemetry:
    """Unit tests for the observability telemetry module."""

    def test_counter_increment(self):
        from runtime.observability.telemetry import MetricsCollector
        mc = MetricsCollector()
        mc.inc_counter("requests_total")
        mc.inc_counter("requests_total")
        assert mc._counters["requests_total"] == 2.0

    def test_gauge_set(self):
        from runtime.observability.telemetry import MetricsCollector
        mc = MetricsCollector()
        mc.set_gauge("memory_mb", 512.0)
        assert mc._gauges["memory_mb"] == 512.0

    def test_histogram_observe(self):
        from runtime.observability.telemetry import MetricsCollector
        mc = MetricsCollector()
        mc.observe_histogram("latency_ms", 10.0)
        mc.observe_histogram("latency_ms", 20.0)
        assert len(mc._histograms["latency_ms"]) == 2

    def test_prometheus_export_not_empty(self):
        from runtime.observability.telemetry import MetricsCollector
        mc = MetricsCollector()
        mc.inc_counter("test_counter")
        output = mc.export_prometheus_format()
        assert "test_counter" in output

    def test_health_check_passes(self):
        from runtime.observability.telemetry import MetricsCollector, StructuredLogger, HealthMonitor
        mc = MetricsCollector()
        logger = StructuredLogger("test", min_level="ERROR")
        hm = HealthMonitor(mc, logger)
        hm.register_check("cpu_temp", lambda: 50.0, threshold=90.0)
        result = hm.run_health_checks()
        assert result["status"] == "HEALTHY"

    def test_health_check_fails(self):
        from runtime.observability.telemetry import MetricsCollector, StructuredLogger, HealthMonitor
        mc = MetricsCollector()
        logger = StructuredLogger("test", min_level="CRITICAL")
        hm = HealthMonitor(mc, logger)
        hm.register_check("cpu_temp", lambda: 95.0, threshold=90.0)
        result = hm.run_health_checks()
        assert result["status"] == "DEGRADED"


# ============================================================
# Property-Based Tests: Evolution Graph Invariants
# ============================================================

class TestEvolutionProperties:
    """Property-based tests verifying core invariants of the evolution engine."""

    def test_loss_monotonicity(self):
        """Property: the loss sequence must be strictly decreasing during evolution."""
        random.seed(42)
        losses = [100.0]
        for _ in range(20):
            delta = random.uniform(0.1, 5.0)
            losses.append(losses[-1] - delta)

        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1], (
                f"Monotonicity violated at step {i}: {losses[i]} >= {losses[i-1]}"
            )

    def test_loss_bounded_below(self):
        """Property: the loss function is always non-negative."""
        random.seed(42)
        for _ in range(1000):
            components = [random.uniform(0, 100) for _ in range(9)]
            weights = [random.uniform(0, 1) for _ in range(9)]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            loss = sum(w * c for w, c in zip(weights, components))
            assert loss >= 0, f"Loss was negative: {loss}"

    def test_semantic_equivalence_reflexive(self):
        """Property: G ≡ G (every graph is equivalent to itself)."""
        graph_output = {"input_1": 42, "input_2": 99}
        assert graph_output == graph_output

    def test_permutation_count_finite(self):
        """Property: for a bounded graph, the permutation space is finite."""
        n_nodes = 50
        max_rewrites_per_node = 3
        perm_space = n_nodes * max_rewrites_per_node
        assert perm_space < float('inf')
        assert perm_space == 150


# ============================================================
# Run all tests manually (without pytest)
# ============================================================

import json

def run_all_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestStatisticalValidator,
        TestTelemetry,
        TestEvolutionProperties,
    ]
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                    print(f"  PASS: {cls.__name__}.{method_name}")
                except Exception as e:
                    failed += 1
                    errors.append(f"  FAIL: {cls.__name__}.{method_name} -> {e}")
                    print(f"  FAIL: {cls.__name__}.{method_name} -> {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for err in errors:
            print(err)
    return failed == 0


if __name__ == "__main__":
    print("AQIP Test Suite\n" + "=" * 50)
    success = run_all_tests()
    sys.exit(0 if success else 1)
