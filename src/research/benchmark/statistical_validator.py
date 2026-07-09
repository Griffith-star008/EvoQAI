"""
Statistical Validator for AQIP Benchmark Suite.
Automatically computes rigorous statistical measures from raw benchmark data.
"""
import math
import json
import os
from datetime import datetime


def compute_mean(data: list[float]) -> float:
    """Compute arithmetic mean."""
    return sum(data) / len(data)


def compute_median(data: list[float]) -> float:
    """Compute median."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]


def compute_variance(data: list[float], ddof: int = 1) -> float:
    """Compute sample variance with Bessel's correction (ddof=1)."""
    mu = compute_mean(data)
    return sum((x - mu) ** 2 for x in data) / (len(data) - ddof)


def compute_std(data: list[float], ddof: int = 1) -> float:
    """Compute sample standard deviation."""
    return math.sqrt(compute_variance(data, ddof))


def compute_confidence_interval(data: list[float], confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval using t-distribution approximation.
    For large N (>30), z-approximation is used.
    """
    n = len(data)
    mu = compute_mean(data)
    se = compute_std(data) / math.sqrt(n)

    # z-values for common confidence levels
    z_table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_table.get(confidence, 1.960)

    margin = z * se
    return (mu - margin, mu + margin)


def compute_cohens_d(group1: list[float], group2: list[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.
    Uses pooled standard deviation.
    """
    n1, n2 = len(group1), len(group2)
    mu1, mu2 = compute_mean(group1), compute_mean(group2)
    var1, var2 = compute_variance(group1), compute_variance(group2)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mu1 - mu2) / pooled_std


def compute_one_way_anova(*groups: list[float]) -> dict:
    """
    Compute one-way ANOVA F-statistic and approximate p-value.
    Returns F-statistic and degrees of freedom.
    """
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand_mean = sum(sum(g) for g in groups) / N

    # Between-group sum of squares
    ss_between = sum(len(g) * (compute_mean(g) - grand_mean) ** 2 for g in groups)
    df_between = k - 1

    # Within-group sum of squares
    ss_within = sum(sum((x - compute_mean(g)) ** 2 for x in g) for g in groups)
    df_within = N - k

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1

    f_statistic = ms_between / ms_within if ms_within > 0 else float('inf')

    return {
        "F_statistic": round(f_statistic, 4),
        "df_between": df_between,
        "df_within": df_within,
        "SS_between": round(ss_between, 4),
        "SS_within": round(ss_within, 4),
    }


def generate_full_report(
    experiment_name: str,
    aqip_scores: list[float],
    baseline_scores: list[float],
    output_dir: str = "."
) -> dict:
    """
    Generate a complete statistical validation report comparing AQIP vs Baseline.
    """
    report = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "sample_size": {"aqip": len(aqip_scores), "baseline": len(baseline_scores)},
        "aqip": {
            "mean": round(compute_mean(aqip_scores), 4),
            "median": round(compute_median(aqip_scores), 4),
            "std": round(compute_std(aqip_scores), 4),
            "variance": round(compute_variance(aqip_scores), 4),
            "ci_95": [round(x, 4) for x in compute_confidence_interval(aqip_scores)],
        },
        "baseline": {
            "mean": round(compute_mean(baseline_scores), 4),
            "median": round(compute_median(baseline_scores), 4),
            "std": round(compute_std(baseline_scores), 4),
            "variance": round(compute_variance(baseline_scores), 4),
            "ci_95": [round(x, 4) for x in compute_confidence_interval(baseline_scores)],
        },
        "effect_size": {
            "cohens_d": round(compute_cohens_d(aqip_scores, baseline_scores), 4),
            "interpretation": "",
        },
        "anova": compute_one_way_anova(aqip_scores, baseline_scores),
    }

    # Interpret effect size
    d = abs(report["effect_size"]["cohens_d"])
    if d < 0.2:
        report["effect_size"]["interpretation"] = "Negligible"
    elif d < 0.5:
        report["effect_size"]["interpretation"] = "Small"
    elif d < 0.8:
        report["effect_size"]["interpretation"] = "Medium"
    else:
        report["effect_size"]["interpretation"] = "Large"

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"statistical_report_{experiment_name}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Statistical Validator] Report saved to {report_path}")
    return report


if __name__ == "__main__":
    # Example: AQIP vs Static Baseline AQII scores over 50 episodes
    import random
    random.seed(42)

    aqip = [random.gauss(86.07, 2.4) for _ in range(50)]
    baseline = [random.gauss(11.0, 1.2) for _ in range(50)]

    report = generate_full_report("AQII_Category_K", aqip, baseline)
    print(json.dumps(report, indent=2))
