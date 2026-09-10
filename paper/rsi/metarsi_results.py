"""Recompute and visualise the numbers reported for MetaRSI-v1 (arXiv:2609.06396v1).

All inputs are transcribed from the paper (Table 3, Section 5.3, Section 5.4).
Nothing here reruns the paper's experiments. The goal is to make the reported
aggregates auditable and to surface derived quantities the paper does not
print directly (e.g. the implied term-1 gain in the five-term run).

Requirements: numpy, matplotlib (already in the repo's pyproject.toml).

Usage:
    uv run python paper/rsi/metarsi_results.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
IMG_DIR = HERE / "images"

# --------------------------------------------------------------------------- #
# Table 3: main results on Qwen3.5-35B-A3B, mean over 5 outer seeds.
# --------------------------------------------------------------------------- #
BENCHMARKS = ["Terminal-Bench 2.1", "SWE-bench Pro", "GPQA-hard100", "AIME"]
TABLE3 = {
    "Initial":         [23.6, 10.3, 71.2, 55.0],
    "Data-RSI":        [27.4, 12.4, 73.8, 57.7],
    "Harness-RSI":     [29.4, 14.9, 78.8, 63.3],
    "Model-RSI":       [27.0, 14.0, 75.4, 59.3],
    "Fixed D->M->H":   [30.3, 15.3, 79.4, 64.3],
    "Random legal":    [28.1, 13.6, 75.8, 60.7],
    "Static routing":  [30.8, 14.9, 79.6, 64.0],
    "MetaRSI-v1":      [31.9, 19.5, 83.8, 68.3],
}
# Average gain column as printed in the paper.
REPORTED_AVG_GAIN = {
    "Data-RSI": 2.8, "Harness-RSI": 6.6, "Model-RSI": 3.9, "Fixed D->M->H": 7.3,
    "Random legal": 4.5, "Static routing": 7.3, "MetaRSI-v1": 10.9,
}

# --------------------------------------------------------------------------- #
# Section 5.4: five consecutive terms, original vs meta-updated improver.
# Only the values stated in the text are transcribed; terms 3 and 4 are not
# printed individually in the paper.
# --------------------------------------------------------------------------- #
META_RUN = {
    "original": {"cumulative": 20.7, "term2": 4.9, "term5": 0.7, "avg_t2_t5": 2.5},
    "meta":     {"cumulative": 26.2, "term2": 7.2, "term5": 1.4, "avg_t2_t5": 3.8},
}
# Per-benchmark advantage of the meta improver at term 2.
META_TERM2_ADVANTAGE = [1.6, 1.7, 2.6, 3.3]

# --------------------------------------------------------------------------- #
# Section 5.3: frontier fleet under the {D, H} loop on Terminal-Bench 2.1.
# --------------------------------------------------------------------------- #
FRONTIER = {"term_gains": [7.3, 2.8, 1.2], "fleet_mean_start": 80.0, "fleet_mean_end": 91.3}


def table3_summary() -> dict[str, float]:
    """Recompute per-method mean accuracy and average gain vs the initial system."""
    base = np.array(TABLE3["Initial"])
    out: dict[str, float] = {}
    print("## Table 3 recomputed\n")
    print("| Method | Mean acc | Avg gain (recomputed) | Avg gain (reported) | Diff |")
    print("|---|---:|---:|---:|---:|")
    for name, scores in TABLE3.items():
        arr = np.array(scores)
        gain = float((arr - base).mean())
        out[name] = gain
        reported = REPORTED_AVG_GAIN.get(name)
        rep_str = f"{reported:+.1f}" if reported is not None else "—"
        diff_str = f"{gain - reported:+.2f}" if reported is not None else "—"
        print(f"| {name} | {arr.mean():.2f} | {gain:+.2f} | {rep_str} | {diff_str} |")

    full, fixed, static = out["MetaRSI-v1"], out["Fixed D->M->H"], out["Static routing"]
    print()
    print(f"MetaRSI-v1 vs fixed pipeline : {full - fixed:+.2f} pts")
    print(f"MetaRSI-v1 vs static routing : {full - static:+.2f} pts")
    print(f"Best single operator          : {max(('Data-RSI', 'Harness-RSI', 'Model-RSI'), key=out.get)}")

    # Per-benchmark gain of the full system over the strongest baseline.
    others = np.array([v for k, v in TABLE3.items() if k not in ("Initial", "MetaRSI-v1")])
    margin = np.array(TABLE3["MetaRSI-v1"]) - others.max(axis=0)
    print("\nMargin of MetaRSI-v1 over the best non-full baseline, per benchmark:")
    for b, m in zip(BENCHMARKS, margin):
        print(f"  {b:<20} {m:+.1f}")
    return out


def meta_run_decomposition() -> None:
    """Derive the quantities Section 5.4 implies but does not print."""
    print("\n## Five-term run (Section 5.4), implied quantities\n")
    print("| Path | Sum t2..t5 | Implied t1 | Implied t3+t4 |")
    print("|---|---:|---:|---:|")
    for path, v in META_RUN.items():
        sum_t2_t5 = 4 * v["avg_t2_t5"]
        implied_t1 = v["cumulative"] - sum_t2_t5
        implied_t3_t4 = sum_t2_t5 - v["term2"] - v["term5"]
        print(f"| {path} | {sum_t2_t5:.1f} | {implied_t1:.1f} | {implied_t3_t4:.1f} |")
    print(
        "\nNote: both branches fork from the same term-1 release, so implied t1 should "
        f"match Table 3's +{REPORTED_AVG_GAIN['MetaRSI-v1']:.1f}. The ~0.3 spread is "
        "consistent with one-decimal rounding of the printed averages."
    )
    adv = np.array(META_TERM2_ADVANTAGE)
    print(f"Mean per-benchmark advantage at term 2: {adv.mean():.2f} (paper: 7.2 - 4.9 = 2.3)")


def frontier_check() -> None:
    """Check the frontier-fleet compounding numbers add up."""
    print("\n## Frontier fleet (Section 5.3)\n")
    gains = np.array(FRONTIER["term_gains"])
    cum = gains.cumsum()
    print(f"Per-term gains      : {gains.tolist()}")
    print(f"Cumulative          : {np.round(cum, 1).tolist()}  (paper: +11.3)")
    print(f"Retention t2/t1     : {gains[1] / gains[0]:.2f},  t3/t2: {gains[2] / gains[1]:.2f}")
    end = FRONTIER["fleet_mean_start"] + cum[-1]
    print(f"Fleet mean start+cum: {end:.1f}  (paper: {FRONTIER['fleet_mean_end']})")


def breakeven_calls(c_train: float, delta_c: float) -> float:
    """Number of later calls needed before training cost is amortised."""
    if delta_c <= 0:
        return float("inf")
    return c_train / delta_c


def make_plots() -> None:
    IMG_DIR.mkdir(exist_ok=True)

    # 1) Table 3 grouped bars.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    methods = list(TABLE3)
    x = np.arange(len(BENCHMARKS))
    w = 0.1
    for i, m in enumerate(methods):
        ax.bar(x + (i - len(methods) / 2 + 0.5) * w, TABLE3[m], w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MetaRSI-v1 Table 3 (Qwen3.5-35B-A3B, mean of 5 seeds)")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "table3_bars.png", dpi=150)
    plt.close(fig)

    # 2) Five-term per-term gain: only the printed points, dashed where unknown.
    fig, ax = plt.subplots(figsize=(6, 4))
    for path, v in META_RUN.items():
        t1 = v["cumulative"] - 4 * v["avg_t2_t5"]
        known_t = [1, 2, 5]
        known_g = [t1, v["term2"], v["term5"]]
        ax.plot(known_t, known_g, "o--", label=f"{path} (t1 implied, t3/t4 not printed)")
        ax.axhline(v["avg_t2_t5"], ls=":", lw=0.8, color=ax.lines[-1].get_color())
    ax.axhline(1.5, color="gray", lw=0.8, ls="-.", label="plateau threshold (1.5)")
    ax.set_xlabel("Term")
    ax.set_ylabel("Gain per term (avg pts)")
    ax.set_title("Per-term gain, original vs meta-updated improver (§5.4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "meta_per_term_gain.png", dpi=150)
    plt.close(fig)

    # 3) Break-even calls for internalising a harness into weights.
    fig, ax = plt.subplots(figsize=(6, 4))
    delta_c = np.linspace(0.001, 0.05, 200)
    for c_train in (50, 200, 1000):
        ax.plot(delta_c, [breakeven_calls(c_train, d) for d in delta_c], label=f"C_train = {c_train}")
    ax.set_yscale("log")
    ax.set_xlabel("Per-call saving Δc (same unit as C_train)")
    ax.set_ylabel("Calls to break even, N")
    ax.set_title("N > C_train / Δc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG_DIR / "breakeven.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    table3_summary()
    meta_run_decomposition()
    frontier_check()
    print("\n## Break-even example\n")
    for c_train, dc in ((200.0, 0.01), (200.0, 0.002)):
        print(f"C_train={c_train}, Δc={dc}: N > {breakeven_calls(c_train, dc):,.0f} calls")
    make_plots()
    print(f"\nPlots written to {IMG_DIR}")
