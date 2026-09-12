"""
src/simulation/test_simulation.py
==================================
Validation script for the physics-based vibration signal generator.

Generates ONE signal for each of the three operating conditions,
computes validation statistics, and saves two diagnostic plots.

WHAT THIS SCRIPT DOES:
  1. Generates one signal per condition using the SDOF dynamic model.
  2. Checks for NaN/Inf values (numerical stability).
  3. Calculates RMS, peak acceleration, and standard deviation.
  4. Verifies that all three conditions produce distinguishable responses.
  5. Saves two plots to results/figures/:
       - Plot 1: Time-domain waveforms (all three conditions)
       - Plot 2: RMS comparison bar chart

HOW TO RUN (from the Digital_Twin_Project root directory):
    python src/simulation/test_simulation.py

OUTPUT:
    results/figures/validation_waveforms.png
    results/figures/validation_rms_comparison.png

NOTE: This is a VALIDATION script, not the dataset generator.
      For the full 500-signal dataset, use generate_dataset.py.
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")    # Non-interactive backend -- saves to file, no blocking window
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_FILE    = os.path.abspath(__file__)
_SIM_DIR      = os.path.dirname(_THIS_FILE)
_SRC_DIR      = os.path.dirname(_SIM_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from config.parameters import (
    CONDITIONS,
    SAMPLING_RATE_HZ,
    SIGNAL_DURATION_S,
    NUM_SAMPLES,
    MASS_KG,
    NATURAL_FREQUENCY_HZ,
    DAMPING_RATIO,
    STIFFNESS_NM,
    DAMPING_COEFF_NSM,
)
from src.simulation.signal_generator import generate_vibration

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")


def compute_statistics(accel: np.ndarray) -> dict:
    """
    Compute basic validation statistics for a vibration acceleration signal.

    Parameters
    ----------
    accel : np.ndarray
        Acceleration signal (m/s^2).

    Returns
    -------
    dict with keys: rms, peak, std, min, max, n_samples, has_nan, has_inf
    """
    return {
        "rms":      float(np.sqrt(np.mean(accel ** 2))),
        "peak":     float(np.max(np.abs(accel))),
        "std":      float(np.std(accel)),
        "min":      float(np.min(accel)),
        "max":      float(np.max(accel)),
        "n_samples": len(accel),
        "has_nan":  bool(np.any(np.isnan(accel))),
        "has_inf":  bool(np.any(np.isinf(accel))),
    }


def print_validation_header():
    """Print the system parameter summary."""
    print("=" * 65)
    print("  Digital Twin -- Physics-Based SDOF Model Validation")
    print("=" * 65)
    print()
    print("  SYSTEM PARAMETERS (all values ASSUMED for educational model):")
    print(f"    Effective mass     m  = {MASS_KG} kg")
    print(f"    Natural frequency  fn = {NATURAL_FREQUENCY_HZ} Hz")
    print(f"    Damping ratio      z  = {DAMPING_RATIO}  (5%)")
    print(f"    Stiffness          k  = {STIFFNESS_NM/1e6:.4f} MN/m")
    print(f"    Damping coeff.     c  = {DAMPING_COEFF_NSM:.2f} N.s/m")
    print()
    print(f"  SIMULATION SETTINGS:")
    print(f"    Sampling rate      Fs = {SAMPLING_RATE_HZ} Hz")
    print(f"    Signal duration    T  = {SIGNAL_DURATION_S} s")
    print(f"    Samples per signal N  = {NUM_SAMPLES}")
    print()


def validate_all_conditions(fixed_seed_base: int = 999):
    """
    Generate one signal per condition and return results + statistics.

    Uses fixed seeds so the validation plot is reproducible.
    """
    results = {}

    for idx, condition in enumerate(CONDITIONS):
        seed = fixed_seed_base + idx   # deterministic, distinct per condition

        print(f"  Generating signal: '{condition}' (seed={seed}) ...", end=" ")
        t, accel = generate_vibration(condition, seed=seed)
        stats = compute_statistics(accel)
        results[condition] = {"t": t, "accel": accel, "stats": stats}

        status = "[FAIL -- NaN/Inf detected!]" if (stats["has_nan"] or stats["has_inf"]) else "[OK]"
        print(status)

    return results


def print_statistics_table(results: dict):
    """Print a formatted statistics table for all three conditions."""
    print()
    print("  VALIDATION STATISTICS (one representative signal per condition):")
    print()
    header = f"  {'Condition':<25} {'RMS':>8} {'Peak':>8} {'Std Dev':>8} {'Min':>8} {'Max':>8} {'N':>6} {'Stable':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rms_values = []
    for condition in CONDITIONS:
        s = results[condition]["stats"]
        stable = "PASS" if not (s["has_nan"] or s["has_inf"]) else "FAIL"
        rms_values.append(s["rms"])
        print(
            f"  {condition:<25} "
            f"{s['rms']:>8.4f} "
            f"{s['peak']:>8.4f} "
            f"{s['std']:>8.4f} "
            f"{s['min']:>8.4f} "
            f"{s['max']:>8.4f} "
            f"{s['n_samples']:>6} "
            f"{stable:>7}"
        )

    # Distinguishability check: each condition should have higher RMS than previous
    print()
    rms_h, rms_sl, rms_sv = rms_values
    print(f"  DISTINGUISHABILITY CHECK:")
    print(f"    RMS (healthy)           = {rms_h:.4f} m/s^2  [baseline]")
    print(f"    RMS (slight_misalign.)  = {rms_sl:.4f} m/s^2  ({rms_sl/rms_h:.2f}x healthy)")
    print(f"    RMS (severe_misalign.)  = {rms_sv:.4f} m/s^2  ({rms_sv/rms_h:.2f}x healthy)")

    if rms_h < rms_sl < rms_sv:
        print(f"    Result: [PASS] -- RMS increases monotonically with severity.")
    else:
        print(f"    Result: [WARN] -- RMS does not increase monotonically. Check parameters.")
    print()


def plot_waveforms(results: dict, output_path: str):
    """
    Plot 1: Time-domain acceleration waveforms for all three conditions.
    Shows first 0.15 s (4.5 shaft revolutions) for clear waveform visibility.
    """
    display_seconds = 0.15
    display_samples = int(display_seconds * SAMPLING_RATE_HZ)

    condition_styles = {
        "healthy":             ("Healthy  (1x = 30 Hz only)",            "steelblue"),
        "slight_misalignment": ("Slight Misalignment  (1x + 2x)",        "darkorange"),
        "severe_misalignment": ("Severe Misalignment  (1x + 2x + 3x)",   "crimson"),
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        "Digital Twin -- Physics-Based SDOF Vibration Signals\n"
        "Motor-Shaft-Bearing System  |  RPM=1800  |  fn=150 Hz  |  Fs=1000 Hz\n"
        "[SIMULATED DATA -- Not real sensor measurements]",
        fontsize=11, fontweight="bold",
    )

    for ax, condition in zip(axes, CONDITIONS):
        label, color = condition_styles[condition]
        t    = results[condition]["t"]
        accel = results[condition]["accel"]
        s    = results[condition]["stats"]

        ax.plot(
            t[:display_samples],
            accel[:display_samples],
            color=color, linewidth=1.2, label=label,
        )
        ax.set_title(label, fontsize=10, color=color, fontweight="bold")
        ax.set_ylabel("Acceleration\n(m/s$^2$)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0, color="gray", linewidth=0.8)

        # Annotate RMS value in the corner of each subplot
        ax.text(
            0.99, 0.91,
            f"RMS = {s['rms']:.4f} m/s$^2$   Peak = {s['peak']:.4f} m/s$^2$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.85),
        )

        # Mark the y-scale limits symmetrically at the worst-case signal
        y_lim = max(abs(accel[:display_samples].min()),
                    abs(accel[:display_samples].max())) * 1.15
        ax.set_ylim(-y_lim, y_lim)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {output_path}")


def plot_rms_comparison(results: dict, output_path: str):
    """
    Plot 2: RMS bar chart comparing the three conditions.
    Also overlays Peak and Std Dev for a complete comparison.
    """
    short_labels = ["Healthy", "Slight\nMisalignment", "Severe\nMisalignment"]
    colors       = ["steelblue", "darkorange", "crimson"]
    metrics      = ["rms", "peak", "std"]
    metric_labels = ["RMS", "Peak |accel|", "Std Dev"]

    x      = np.arange(len(CONDITIONS))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        values = [results[c]["stats"][metric] for c in CONDITIONS]
        bars = ax.bar(x + (i - 1) * width, values, width, label=mlabel,
                      color=[c for c in colors], alpha=0.75 + 0.08*i,
                      edgecolor="black", linewidth=0.6)
        # Only annotate the RMS bars to avoid clutter
        if metric == "rms":
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel("Acceleration (m/s$^2$)", fontsize=10)
    ax.set_title(
        "Vibration Metrics by Condition  --  Physics-Based SDOF Model\n"
        "[SIMULATED DATA -- Not real sensor measurements]",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(results[c]["stats"]["peak"] for c in CONDITIONS) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {output_path}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    # 1. Print system parameters
    print_validation_header()

    # 2. Create output directory
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 3. Generate one signal per condition
    print("  Generating validation signals (fixed seeds for reproducibility)...")
    results = validate_all_conditions(fixed_seed_base=999)

    # 4. Print statistics table + distinguishability check
    print_statistics_table(results)

    # 5. Save plots
    print("  Saving validation plots...")
    plot_waveforms(
        results,
        os.path.join(FIGURES_DIR, "validation_waveforms.png"),
    )
    plot_rms_comparison(
        results,
        os.path.join(FIGURES_DIR, "validation_rms_comparison.png"),
    )

    print()
    print("=" * 65)
    print("  Validation COMPLETE. Review the statistics above and")
    print("  the plots saved to: results/figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()
