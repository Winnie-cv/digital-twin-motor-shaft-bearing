"""
src/simulation/generate_dataset.py
====================================
Physics-based dataset generator for the Digital Twin project.

Generates 500 physics-based vibration signals for each of the three
operating conditions and saves them to data/raw/<condition>/.

DATA SEPARATION:
  NEW physics-based dataset  -->  data/raw/healthy/
                                  data/raw/slight_misalignment/
                                  data/raw/severe_misalignment/

  OLD prototype (sine-wave)  -->  data/raw/prototype_synthetic/healthy/
                                  data/raw/prototype_synthetic/slight_misalignment/
                                  data/raw/prototype_synthetic/severe_misalignment/

This script does NOT touch or overwrite the prototype_synthetic directory.

OUTPUT FORMAT (per CSV file):
  Two columns, 1000 rows of data + 1 header row:
    time      -- time axis (s), uniform from 0.000 to 0.999 s
    vibration -- simulated lateral acceleration (m/s^2)

REPRODUCIBILITY:
  Each signal uses seed = condition_index * SIGNALS_PER_CONDITION + sample_index,
  guaranteeing the full dataset is reproducible across runs.

HOW TO RUN (from the Digital_Twin_Project root directory):
    python src/simulation/generate_dataset.py
"""

import sys
import os
import csv
import time as time_module

import numpy as np

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
    SIGNALS_PER_CONDITION,
    NUM_SAMPLES,
    MASS_KG,
    NATURAL_FREQUENCY_HZ,
    DAMPING_RATIO,
    STIFFNESS_NM,
    DAMPING_COEFF_NSM,
    SAMPLING_RATE_HZ,
    SIGNAL_DURATION_S,
)
from src.simulation.signal_generator import generate_vibration

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR       = os.path.join(_PROJECT_ROOT, "data", "raw")
PROTOTYPE_DIR      = os.path.join(DATA_RAW_DIR, "prototype_synthetic")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def create_output_directories():
    """Create data/raw/<condition>/ directories. Prototype dir is NOT touched."""
    print("  Creating physics-based dataset directories...")
    for condition in CONDITIONS:
        folder = os.path.join(DATA_RAW_DIR, condition)
        os.makedirs(folder, exist_ok=True)
        print(f"    [OK] {folder}")


def save_signal_as_csv(t: np.ndarray, vibration: np.ndarray, filepath: str):
    """
    Save one vibration signal to a CSV file.

    Columns: time (s), vibration (m/s^2 acceleration)
    """
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "vibration"])
        for t_val, v_val in zip(t, vibration):
            writer.writerow([f"{t_val:.6f}", f"{v_val:.8f}"])


def compute_rms(arr: np.ndarray) -> float:
    """Return RMS value of an array."""
    return float(np.sqrt(np.mean(arr ** 2)))


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset():
    """
    Generate 500 physics-based vibration signals for each condition,
    save as CSV files, then run a full verification report.
    """
    print("\n" + "=" * 65)
    print("  Digital Twin -- Physics-Based Dataset Generator")
    print("=" * 65)

    print("\n  MODEL PARAMETERS (ASSUMED -- not from real measurements):")
    print(f"    m   = {MASS_KG} kg             (effective mass)")
    print(f"    fn  = {NATURAL_FREQUENCY_HZ} Hz            (natural frequency)")
    print(f"    z   = {DAMPING_RATIO}           (damping ratio)")
    print(f"    k   = {STIFFNESS_NM/1e6:.4f} MN/m    (stiffness, derived)")
    print(f"    c   = {DAMPING_COEFF_NSM:.2f} N.s/m  (damping coeff., derived)")

    print(f"\n  DATASET SETTINGS:")
    print(f"    Conditions          : {CONDITIONS}")
    print(f"    Signals per cond.   : {SIGNALS_PER_CONDITION}")
    print(f"    Samples per signal  : {NUM_SAMPLES}")
    print(f"    Sampling rate       : {SAMPLING_RATE_HZ} Hz")
    print(f"    Signal duration     : {SIGNAL_DURATION_S} s")
    print(f"    Total files         : {len(CONDITIONS) * SIGNALS_PER_CONDITION}")
    print(f"    Output directory    : {DATA_RAW_DIR}")

    # ------------------------------------------------------------------
    # Step 1 — Confirm prototype dataset is safe before doing anything
    # ------------------------------------------------------------------
    print("\n[0/4] Verifying prototype dataset is present and untouched...")
    proto_ok = True
    for condition in CONDITIONS:
        proto_cond = os.path.join(PROTOTYPE_DIR, condition)
        if os.path.isdir(proto_cond):
            n = len([f for f in os.listdir(proto_cond) if f.endswith(".csv")])
            status = "[OK]" if n == SIGNALS_PER_CONDITION else f"[WARN: {n} files]"
            print(f"    {status} prototype_synthetic/{condition}: {n} CSV files")
            if n != SIGNALS_PER_CONDITION:
                proto_ok = False
        else:
            print(f"    [WARN] prototype_synthetic/{condition} directory not found!")
            proto_ok = False

    if not proto_ok:
        print("\n  [WARNING] Prototype dataset may be incomplete. "
              "Proceeding with physics-based generation anyway.")
    else:
        print("    Prototype dataset intact. Will not be modified.")

    # ------------------------------------------------------------------
    # Step 2 — Create output directories
    # ------------------------------------------------------------------
    print("\n[1/4] Creating output directories...")
    create_output_directories()

    # ------------------------------------------------------------------
    # Step 3 — Generate signals
    # ------------------------------------------------------------------
    print("\n[2/4] Generating physics-based signals...")
    start_time = time_module.time()

    # Track per-condition RMS for the inline progress report
    condition_rms_samples = {c: [] for c in CONDITIONS}

    for cond_idx, condition in enumerate(CONDITIONS):
        condition_dir = os.path.join(DATA_RAW_DIR, condition)
        print(f"\n  [{cond_idx+1}/3] Condition: '{condition}'")
        print(f"    Generating {SIGNALS_PER_CONDITION} signals...", end="", flush=True)

        for sample_idx in range(SIGNALS_PER_CONDITION):
            # Unique, reproducible seed for every (condition, sample) pair
            seed = cond_idx * SIGNALS_PER_CONDITION + sample_idx

            # Generate physics-based vibration signal
            t, vibration = generate_vibration(condition, seed=seed)

            # Collect RMS from a few samples for the progress report
            if sample_idx < 20:
                condition_rms_samples[condition].append(
                    compute_rms(vibration)
                )

            # Save to CSV
            filename = f"signal_{sample_idx + 1:04d}.csv"
            filepath = os.path.join(condition_dir, filename)
            save_signal_as_csv(t, vibration, filepath)

            # Progress dots every 100 signals
            if (sample_idx + 1) % 100 == 0:
                print(f" {sample_idx + 1}", end="", flush=True)

        rms_mean = np.mean(condition_rms_samples[condition])
        print(f"\n    [DONE] {SIGNALS_PER_CONDITION} files saved. "
              f"Mean RMS (first 20 samples): {rms_mean:.4f} m/s^2")

    elapsed = time_module.time() - start_time
    print(f"\n  Total generation time: {elapsed:.2f} seconds")

    # ------------------------------------------------------------------
    # Step 4 — Full verification
    # ------------------------------------------------------------------
    print("\n[3/4] Running full verification...")
    all_passed = verify_dataset()

    # ------------------------------------------------------------------
    # Step 5 — Re-confirm prototype is still untouched
    # ------------------------------------------------------------------
    print("\n[4/4] Re-confirming prototype_synthetic dataset is intact...")
    for condition in CONDITIONS:
        proto_cond = os.path.join(PROTOTYPE_DIR, condition)
        n = len([f for f in os.listdir(proto_cond) if f.endswith(".csv")]) \
            if os.path.isdir(proto_cond) else 0
        status = "[OK]" if n == SIGNALS_PER_CONDITION else "[FAIL]"
        print(f"    {status} prototype_synthetic/{condition}: {n} files")

    print("\n" + "=" * 65)
    if all_passed:
        print("  Dataset generation and verification COMPLETE.")
        print("  Physics-based dataset: data/raw/")
        print("  Prototype dataset:     data/raw/prototype_synthetic/")
    else:
        print("  [WARNING] Some verification checks failed. See above.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Verification function
# ---------------------------------------------------------------------------

def verify_dataset(target_dir: str = None, label: str = "physics-based") -> bool:
    """
    Verify the generated dataset.

    Checks:
      1. Correct file count per condition
      2. Correct number of rows (NUM_SAMPLES) in spot-checked files
      3. Correct CSV header (time, vibration)
      4. No NaN or Inf values (checks 5 files per condition)
      5. RMS statistics (mean, std, min, max over 20 random files)
      6. Distinguishability: RMS increases with severity

    Parameters
    ----------
    target_dir : str, optional
        Directory to verify. Defaults to DATA_RAW_DIR.
    label : str
        Label printed in the report header.

    Returns
    -------
    bool : True if all checks passed, False otherwise.
    """
    if target_dir is None:
        target_dir = DATA_RAW_DIR

    all_passed = True
    rng_verify = np.random.default_rng(seed=42)

    print(f"\n  --- Verifying {label} dataset: {target_dir} ---\n")
    print(f"  {'Condition':<25} {'Files':>6} {'Count':>6}  "
          f"{'Rows':>6}  {'Header':>8}  {'NaN/Inf':>8}  Status")
    print("  " + "-" * 72)

    condition_rms_means = []

    for condition in CONDITIONS:
        cond_dir  = os.path.join(target_dir, condition)
        csv_files = sorted([f for f in os.listdir(cond_dir) if f.endswith(".csv")])
        n_files   = len(csv_files)

        # --- Check 1: File count ---
        count_ok = n_files == SIGNALS_PER_CONDITION

        # --- Check 2 & 3: Row count and header (spot-check first file) ---
        rows_ok  = False
        header_ok = False
        if csv_files:
            first_path = os.path.join(cond_dir, csv_files[0])
            with open(first_path, "r") as f:
                reader  = csv.reader(f)
                header  = next(reader)
                row_cnt = sum(1 for _ in reader)
            rows_ok   = (row_cnt == NUM_SAMPLES)
            header_ok = (header == ["time", "vibration"])

        # --- Check 4: NaN/Inf in 5 random files ---
        nan_inf_ok = True
        check_files = rng_verify.choice(csv_files, size=min(5, n_files), replace=False)
        for fname in check_files:
            fpath = os.path.join(cond_dir, fname)
            with open(fpath, "r") as f:
                reader = csv.reader(f)
                next(reader)   # skip header
                vals = [float(row[1]) for row in reader]
            arr = np.array(vals)
            if not np.all(np.isfinite(arr)):
                nan_inf_ok = False
                break

        # --- Check 5: RMS statistics over 20 random files ---
        stat_files = rng_verify.choice(csv_files, size=min(20, n_files), replace=False)
        rms_values = []
        for fname in stat_files:
            fpath = os.path.join(cond_dir, fname)
            with open(fpath, "r") as f:
                reader = csv.reader(f)
                next(reader)
                vals = [float(row[1]) for row in reader]
            rms_values.append(compute_rms(np.array(vals)))

        rms_mean = float(np.mean(rms_values))
        rms_std  = float(np.std(rms_values))
        rms_min  = float(np.min(rms_values))
        rms_max  = float(np.max(rms_values))
        condition_rms_means.append(rms_mean)

        # --- Overall status for this condition ---
        cond_ok = count_ok and rows_ok and header_ok and nan_inf_ok
        if not cond_ok:
            all_passed = False

        status = "PASS" if cond_ok else "FAIL"
        row_str    = str(NUM_SAMPLES) if rows_ok   else f"FAIL({row_cnt})"
        header_str = "OK"            if header_ok  else "FAIL"
        nan_str    = "OK"            if nan_inf_ok else "FAIL"

        print(f"  {condition:<25} {n_files:>6} {'OK' if count_ok else 'FAIL':>6}  "
              f"{row_str:>6}  {header_str:>8}  {nan_str:>8}  {status}")
        print(f"    RMS (20 samples): mean={rms_mean:.4f}  std={rms_std:.4f}  "
              f"min={rms_min:.4f}  max={rms_max:.4f}  m/s^2")

    # --- Check 6: Monotonic severity ordering ---
    print()
    if len(condition_rms_means) == 3:
        h, sl, sv = condition_rms_means
        mono_ok = (h < sl < sv)
        print(f"  DISTINGUISHABILITY CHECK (RMS mean over 20 samples):")
        print(f"    Healthy          = {h:.4f} m/s^2  (baseline)")
        print(f"    Slight misalign. = {sl:.4f} m/s^2  ({sl/h:.2f}x healthy)")
        print(f"    Severe misalign. = {sv:.4f} m/s^2  ({sv/h:.2f}x healthy)")
        if mono_ok:
            print(f"    [PASS] RMS is monotonically increasing with severity.")
        else:
            print(f"    [FAIL] RMS is NOT monotonically increasing. Check parameters.")
            all_passed = False

    print()
    if all_passed:
        print(f"  [PASS] All {label} dataset verification checks PASSED.")
    else:
        print(f"  [FAIL] Some {label} dataset checks FAILED. See above.")

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_dataset()
