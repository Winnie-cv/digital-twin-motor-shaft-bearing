import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "physics_robustness"

FS = 1000.0
DURATION = 1.0
N_SAMPLES = int(FS * DURATION)

N_PER_CLASS = 200

RNG = np.random.default_rng(456)


# ============================================================
# PHYSICS MODEL PARAMETERS
# Same model used for the original dataset
# ============================================================

MASS = 5.0

NOMINAL_FN = 150.0
NOMINAL_ZETA = 0.05


# Misalignment excitation forces
BASE_FORCE_1X = 50.0

SLIGHT_FORCE_2X = 20.0

SEVERE_FORCE_2X = 35.0
SEVERE_FORCE_3X = 15.0


# ============================================================
# MASS-SPRING-DAMPER SIMULATION
# ============================================================

def simulate_signal(condition, rotational_frequency,
                    force_scale, stiffness_scale,
                    damping_scale, noise_level):

    t = np.arange(N_SAMPLES) / FS

    # Natural frequency
    natural_frequency = NOMINAL_FN

    omega_n = 2 * np.pi * natural_frequency

    # Slight variation in stiffness
    k = MASS * omega_n**2 * stiffness_scale

    # Damping
    zeta = NOMINAL_ZETA * damping_scale

    c = 2 * zeta * np.sqrt(k * MASS)

    # --------------------------------------------------------
    # Excitation frequencies
    # --------------------------------------------------------

    f1 = rotational_frequency
    f2 = 2 * rotational_frequency
    f3 = 3 * rotational_frequency

    # Random phase for each harmonic
    phase1 = RNG.uniform(0, 2 * np.pi)
    phase2 = RNG.uniform(0, 2 * np.pi)
    phase3 = RNG.uniform(0, 2 * np.pi)

    # --------------------------------------------------------
    # Force amplitudes
    # --------------------------------------------------------

    force_1x = BASE_FORCE_1X * force_scale

    force_2x = 0.0
    force_3x = 0.0

    if condition == "slight_misalignment":

        force_2x = (
            SLIGHT_FORCE_2X
            * force_scale
            * RNG.uniform(0.90, 1.10)
        )

    elif condition == "severe_misalignment":

        force_2x = (
            SEVERE_FORCE_2X
            * force_scale
            * RNG.uniform(0.90, 1.10)
        )

        force_3x = (
            SEVERE_FORCE_3X
            * force_scale
            * RNG.uniform(0.90, 1.10)
        )

    # --------------------------------------------------------
    # Force signal
    # --------------------------------------------------------

    force = (
        force_1x * np.sin(2 * np.pi * f1 * t + phase1)
        + force_2x * np.sin(2 * np.pi * f2 * t + phase2)
        + force_3x * np.sin(2 * np.pi * f3 * t + phase3)
    )

    # --------------------------------------------------------
    # Steady-state response of:
    #
    # m*x'' + c*x' + k*x = F(t)
    #
    # Calculate each harmonic response analytically.
    # This avoids startup transients.
    # --------------------------------------------------------

    displacement = np.zeros_like(t)

    harmonic_data = [
        (f1, force_1x, phase1),
        (f2, force_2x, phase2),
        (f3, force_3x, phase3),
    ]

    for frequency, force_amplitude, phase in harmonic_data:

        if force_amplitude == 0:
            continue

        omega = 2 * np.pi * frequency

        denominator = np.sqrt(
            (k - MASS * omega**2)**2
            + (c * omega)**2
        )

        displacement_amplitude = force_amplitude / denominator

        displacement_phase = np.arctan2(
            c * omega,
            k - MASS * omega**2
        )

        displacement += (
            displacement_amplitude
            * np.sin(
                omega * t
                + phase
                - displacement_phase
            )
        )

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    acceleration = np.zeros_like(t)

    for frequency, force_amplitude, phase in harmonic_data:

        if force_amplitude == 0:
            continue

        omega = 2 * np.pi * frequency

        denominator = np.sqrt(
            (k - MASS * omega**2)**2
            + (c * omega)**2
        )

        displacement_amplitude = force_amplitude / denominator

        displacement_phase = np.arctan2(
            c * omega,
            k - MASS * omega**2
        )

        acceleration += (
            -omega**2
            * displacement_amplitude
            * np.sin(
                omega * t
                + phase
                - displacement_phase
            )
        )

    # --------------------------------------------------------
    # Add measurement noise
    # --------------------------------------------------------

    noise = RNG.normal(
        0,
        noise_level,
        N_SAMPLES
    )

    acceleration = acceleration + noise

    return t, acceleration


# ============================================================
# SAVE CSV
# ============================================================

def save_signal(t, acceleration, condition, index):

    folder = OUTPUT_DIR / condition

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = folder / f"signal_{index:04d}.csv"

    df = pd.DataFrame({
        "time": t,
        "vibration": acceleration
    })

    df.to_csv(
        filename,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================================")
    print("PHYSICS-BASED ROBUSTNESS DATASET")
    print("==============================================")
    print()

    conditions = [
        "healthy",
        "slight_misalignment",
        "severe_misalignment"
    ]

    total = 0

    for condition in conditions:

        print(f"Generating {condition}...")

        for i in range(1, N_PER_CLASS + 1):

            # ------------------------------------------------
            # Operating-condition variations
            # ------------------------------------------------

            rotational_frequency = RNG.uniform(
                28.5,
                31.5
            )

            force_scale = RNG.uniform(
                0.85,
                1.15
            )

            stiffness_scale = RNG.uniform(
                0.95,
                1.05
            )

            damping_scale = RNG.uniform(
                0.90,
                1.10
            )

            noise_level = RNG.uniform(
                0.03,
                0.08
            )

            # ------------------------------------------------
            # Simulate
            # ------------------------------------------------

            t, acceleration = simulate_signal(
                condition,
                rotational_frequency,
                force_scale,
                stiffness_scale,
                damping_scale,
                noise_level
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            save_signal(
                t,
                acceleration,
                condition,
                i
            )

            total += 1

        print(
            f"{condition}: "
            f"{N_PER_CLASS} files"
        )

    print()
    print("==============================================")
    print("PHYSICS ROBUSTNESS DATASET COMPLETE")
    print("==============================================")

    print()
    print(f"Total signals: {total}")
    print(f"Sampling rate: {FS} Hz")
    print(f"Duration: {DURATION} seconds")

    print()
    print(f"Saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()