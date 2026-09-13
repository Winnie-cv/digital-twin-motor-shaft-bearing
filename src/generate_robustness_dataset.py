import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "robustness_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 1000.0
DURATION = 1.0
N = int(FS * DURATION)

CONDITIONS = {
    "healthy": 0,
    "slight_misalignment": 1,
    "severe_misalignment": 2,
}

def generate_signal(condition, rng):

    t = np.arange(N) / FS

    # Vary rotational speed around the nominal 30 Hz
    rotation_frequency = rng.uniform(28.5, 31.5)

    # Small variation in machine response
    amplitude_scale = rng.uniform(0.85, 1.15)

    if condition == "healthy":

        signal = (
            0.50 * amplitude_scale *
            np.sin(2 * np.pi * rotation_frequency * t)
        )

    elif condition == "slight_misalignment":

        signal = (
            0.50 * amplitude_scale *
            np.sin(2 * np.pi * rotation_frequency * t)
            +
            rng.uniform(0.15, 0.28) *
            np.sin(2 * np.pi * 2 * rotation_frequency * t + rng.uniform(0, 2*np.pi))
        )

    else:

        signal = (
            0.50 * amplitude_scale *
            np.sin(2 * np.pi * rotation_frequency * t)
            +
            rng.uniform(0.25, 0.45) *
            np.sin(2 * np.pi * 2 * rotation_frequency * t + rng.uniform(0, 2*np.pi))
            +
            rng.uniform(0.08, 0.22) *
            np.sin(2 * np.pi * 3 * rotation_frequency * t + rng.uniform(0, 2*np.pi))
        )

    # Add broadband measurement/environmental noise
    noise_level = rng.uniform(0.03, 0.12)
    signal += rng.normal(0, noise_level, N)

    # Add another unrelated vibration component
    extra_frequency = rng.uniform(70, 120)
    extra_amplitude = rng.uniform(0.00, 0.12)

    signal += (
        extra_amplitude *
        np.sin(2 * np.pi * extra_frequency * t + rng.uniform(0, 2*np.pi))
    )

    return signal


def main():

    rng = np.random.default_rng(123)

    total = 0

    for condition in CONDITIONS:

        folder = OUTPUT_DIR / condition
        folder.mkdir(parents=True, exist_ok=True)

        for i in range(1, 201):

            signal = generate_signal(condition, rng)

            time = np.arange(N) / FS

            df = pd.DataFrame({
                "time": time,
                "vibration": signal
            })

            df.to_csv(
                folder / f"signal_{i:04d}.csv",
                index=False
            )

            total += 1

        print(f"{condition}: 200 files")

    print()
    print("===== ROBUSTNESS DATASET COMPLETE =====")
    print(f"Total signals: {total}")
    print(f"Sampling rate: {FS} Hz")
    print(f"Duration per signal: {DURATION} s")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
