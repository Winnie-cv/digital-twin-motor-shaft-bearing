import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "raw"
ROBUST_DIR = PROJECT_ROOT / "data" / "robustness_test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 1000.0

CONDITIONS = {
    "healthy": 0,
    "slight_misalignment": 1,
    "severe_misalignment": 2,
}


def extract_features(csv_path):

    df = pd.read_csv(csv_path)

    signal = df["vibration"].values
    n = len(signal)

    # -----------------------------
    # Time-domain features
    # -----------------------------

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    mean = np.mean(signal)
    std = np.std(signal)
    crest_factor = peak / rms if rms != 0 else 0

    # -----------------------------
    # FFT
    # -----------------------------

    frequencies = np.fft.rfftfreq(n, d=1 / FS)

    spectrum = np.abs(np.fft.rfft(signal)) * 2 / n

    # Ignore DC component
    spectrum[0] = 0

    # -----------------------------
    # Find dominant rotational
    # frequency
    #
    # Expected machine speed:
    # approximately 30 Hz
    # -----------------------------

    search_mask = (frequencies >= 25) & (frequencies <= 35)

    search_indices = np.where(search_mask)[0]

    dominant_index = search_indices[
        np.argmax(spectrum[search_indices])
    ]

    rotational_frequency = frequencies[dominant_index]

    # -----------------------------
    # Speed-aware harmonic amplitudes
    # -----------------------------

    def harmonic_amplitude(harmonic):

        target_frequency = rotational_frequency * harmonic

        # Search within +/- 2 Hz
        # around expected harmonic
        mask = (
            (frequencies >= target_frequency - 2)
            &
            (frequencies <= target_frequency + 2)
        )

        indices = np.where(mask)[0]

        if len(indices) == 0:
            return 0

        return np.max(spectrum[indices])

    amp_1x = harmonic_amplitude(1)
    amp_2x = harmonic_amplitude(2)
    amp_3x = harmonic_amplitude(3)

    # -----------------------------
    # Harmonic ratios
    # -----------------------------

    ratio_2x_1x = amp_2x / amp_1x if amp_1x != 0 else 0
    ratio_3x_1x = amp_3x / amp_1x if amp_1x != 0 else 0

    return {
        "rms": rms,
        "peak": peak,
        "mean": mean,
        "std": std,
        "crest_factor": crest_factor,

        "rotational_frequency": rotational_frequency,

        "amp_1x": amp_1x,
        "amp_2x": amp_2x,
        "amp_3x": amp_3x,

        "ratio_2x_1x": ratio_2x_1x,
        "ratio_3x_1x": ratio_3x_1x,
    }


def process_dataset(input_dir, output_file):

    rows = []

    for condition, label in CONDITIONS.items():

        folder = input_dir / condition

        files = sorted(folder.glob("*.csv"))

        print(f"{condition}: {len(files)} files")

        for csv_file in files:

            features = extract_features(csv_file)

            features["condition"] = condition
            features["label"] = label
            features["file"] = csv_file.name

            rows.append(features)

    dataset = pd.DataFrame(rows)

    dataset.to_csv(output_file, index=False)

    print()
    print(f"Saved: {output_file}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    return dataset


def main():

    print("==============================================")
    print("VERSION 2: SPEED-AWARE FEATURE EXTRACTION")
    print("==============================================")
    print()

    # Original training dataset
    print("Processing ORIGINAL training dataset...")
    train_output = OUTPUT_DIR / "speed_aware_train.csv"

    train_df = process_dataset(
        TRAIN_DIR,
        train_output
    )

    print()

    # Robustness dataset
    print("Processing ROBUSTNESS dataset...")
    robustness_output = OUTPUT_DIR / "speed_aware_robustness.csv"

    robustness_df = process_dataset(
        ROBUST_DIR,
        robustness_output
    )

    print()
    print("==============================================")
    print("SPEED-AWARE FEATURE EXTRACTION COMPLETE")
    print("==============================================")

    print()
    print("Training dataset:")
    print(train_df.groupby("condition").size())

    print()
    print("Robustness dataset:")
    print(robustness_df.groupby("condition").size())

    print()
    print("Example rotational frequencies:")

    print(
        robustness_df[
            ["condition", "rotational_frequency"]
        ].groupby("condition").mean()
    )


if __name__ == "__main__":
    main()