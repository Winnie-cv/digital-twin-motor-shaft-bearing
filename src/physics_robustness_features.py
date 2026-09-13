import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "physics_robustness"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "physics_robustness_features.csv"

FS = 1000.0

CONDITIONS = {
    "healthy": 0,
    "slight_misalignment": 1,
    "severe_misalignment": 2,
}


def extract_features(csv_file):

    df = pd.read_csv(csv_file)

    signal = df["vibration"].values
    n = len(signal)

    # Time-domain features
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    mean = np.mean(signal)
    std = np.std(signal)

    crest_factor = peak / rms if rms != 0 else 0

    # FFT
    frequencies = np.fft.rfftfreq(n, d=1 / FS)
    spectrum = np.abs(np.fft.rfft(signal)) * 2 / n

    spectrum[0] = 0

    def get_amplitude(target_frequency):

        index = np.argmin(
            np.abs(frequencies - target_frequency)
        )

        return spectrum[index]

    amp_30hz = get_amplitude(30)
    amp_60hz = get_amplitude(60)
    amp_90hz = get_amplitude(90)

    return {
        "rms": rms,
        "peak": peak,
        "mean": mean,
        "std": std,
        "crest_factor": crest_factor,
        "amp_30hz": amp_30hz,
        "amp_60hz": amp_60hz,
        "amp_90hz": amp_90hz,
    }


def main():

    rows = []

    print("==============================================")
    print("PHYSICS ROBUSTNESS FEATURE EXTRACTION")
    print("==============================================")
    print()

    for condition, label in CONDITIONS.items():

        folder = INPUT_DIR / condition

        files = sorted(folder.glob("*.csv"))

        print(f"{condition}: {len(files)} files")

        for csv_file in files:

            features = extract_features(csv_file)

            features["condition"] = condition
            features["label"] = label
            features["file"] = csv_file.name

            rows.append(features)

    dataset = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("==============================================")
    print("FEATURE EXTRACTION COMPLETE")
    print("==============================================")

    print()
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    print()
    print("Class counts:")
    print(dataset.groupby("condition").size())

    print()
    print(f"Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()