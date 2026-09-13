import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "robustness_test"
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

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    mean = np.mean(signal)
    std = np.std(signal)
    crest_factor = peak / rms

    n = len(signal)

    frequencies = np.fft.rfftfreq(n, d=1 / FS)
    spectrum = np.abs(np.fft.rfft(signal)) * 2 / n

    def amplitude_at(freq):

        index = np.argmin(np.abs(frequencies - freq))
        return spectrum[index]

    return {
        "rms": rms,
        "peak": peak,
        "mean": mean,
        "std": std,
        "crest_factor": crest_factor,
        "amp_30hz": amplitude_at(30),
        "amp_60hz": amplitude_at(60),
        "amp_90hz": amplitude_at(90),
    }


def main():

    rows = []

    for condition, label in CONDITIONS.items():

        folder = RAW_DIR / condition
        files = sorted(folder.glob("*.csv"))

        print(f"{condition}: {len(files)} files")

        for csv_file in files:

            features = extract_features(csv_file)

            features["condition"] = condition
            features["label"] = label
            features["file"] = csv_file.name

            rows.append(features)

    dataset = pd.DataFrame(rows)

    output_file = OUTPUT_DIR / "robustness_features.csv"
    dataset.to_csv(output_file, index=False)

    print()
    print("===== ROBUSTNESS FEATURE EXTRACTION COMPLETE =====")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Saved to: {output_file}")
    print()
    print(dataset.groupby("condition").size())


if __name__ == "__main__":
    main()
