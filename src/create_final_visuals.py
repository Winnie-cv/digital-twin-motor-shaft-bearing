import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "ml_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "final_visuals"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. WAVEFORM COMPARISON
# ============================================================

conditions = [
    ("healthy", "Healthy"),
    ("slight_misalignment", "Slight Misalignment"),
    ("severe_misalignment", "Severe Misalignment")
]

plt.figure(figsize=(10, 6))

for folder, label in conditions:
    file = RAW_DIR / folder / "signal_0001.csv"
    df = pd.read_csv(file)

    plt.plot(
        df["time"].values[:300],
        df["vibration"].values[:300],
        label=label
    )

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Simulated Vibration Waveforms")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "waveform_comparison.png", dpi=300)
plt.close()

# ============================================================
# 2. FFT COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

for folder, label in conditions:
    file = RAW_DIR / folder / "signal_0001.csv"
    df = pd.read_csv(file)

    signal = df["vibration"].values
    fs = 1000.0
    n = len(signal)

    signal = signal - np.mean(signal)

    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, 1 / fs)

    amplitude = (2.0 / n) * np.abs(fft_values)

    mask = frequencies <= 120

    plt.plot(
        frequencies[mask],
        amplitude[mask],
        label=label
    )

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.title("FFT Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "fft_comparison.png", dpi=300)
plt.close()

# ============================================================
# 3. RMS COMPARISON
# ============================================================

df = pd.read_csv(FEATURE_FILE)

rms_values = df.groupby("condition")["rms"].mean()

order = [
    "healthy",
    "slight_misalignment",
    "severe_misalignment"
]

rms_values = rms_values.reindex(order)

plt.figure(figsize=(8, 6))

plt.bar(
    ["Healthy", "Slight\nMisalignment", "Severe\nMisalignment"],
    rms_values.values
)

plt.ylabel("RMS Acceleration (m/s²)")
plt.title("Average RMS Vibration by Condition")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "rms_comparison.png", dpi=300)
plt.close()

# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

FEATURES = [
    "rms",
    "peak",
    "mean",
    "std",
    "crest_factor",
    "amp_30hz",
    "amp_60hz",
    "amp_90hz"
]

X = df[FEATURES]
y = df["label"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values()

plt.figure(figsize=(9, 6))

plt.barh(
    importance.index,
    importance.values
)

plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.grid(axis="x")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
plt.close()

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

predictions = model.predict(X)

plt.figure(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y,
    predictions,
    display_labels=[
        "Healthy",
        "Slight",
        "Severe"
    ],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

print("==============================================")
print("FINAL VISUALIZATIONS CREATED")
print("==============================================")
print()
print(f"Output folder: {OUTPUT_DIR}")
print()
print("Created:")
print("1. waveform_comparison.png")
print("2. fft_comparison.png")
print("3. rms_comparison.png")
print("4. feature_importance.png")
print("5. confusion_matrix.png")
print()
print("Visualization stage COMPLETE.")