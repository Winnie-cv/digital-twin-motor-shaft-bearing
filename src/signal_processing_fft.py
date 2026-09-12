import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

files = {
    "Healthy": PROJECT_ROOT / "data/raw/healthy/signal_0001.csv",
    "Slight Misalignment": PROJECT_ROOT / "data/raw/slight_misalignment/signal_0001.csv",
    "Severe Misalignment": PROJECT_ROOT / "data/raw/severe_misalignment/signal_0001.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "data/processed/fft_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_fft(csv_path):
    data = pd.read_csv(csv_path)

    time = data["time"].to_numpy()
    vibration = data["vibration"].to_numpy()

    # Sampling interval and frequency
    dt = time[1] - time[0]
    fs = 1 / dt

    n = len(vibration)

    # Remove DC component
    vibration = vibration - np.mean(vibration)

    # FFT
    fft_values = np.fft.rfft(vibration)
    frequencies = np.fft.rfftfreq(n, d=dt)

    # Convert FFT magnitude to single-sided amplitude spectrum
    amplitude = (2.0 / n) * np.abs(fft_values)
    amplitude[0] = amplitude[0] / 2

    return time, vibration, frequencies, amplitude, fs


results = {}

for condition, csv_path in files.items():
    time, vibration, frequencies, amplitude, fs = calculate_fft(csv_path)

    results[condition] = {
        "frequencies": frequencies,
        "amplitude": amplitude,
        "fs": fs
    }

    print(f"\n{condition}")
    print(f"Sampling frequency: {fs:.1f} Hz")
    print(f"Samples: {len(vibration)}")
    print(f"Frequency resolution: {fs / len(vibration):.1f} Hz")

    # Find amplitudes near expected rotational harmonics
    for target in [30, 60, 90]:
        index = np.argmin(np.abs(frequencies - target))
        print(
            f"{target:3d} Hz -> amplitude = "
            f"{amplitude[index]:.5f} m/s2"
        )


# Plot all three spectra
plt.figure(figsize=(10, 6))

for condition, result in results.items():
    frequencies = result["frequencies"]
    amplitude = result["amplitude"]

    # Only display 0-150 Hz
    mask = frequencies <= 150

    plt.plot(
        frequencies[mask],
        amplitude[mask],
        label=condition
    )

plt.axvline(30, linestyle="--", alpha=0.5, label="30 Hz (1Ã—)")
plt.axvline(60, linestyle="--", alpha=0.5, label="60 Hz (2Ã—)")
plt.axvline(90, linestyle="--", alpha=0.5, label="90 Hz (3Ã—)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s2)")
plt.title("FFT Spectrum - Motor Shaft Conditions")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_file = OUTPUT_DIR / "fft_validation.png"
plt.savefig(output_file, dpi=200)
plt.show()

print(f"\nFFT validation plot saved to:")
print(output_file)


