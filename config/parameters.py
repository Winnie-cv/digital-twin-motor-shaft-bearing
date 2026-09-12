"""
config/parameters.py
====================
Central configuration file for the Digital Twin Motor-Shaft-Bearing System project.

All engineering parameters are defined here and imported by other modules.
DO NOT hard-code these values in other files — always import from this module.

PROJECT:  Digital Twin of a Motor-Shaft-Bearing System
          for Misalignment Detection Using Machine Learning
STAGE:    Stage 2 (v2) -- Physics-Based Reduced-Order Dynamic Model

IMPORTANT DISCLAIMER:
---------------------
This project uses a SIMPLIFIED REDUCED-ORDER DYNAMIC MODEL for educational
purposes. It is NOT a full finite-element or multi-body simulation.

The signals represent a SYNTHETIC approximation of what an accelerometer
mounted on the bearing housing might measure on a real rotating system.
They are NOT real sensor measurements.

Parameter values marked [ASSUMED] are illustrative values chosen to produce
a physically stable and meaningful simulation. They are NOT measured from
the real SolidWorks assembly and should NOT be cited as real machine data.

MECHANICAL REFERENCE (from SolidWorks CAD):
  - Shaft diameter  : 50 mm
  - Bearing bore    : 50 mm
  - Bearing OD      : 90 mm
  - Bearing width   : 20 mm
"""

import numpy as np

# ===========================================================================
# SECTION 1 — MOTOR / SHAFT OPERATING PARAMETERS
# ===========================================================================

RPM = 1800
"""Rotational speed of the motor shaft (revolutions per minute)."""

ROTATIONAL_FREQUENCY_HZ = RPM / 60          # = 30 Hz
"""
Fundamental rotational frequency of the shaft (Hz).
  Formula : f_rot = RPM / 60 = 1800 / 60 = 30 Hz
  Physical: One vibration cycle per shaft revolution (1x rev).
            Dominant frequency in a healthy vibration spectrum.
"""

SECOND_HARMONIC_HZ = ROTATIONAL_FREQUENCY_HZ * 2   # = 60 Hz
"""
Second harmonic of the rotational frequency (2x rev = 60 Hz).
  Physical: Angular misalignment excites the shaft twice per revolution.
            A significant 2x/1x ratio is a classical misalignment indicator.
"""

THIRD_HARMONIC_HZ = ROTATIONAL_FREQUENCY_HZ * 3    # = 90 Hz
"""
Third harmonic (3x rev = 90 Hz).
  Physical: Severe misalignment and non-linear bearing loading can excite
            higher harmonics. Present only in severe misalignment condition.
"""

# ---------------------------------------------------------------------------
# Shaft / Bearing dimensions  (informational — from SolidWorks model)
# ---------------------------------------------------------------------------
SHAFT_DIAMETER_MM  = 50     # mm
BEARING_BORE_MM    = 50     # mm
BEARING_OD_MM      = 90     # mm
BEARING_WIDTH_MM   = 20     # mm


# ===========================================================================
# SECTION 2 — SIMULATION TIME / SAMPLING PARAMETERS
# ===========================================================================

SAMPLING_RATE_HZ = 1000
"""
Sampling rate (samples per second).
  Nyquist: max frequency of interest = 90 Hz -> min Fs = 180 Hz.
  1000 Hz provides ~11x oversampling for clean waveform capture.
"""

SIGNAL_DURATION_S = 1.0
"""Duration of each simulated vibration signal (seconds)."""

NUM_SAMPLES = int(SAMPLING_RATE_HZ * SIGNAL_DURATION_S)
"""
Total number of output samples per signal.
  N = Fs x T = 1000 x 1.0 = 1000 samples
"""

# ===========================================================================
# SECTION 3 — SDOF PHYSICS MODEL PARAMETERS  [ALL VALUES: ASSUMED]
# ===========================================================================
#
# The dynamic equation of motion solved by signal_generator.py is:
#
#     m * x''(t)  +  c * x'(t)  +  k * x(t)  =  F(t)
#
# where:
#   x    = lateral displacement of the shaft centre (m)
#   x'   = velocity (m/s)
#   x''  = acceleration (m/s^2)
#   m    = effective system mass (kg)
#   c    = viscous damping coefficient (N.s/m)
#   k    = effective stiffness of shaft + bearing assembly (N/m)
#   F(t) = time-varying excitation force from rotation + misalignment (N)
#
# The simulated sensor output is the shaft acceleration x''(t), which
# approximates what a piezoelectric accelerometer mounted on the bearing
# housing would measure.
#
# PARAMETER SELECTION RATIONALE:
#   A pre-implementation analytical sweep was performed over fn = 100 to
#   250 Hz. fn = 150 Hz was selected because:
#     - All forcing frequencies (30, 60, 90 Hz) remain well below resonance
#       (frequency ratio r_max = 90/150 = 0.60).
#     - At fn = 100 Hz, the 90 Hz excitation (r=0.90) caused near-resonance
#       amplification that made the severe signal ~12x the healthy signal,
#       which is not realistic for this magnitude of misalignment.
#     - fn = 150 Hz gives a physically reasonable severity gradient:
#       healthy : slight : severe RMS ~ 1 : 2.1 : 5.2
#     - Resulting accelerations (~0.3 to ~1.5 m/s^2 RMS) are consistent
#       with light industrial machinery vibration levels.
#
# ---------------------------------------------------------------------------

# [ASSUMED] Effective system mass
# Represents the combined inertia of the rotor, shaft stub, and coupling
# that loads the bearing in the radial direction.
# A 5 kg value is representative of a small laboratory motor assembly.
# Source: engineering judgement, NOT extracted from SolidWorks mass properties.
MASS_KG = 5.0
"""[ASSUMED] Effective dynamic mass (kg). See section header for rationale."""

# [ASSUMED] Natural frequency
# Chosen from a pre-implementation parameter sweep (see implementation plan).
# At fn = 150 Hz, all three excitation frequencies are sub-resonant (r < 0.6),
# producing a stable and physically meaningful dynamic response.
NATURAL_FREQUENCY_HZ = 150.0
"""[ASSUMED] System natural frequency (Hz). Derived k and c are computed below."""

# [ASSUMED] Damping ratio (dimensionless)
# 5% critical damping (zeta = 0.05) is typical for a lightly damped steel
# shaft in rolling-element bearings with standard lubrication.
# A well-lubricated ball bearing assembly usually has zeta in the range 0.02-0.08.
DAMPING_RATIO = 0.05
"""[ASSUMED] Viscous damping ratio zeta (dimensionless, 0 < zeta < 1)."""

# --- Derived structural parameters (calculated, not assumed) ---
_omega_n = 2.0 * np.pi * NATURAL_FREQUENCY_HZ          # rad/s

STIFFNESS_NM = MASS_KG * (_omega_n ** 2)
"""
Effective shaft/bearing stiffness k (N/m).
  Formula  : k = m * omega_n^2 = m * (2*pi*fn)^2
  Value    : {:.0f} N/m = {:.3f} MN/m
  Physical : Combined lateral stiffness of the shaft in bending and the
             bearing radial stiffness, lumped into a single spring.
             [DERIVED from ASSUMED m and fn]
""".format(STIFFNESS_NM, STIFFNESS_NM / 1e6)

DAMPING_COEFF_NSM = 2.0 * DAMPING_RATIO * np.sqrt(STIFFNESS_NM * MASS_KG)
"""
Viscous damping coefficient c (N.s/m).
  Formula  : c = 2 * zeta * sqrt(k * m) = 2 * zeta * m * omega_n
  Value    : {:.2f} N.s/m
  Physical : Represents energy dissipation from bearing film, material
             hysteresis, and structural coupling losses.
             [DERIVED from ASSUMED zeta, m, fn]
""".format(DAMPING_COEFF_NSM)


# ===========================================================================
# SECTION 4 — EXCITATION FORCE MODEL  [ALL VALUES: ASSUMED]
# ===========================================================================
#
# F(t) is constructed as a superposition of harmonic force components.
# Each component represents a physical excitation mechanism:
#
#   F1 (1x rev, 30 Hz) -- Residual shaft imbalance force.
#       Present in ALL conditions. Even a well-balanced shaft has some
#       residual imbalance from manufacturing tolerances.
#
#   F2 (2x rev, 60 Hz) -- Misalignment-induced force.
#       Angular or parallel misalignment causes the shaft to deflect twice
#       per revolution, exciting the 2x harmonic.
#
#   F3 (3x rev, 90 Hz) -- Severe misalignment / non-linear bearing force.
#       At large misalignment angles, non-linear bearing loading generates
#       higher harmonic content. Included only for severe condition.
#
# FORCE VALUES: These amplitudes are ASSUMED for the educational model.
# They produce physically plausible relative severity levels but are NOT
# derived from imbalance grade standards or bearing load calculations.
# ---------------------------------------------------------------------------

# --- Healthy condition ---
FORCE_HEALTHY_1X_N  = 50.0
"""[ASSUMED] Imbalance force at 1x rev (30 Hz) for healthy condition (N)."""

# --- Slight misalignment ---
FORCE_SLIGHT_1X_N   = 50.0
"""[ASSUMED] Imbalance force at 1x rev (30 Hz), slight misalignment (N)."""

FORCE_SLIGHT_2X_N   = 20.0
"""
[ASSUMED] Misalignment force at 2x rev (60 Hz), slight misalignment (N).
  2x/1x force ratio = 20/50 = 0.40 -- moderate misalignment indicator.
"""

# --- Severe misalignment ---
FORCE_SEVERE_1X_N   = 50.0
"""[ASSUMED] Imbalance force at 1x rev (30 Hz), severe misalignment (N)."""

FORCE_SEVERE_2X_N   = 35.0
"""
[ASSUMED] Misalignment force at 2x rev (60 Hz), severe misalignment (N).
  2x/1x force ratio = 35/50 = 0.70 -- strong misalignment indicator.
"""

FORCE_SEVERE_3X_N   = 15.0
"""
[ASSUMED] Non-linear bearing force at 3x rev (90 Hz), severe misalignment (N).
  3x component appears only at severe misalignment.
"""


# ===========================================================================
# SECTION 5 — SAMPLE VARIABILITY PARAMETERS
# ===========================================================================
#
# To ensure the 500 signals per condition are NOT identical, each sample
# introduces small random variations that simulate real-world variability:
#   - Random phase offset on each harmonic (simulates measurement start time)
#   - Small amplitude jitter (+/- AMP_JITTER_FRACTION of the nominal force)
#   - Additive Gaussian measurement noise on the final acceleration signal
# ---------------------------------------------------------------------------

AMP_JITTER_FRACTION = 0.10
"""
Random amplitude variation per signal (+/- 10% of nominal force).
Simulates run-to-run variability in shaft imbalance and load.
[ASSUMED: educational model only]
"""

NOISE_STD_MS2 = 0.02
"""
Standard deviation of Gaussian sensor noise (m/s^2).
Simulates accelerometer electronic noise and environmental background.
Kept small (~7% of healthy RMS) so conditions remain distinguishable.
[ASSUMED: educational model only]
"""


# ===========================================================================
# SECTION 6 — DATASET PARAMETERS
# ===========================================================================

SIGNALS_PER_CONDITION = 500
"""Number of physics-based signals to generate per condition."""

CONDITIONS = ["healthy", "slight_misalignment", "severe_misalignment"]
"""List of all simulated operating conditions."""

# Dataset directory names
DATASET_PHYSICS_DIR   = "data/raw"
"""Root directory for the new physics-based dataset."""

DATASET_PROTOTYPE_DIR = "data/raw/prototype_synthetic"
"""Root directory where the original sine-wave prototype dataset is preserved."""


# ===========================================================================
# SECTION 7 — LEGACY SINE-WAVE AMPLITUDE CONSTANTS
# (Kept for backward compatibility with prototype_synthetic dataset)
# [DEPRECATED — not used by the physics model]
# ===========================================================================

HEALTHY_AMP_1X = 1.0
SLIGHT_AMP_1X  = 1.0
SLIGHT_AMP_2X  = 0.4
SEVERE_AMP_1X  = 1.0
SEVERE_AMP_2X  = 0.7
SEVERE_AMP_3X  = 0.3
NOISE_STD      = 0.05   # legacy noise value (m/s^2)
