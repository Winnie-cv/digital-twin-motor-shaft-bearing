"""
src/simulation/signal_generator.py
====================================
Physics-based vibration signal generator for the Digital Twin project.

Solves a single-degree-of-freedom (SDOF) equation of motion to simulate
the lateral vibration of a rotating shaft in its bearing housing.

EQUATION OF MOTION:
-------------------
    m * x''(t)  +  c * x'(t)  +  k * x(t)  =  F(t)

    m   = effective system mass (kg)
    c   = viscous damping coefficient (N.s/m)
    k   = effective shaft/bearing stiffness (N/m)
    x   = lateral displacement of shaft centre (m)
    x'  = velocity (m/s)
    x'' = acceleration (m/s^2)
    F   = time-varying excitation force (N)

STEADY-STATE INITIAL CONDITIONS (important engineering detail):
---------------------------------------------------------------
The ODE is initialised at the analytical steady-state solution rather
than at rest (x=0, v=0). Starting from rest would introduce a free-
vibration transient at the natural frequency (fn = 150 Hz) that decays
with time constant tau = 1/(zeta*omega_n) ~= 21 ms. This transient would
distort the signal statistics and make the healthy condition appear to have
higher RMS than it physically should (because the transient at 150 Hz
temporarily dominates).

By starting at the superposition of all steady-state particular solutions,
the response is immediately in the correct periodic regime with no artefacts.

ANALYTICAL STEADY-STATE SOLUTION (for one harmonic F0*sin(omega*t + phi)):
    X_ss(t) = H * F0/k * sin(omega*t + phi + psi)
    where:
        r    = omega / omega_n  (frequency ratio)
        H    = 1 / sqrt((1-r^2)^2 + (2*zeta*r)^2)  (magnification factor)
        psi  = atan2(-2*zeta*r, 1-r^2)               (phase lag)
    velocity:
        V_ss(t) = H * F0/k * omega * cos(omega*t + phi + psi)

For multiple harmonics, the initial x(0) and v(0) are the sum of each
component's steady-state value at t=0.

SIMULATED SENSOR OUTPUT:
-------------------------
Returns ACCELERATION (x'', m/s^2), representing what a piezoelectric
accelerometer mounted on the bearing housing would measure.

IMPORTANT DISCLAIMER:
---------------------
This is a SIMPLIFIED REDUCED-ORDER EDUCATIONAL MODEL.
All parameters are ASSUMED values (see config/parameters.py).
Output is SYNTHETIC DATA, NOT real sensor measurements.

USAGE:
    from src.simulation.signal_generator import generate_vibration
    t, accel = generate_vibration("healthy")
    t, accel = generate_vibration("slight_misalignment", seed=42)
    t, accel = generate_vibration("severe_misalignment", seed=7)
"""

import sys
import os

import numpy as np
from scipy.integrate import solve_ivp

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
# Import all parameters from the central config
# ---------------------------------------------------------------------------
from config.parameters import (
    SAMPLING_RATE_HZ,
    SIGNAL_DURATION_S,
    NUM_SAMPLES,
    ROTATIONAL_FREQUENCY_HZ,
    SECOND_HARMONIC_HZ,
    THIRD_HARMONIC_HZ,
    MASS_KG,
    STIFFNESS_NM,
    DAMPING_COEFF_NSM,
    NATURAL_FREQUENCY_HZ,
    DAMPING_RATIO,
    FORCE_HEALTHY_1X_N,
    FORCE_SLIGHT_1X_N,  FORCE_SLIGHT_2X_N,
    FORCE_SEVERE_1X_N,  FORCE_SEVERE_2X_N, FORCE_SEVERE_3X_N,
    AMP_JITTER_FRACTION,
    NOISE_STD_MS2,
    CONDITIONS,
)

# Pre-compute frequently used derived quantities
_OMEGA_N = 2.0 * np.pi * NATURAL_FREQUENCY_HZ   # natural angular frequency (rad/s)


# ---------------------------------------------------------------------------
# Helper: steady-state particular solution for one harmonic force component
# ---------------------------------------------------------------------------

def _steady_state_x0_v0(amplitude_N: float, frequency_hz: float, phase_rad: float):
    """
    Compute the displacement x(0) and velocity v(0) that correspond to the
    steady-state (particular) solution of the SDOF system for a single
    harmonic force component:

        F(t) = amplitude * sin(2*pi*frequency*t + phase)

    The steady-state solution is:
        x_ss(t) = A_ss * sin(2*pi*f*t + phase + psi)

    where:
        r    = (2*pi*f) / omega_n            (forcing-to-natural frequency ratio)
        H    = 1/sqrt((1-r^2)^2 + (2*zeta*r)^2)  (dynamic magnification factor)
        A_ss = H * F0 / k                    (steady-state displacement amplitude)
        psi  = atan2(-2*zeta*r, 1-r^2)      (phase lag due to damping)

    Evaluated at t = 0:
        x_ss(0) = A_ss * sin(phase + psi)
        v_ss(0) = A_ss * omega * cos(phase + psi)

    Parameters
    ----------
    amplitude_N   : Force amplitude (N)
    frequency_hz  : Forcing frequency (Hz)
    phase_rad     : Phase offset (rad)

    Returns
    -------
    x0 : float  Displacement at t=0 (m)
    v0 : float  Velocity at t=0 (m/s)
    """
    omega = 2.0 * np.pi * frequency_hz
    r     = omega / _OMEGA_N

    # Dynamic magnification factor (dimensionless)
    H = 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * DAMPING_RATIO * r)**2)

    # Phase lag (the damped response lags the force)
    psi = np.arctan2(-2.0 * DAMPING_RATIO * r, 1.0 - r**2)

    # Steady-state displacement amplitude
    A_ss = H * (amplitude_N / STIFFNESS_NM)

    # Evaluate at t = 0
    total_phase = phase_rad + psi
    x0 = A_ss * np.sin(total_phase)
    v0 = A_ss * omega * np.cos(total_phase)

    return x0, v0


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_vibration(condition: str, seed: int = None):
    """
    Generate a physics-based vibration signal by numerically solving the SDOF
    equation of motion for the specified operating condition.

    Parameters
    ----------
    condition : str
        One of: "healthy", "slight_misalignment", "severe_misalignment"
    seed : int, optional
        Seeds the random number generator for reproducible output.
        Set None for fully random samples.

    Returns
    -------
    t_out    : np.ndarray, shape (NUM_SAMPLES,)  -- time axis (s)
    accel_out: np.ndarray, shape (NUM_SAMPLES,)  -- lateral acceleration (m/s^2)

    Raises
    ------
    ValueError  -- if condition name is invalid
    RuntimeError -- if ODE solver fails or produces non-finite values
    """

    # ------------------------------------------------------------------
    # 1. Validate condition name
    # ------------------------------------------------------------------
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition: '{condition}'.\n"
            f"Valid options are: {CONDITIONS}"
        )

    # ------------------------------------------------------------------
    # 2. Initialise random number generator
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 3. Build harmonic force components with random phase + amplitude jitter
    #
    # Each component: (amplitude_N, frequency_hz, phase_rad)
    # Jitter: +/- AMP_JITTER_FRACTION applied to each force amplitude
    # Phase:  uniformly random in [0, 2*pi]
    # ------------------------------------------------------------------

    def jittered_amplitude(nominal_N):
        jitter = rng.uniform(-AMP_JITTER_FRACTION, AMP_JITTER_FRACTION)
        return nominal_N * (1.0 + jitter)

    def rand_phase():
        return rng.uniform(0.0, 2.0 * np.pi)

    if condition == "healthy":
        # 1x rotational (imbalance) force only
        force_components = [
            (jittered_amplitude(FORCE_HEALTHY_1X_N),  ROTATIONAL_FREQUENCY_HZ, rand_phase()),
        ]

    elif condition == "slight_misalignment":
        # 1x imbalance + moderate 2x misalignment force
        force_components = [
            (jittered_amplitude(FORCE_SLIGHT_1X_N),   ROTATIONAL_FREQUENCY_HZ, rand_phase()),
            (jittered_amplitude(FORCE_SLIGHT_2X_N),   SECOND_HARMONIC_HZ,      rand_phase()),
        ]

    elif condition == "severe_misalignment":
        # 1x imbalance + strong 2x + elevated 3x (non-linear bearing force)
        force_components = [
            (jittered_amplitude(FORCE_SEVERE_1X_N),   ROTATIONAL_FREQUENCY_HZ, rand_phase()),
            (jittered_amplitude(FORCE_SEVERE_2X_N),   SECOND_HARMONIC_HZ,      rand_phase()),
            (jittered_amplitude(FORCE_SEVERE_3X_N),   THIRD_HARMONIC_HZ,       rand_phase()),
        ]

    # ------------------------------------------------------------------
    # 4. Compute steady-state initial conditions
    #
    # Sum the particular-solution x(0) and v(0) from each harmonic.
    # This eliminates the free-vibration transient at fn = 150 Hz that
    # would otherwise appear for the first ~85 ms of the signal.
    # ------------------------------------------------------------------
    x0_total = 0.0
    v0_total = 0.0
    for (amp, freq, phase) in force_components:
        x0_i, v0_i = _steady_state_x0_v0(amp, freq, phase)
        x0_total  += x0_i
        v0_total  += v0_i

    # ------------------------------------------------------------------
    # 5. Define the ODE right-hand side
    #
    # State vector: y = [x, v]
    # dy/dt = [v,  (F(t) - c*v - k*x) / m]
    # ------------------------------------------------------------------

    def force_at_time(t_val):
        """Sum of all harmonic force components at time t."""
        total = 0.0
        for (amp, freq, phase) in force_components:
            total += amp * np.sin(2.0 * np.pi * freq * t_val + phase)
        return total

    def sdof_odes(t_val, y):
        """
        ODE right-hand side for solve_ivp.
        y = [displacement (m), velocity (m/s)]
        returns dy/dt = [velocity, acceleration]
        """
        x_val  = y[0]
        v_val  = y[1]
        F_t    = force_at_time(t_val)
        x_ddot = (F_t - DAMPING_COEFF_NSM * v_val - STIFFNESS_NM * x_val) / MASS_KG
        return [v_val, x_ddot]

    # ------------------------------------------------------------------
    # 6. Solve ODE with RK45 adaptive integrator
    #
    # Starting from steady-state initial conditions (step 4) so the signal
    # is immediately periodic with no transient artefact.
    # dense_output=True enables accurate resampling to the uniform grid.
    # ------------------------------------------------------------------
    t_span   = (0.0, SIGNAL_DURATION_S)
    y0       = [x0_total, v0_total]
    dt_out   = 1.0 / SAMPLING_RATE_HZ
    max_step = dt_out / 2.0     # solver step <= half the output interval

    sol = solve_ivp(
        fun          = sdof_odes,
        t_span       = t_span,
        y0           = y0,
        method       = "RK45",
        max_step     = max_step,
        dense_output = True,
        rtol         = 1e-6,
        atol         = 1e-9,
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE solver failed for condition '{condition}': {sol.message}"
        )

    # ------------------------------------------------------------------
    # 7. Resample onto uniform output time grid
    # ------------------------------------------------------------------
    t_out = np.linspace(0.0, SIGNAL_DURATION_S, NUM_SAMPLES, endpoint=False)
    y_out = sol.sol(t_out)    # shape: (2, NUM_SAMPLES)
    x_out = y_out[0]          # displacement (m)
    v_out = y_out[1]          # velocity (m/s)

    # ------------------------------------------------------------------
    # 8. Calculate acceleration directly from the equation of motion
    #
    # x''(t) = [F(t) - c*v(t) - k*x(t)] / m
    #
    # This is mathematically exact (uses the ODE itself) rather than
    # numerically differentiating displacement (which amplifies noise).
    # ------------------------------------------------------------------
    F_out     = np.array([force_at_time(ti) for ti in t_out])
    accel_out = (F_out - DAMPING_COEFF_NSM * v_out - STIFFNESS_NM * x_out) / MASS_KG

    # ------------------------------------------------------------------
    # 9. Add Gaussian sensor noise
    # ------------------------------------------------------------------
    noise     = rng.normal(loc=0.0, scale=NOISE_STD_MS2, size=NUM_SAMPLES)
    accel_out = accel_out + noise

    # ------------------------------------------------------------------
    # 10. Numerical safety check
    # ------------------------------------------------------------------
    if not np.all(np.isfinite(accel_out)):
        raise RuntimeError(
            f"Non-finite values in acceleration for condition '{condition}'. "
            f"Check model parameters in config/parameters.py."
        )

    return t_out, accel_out
