# Digital Twin of a Motor–Shaft–Bearing System
## Simulation Model Documentation

> **Document type:** Beginner-friendly technical explanation  
> **Model type:** Simplified physics-based digital twin / reduced-order dynamic model  
> **Stage:** 2 — Vibration Simulation & Dataset Generation  
> **Author note:** All parameters marked [ASSUMED] are illustrative values for educational purposes. This is NOT a full finite-element or experimental model.

---

## 1. What Is a Digital Twin in This Project?

A **digital twin** is a computer model that mimics the behaviour of a real physical object or system. In industry, digital twins of machines are used to:

- Monitor machine health remotely
- Detect faults before they cause breakdowns
- Test different operating conditions safely, without running the real machine

In this project, our digital twin is a **Python simulation** that mimics the vibration behaviour of a **motor–shaft–bearing assembly** that was designed in SolidWorks. Instead of mounting a real accelerometer on a real machine, we use the simulation to generate vibration data — data we can then use to train a machine-learning fault detector.

> **Important:** This is a *simplified, educational* digital twin. It does not capture every physical detail of the real assembly. It is called a **reduced-order dynamic model** because it reduces the infinite complexity of a real structure into a single, solvable equation.

---

## 2. What Does the SolidWorks Model Represent?

The SolidWorks CAD model defines the physical geometry of the system:

| Component | Dimension |
|---|---|
| Shaft diameter | 50 mm |
| Bearing bore | 50 mm |
| Bearing outer diameter | 90 mm |
| Bearing width | 20 mm |

The shaft is driven by a motor at **1800 RPM**. The bearing supports the shaft radially (sideways). When the shaft rotates — even in a healthy, well-aligned state — there is always some small vibration due to manufacturing imperfections (called *residual imbalance*). When the shaft is misaligned with the motor, the vibration grows stronger.

The SolidWorks model gives us the geometry. The Python model gives us the dynamic vibration behaviour.

---

## 3. Why Python Instead of ANSYS or FEA?

A full **finite-element analysis (FEA)** in ANSYS would divide the shaft and bearing into thousands of tiny elements and solve for the exact stress and vibration at every point. This is very accurate but:

- Requires ANSYS software (not available for this project)
- Is computationally expensive
- Produces results that are difficult to use directly for machine-learning training

Instead, we use a **single-degree-of-freedom (SDOF) lumped-parameter model** in Python. This:

- Requires only NumPy and SciPy (free, open-source)
- Runs in seconds
- Captures the essential vibration physics needed for fault detection
- Can generate hundreds of training samples quickly

---

## 4. The Equation of Motion — What It Means Physically

The entire simulation is based on one equation:

$$m\ddot{x}(t) + c\dot{x}(t) + kx(t) = F(t)$$

This is **Newton's Second Law** applied to a vibrating mechanical system. Every term has a physical meaning:

| Term | Symbol | Physical meaning |
|---|---|---|
| **Inertia force** | $m\ddot{x}$ | The force needed to accelerate the rotating mass |
| **Damping force** | $c\dot{x}$ | The energy lost to friction, lubrication, and material damping — opposes motion |
| **Stiffness force** | $kx$ | The restoring force from the shaft stiffness and bearing — pulls the shaft back to centre |
| **Excitation force** | $F(t)$ | The time-varying force from shaft rotation and misalignment |
| **Displacement** | $x(t)$ | How far the shaft centre moves sideways from its rest position |

> Think of it like a car suspension: the mass is the car body, the spring is the suspension spring (stiffness), and the damper is the shock absorber (damping). The road bumps are the excitation force.

---

## 5. What Mass, Stiffness, and Damping Represent

### Mass — m [kg]

The **effective mass** represents the combined inertia of all rotating parts: the motor rotor, shaft, and coupling. We use a single lumped mass rather than distributing mass along the shaft (as FEA would).

> **[ASSUMED value: m = 5.0 kg]**  
> Representative of a small laboratory motor assembly. Not extracted from the SolidWorks mass properties.

### Stiffness — k [N/m]

The **effective stiffness** lumps together two physical effects:
1. **Shaft bending stiffness** — a steel shaft resists lateral bending
2. **Bearing radial stiffness** — rolling-element bearings resist lateral displacement of the shaft

A higher stiffness means the shaft returns to centre faster → higher natural frequency.

> **[DERIVED: k = m · (2πfn)² = 4.4413 MN/m]**  
> Calculated from the assumed mass and natural frequency. Not measured from the SolidWorks assembly.

### Damping — c [N·s/m]

The **damping coefficient** represents energy dissipation from:
- Lubricant film viscosity in the bearing
- Internal material damping of the steel shaft
- Structural coupling losses at joints

Damping is what causes free vibrations to die out after a disturbance. We use a **damping ratio ζ (zeta)** to specify how much damping exists relative to the critical value:
- ζ = 0 → no damping (oscillates forever)  
- ζ = 1 → critically damped (returns to rest without oscillating)  
- ζ = 0.05 → lightly damped (oscillates for a long time, typical of steel machinery)

> **[ASSUMED: ζ = 0.05]**  
> **[DERIVED: c = 2ζ√(km) = 471.24 N·s/m]**

---

## 6. How Rotational Frequency Is Calculated from RPM

The motor runs at **1800 RPM** (revolutions per minute). The rotational frequency in Hz (cycles per second) is:

$$f_{rot} = \frac{\text{RPM}}{60} = \frac{1800}{60} = 30 \text{ Hz}$$

This means the shaft completes **30 full rotations per second**. Each rotation creates one cycle of the imbalance force, so the fundamental vibration frequency is **30 Hz** (also written as 1× — "one times the rotational frequency").

We also use its **harmonics** (integer multiples):

| Harmonic | Frequency | Notation |
|---|---|---|
| 1st | 30 Hz | 1× rev |
| 2nd | 60 Hz | 2× rev |
| 3rd | 90 Hz | 3× rev |

---

## 7. How Misalignment Is Represented — The Fault Model

In a real rotating machine, **misalignment** occurs when the motor shaft and the driven shaft are not perfectly co-axial. Even small angular or parallel offsets cause the shaft to deflect in a repeating pattern as it rotates.

The key physical insight is: **misalignment excites higher harmonic forces** in addition to the normal 1× imbalance. Our model represents this as follows:

| Condition | Forces included | Physical interpretation |
|---|---|---|
| **Healthy** | 50 N at 30 Hz (1×) | Only residual imbalance — shaft rotates smoothly |
| **Slight misalignment** | 50 N at 30 Hz + **20 N at 60 Hz** | Misalignment introduces a 2× loading each revolution |
| **Severe misalignment** | 50 N at 30 Hz + **35 N at 60 Hz** + **15 N at 90 Hz** | Large misalignment; non-linear bearing forces excite 3× harmonic |

The excitation force for each harmonic takes the form:

$$F_i(t) = A_i \cdot \sin(2\pi f_i t + \phi_i)$$

where $\phi_i$ is a **random phase offset** (different for every signal sample) and $A_i$ includes a small random ±10% amplitude jitter. This prevents all 500 signals in each class from being identical.

> **[ASSUMED force values]** — The 2× and 3× force amplitudes are illustrative values that produce realistic severity ratios. They are NOT derived from imbalance grade standards or bearing load calculations.

---

## 8. Why These Parameters Were Chosen (Natural Frequency Selection)

A pre-implementation analytical sweep was performed over multiple candidate natural frequencies. The key concern was **near-resonance amplification**: if the natural frequency is too close to an excitation frequency, the system response becomes unrealistically large.

| fn tested | 90 Hz component (r=0.9) | Severe/Healthy RMS ratio | Decision |
|---|---|---|---|
| 100 Hz | **11.6 m/s²** (near resonance!) | 12.4× | ❌ Too extreme |
| **150 Hz** | **1.68 m/s²** | **5.2×** | ✅ Selected |
| 200 Hz | 0.76 m/s² | 4.6× | Acceptable but lower contrast |

At **fn = 150 Hz**, all excitation frequencies are comfortably below resonance (maximum frequency ratio r = 90/150 = 0.60), giving a clean physical response.

---

## 9. Why Acceleration Is Used as the Simulated Sensor Signal

In real rotating machinery condition monitoring, the most common sensor type is a **piezoelectric accelerometer**. An accelerometer:

- Is bolted directly to the bearing housing
- Measures the **acceleration** of that surface in m/s²
- Is small, robust, and sensitive to the frequency range of interest (10–10,000 Hz)

The simulation therefore outputs **acceleration** (`d²x/dt²`) rather than displacement or velocity. Acceleration is recovered from the equation of motion directly:

$$\ddot{x}(t) = \frac{F(t) - c\dot{x}(t) - kx(t)}{m}$$

This is computed algebraically from the ODE rather than by numerically differentiating the displacement, which would amplify numerical noise.

---

## 10. Steady-State Initial Conditions — Why They Matter

When a numerical ODE solver starts from rest (`x=0, v=0`), the response includes a **free-vibration transient** at the natural frequency (150 Hz). This transient decays exponentially with time constant:

$$\tau = \frac{1}{\zeta \omega_n} = \frac{1}{0.05 \times 2\pi \times 150} \approx 21 \text{ ms}$$

The transient is ~98% gone after 4τ ≈ 85 ms. In a 1-second signal this may seem small, but the 150 Hz transient briefly dominates the healthy signal (which has only one small force component), causing incorrect RMS values.

**Solution:** The solver is initialised at the **analytical steady-state** displacement and velocity for each harmonic force component. For a single harmonic $F_0 \sin(\omega t + \phi)$, the steady-state particular solution is:

$$x_{ss}(0) = \frac{H \cdot F_0}{k} \sin(\phi + \psi), \quad \dot{x}_{ss}(0) = \frac{H \cdot F_0 \cdot \omega}{k} \cos(\phi + \psi)$$

where $H = \frac{1}{\sqrt{(1-r^2)^2 + (2\zeta r)^2}}$ (dynamic magnification factor) and $\psi = \arctan\!\left(\frac{-2\zeta r}{1-r^2}\right)$ (phase lag).

The total initial conditions are the sum of all harmonic components. This immediately places the system in its correct periodic steady-state — no transient, no artefacts.

---

## 11. Why the Model Is Simplified

This model deliberately makes several simplifications:

| Simplification | Real-world equivalent |
|---|---|
| 1 degree of freedom (x only) | A real shaft vibrates in x, y, z and also rotates — many DOF |
| Lumped mass | Real mass is distributed along the shaft length |
| Linear stiffness | Real bearing stiffness is slightly non-linear with load |
| Sinusoidal force model | Real misalignment forces also include sidebands and modulation |
| No cross-coupling between axes | Real vibration has x-y coupling |
| No bearing defect frequencies | BPFO, BPFI, BSF are not modelled |

These simplifications are acceptable for this educational project because:
- The goal is to demonstrate the full pipeline (simulation → FFT → ML), not to replicate FEA accuracy
- The simplified model still produces clearly distinguishable frequency signatures per condition
- The model is transparent, documented, and easily modified

---

## 12. Why the Generated Data Is Synthetic

The 1500 CSV files in `data/raw/` are **synthetic data** — generated entirely by mathematical simulation. They are NOT:
- Recorded from a real motor-shaft-bearing assembly
- Calibrated against any physical measurement
- Representative of a specific real machine's actual response

They ARE:
- Physically meaningful (based on the SDOF equation of motion with reasonable parameters)
- Statistically diverse (random phase and amplitude jitter per sample)
- Suitable for training a proof-of-concept machine-learning classifier

**Always state that this data is simulated** in any report, paper, or presentation.

---

## 13. Limitations Summary

| Limitation | Impact on ML stage |
|---|---|
| No amplitude modulation | AM-based features will not appear; power spectral density (PSD) features will still work |
| No sub-harmonics (0.5× component) | Some real misalignment signatures not captured |
| Constant mean amplitude per class | Real signals have slow drift; add if more realism is needed |
| Gaussian noise only | Real noise can be non-Gaussian (impulsive); augmentation may be needed later |
| Only radial x-direction | Axial vibration (common in angular misalignment) not included |
| ASSUMED parameters | Model accuracy depends entirely on how well the assumed m, k, c represent the real machine |

---

## 14. Parameter Quick-Reference

```
Operating parameters:
  RPM                  = 1800
  f_rot (1x)           = 30 Hz
  f_2x                 = 60 Hz
  f_3x                 = 90 Hz

SDOF model parameters  [ALL ASSUMED/DERIVED for educational model]:
  m    = 5.0 kg         [ASSUMED]
  fn   = 150.0 Hz       [ASSUMED — selected from pre-impl. sweep]
  zeta = 0.05           [ASSUMED]
  k    = 4.4413 MN/m    [DERIVED: k = m*(2*pi*fn)^2]
  c    = 471.24 N.s/m   [DERIVED: c = 2*zeta*sqrt(k*m)]

Excitation forces       [ALL ASSUMED]:
  Healthy:              F1 = 50 N  at 30 Hz
  Slight misalignment:  F1 = 50 N  at 30 Hz,  F2 = 20 N at 60 Hz
  Severe misalignment:  F1 = 50 N  at 30 Hz,  F2 = 35 N at 60 Hz,
                        F3 = 15 N  at 90 Hz

Simulation settings:
  Fs   = 1000 Hz
  T    = 1.0 s
  N    = 1000 samples per signal
  Noise sigma = 0.02 m/s^2 (Gaussian, ASSUMED)

Dataset:
  500 signals per condition
  1500 total physics-based CSV files
  1500 prototype (sine-wave) CSV files preserved in data/raw/prototype_synthetic/
```
