\# Digital Twin of a Motor–Shaft–Bearing System for Misalignment Detection



A simplified physics-based digital twin for simulating vibration behavior of a motor–shaft–bearing system under different shaft alignment conditions and classifying the machine condition using Machine Learning.



\## Project Overview



Shaft misalignment is a common source of vibration in rotating machinery. This project develops a simplified digital-twin workflow to investigate how misalignment affects vibration signatures.



The project combines:



\- SolidWorks CAD modeling

\- Physics-based vibration simulation in Python

\- FFT-based frequency analysis

\- Vibration feature extraction

\- Random Forest classification

\- Robustness testing under operating-condition variations



> \*\*Note:\*\* This is a simplified/reduced-order digital twin using simulated data. It is not a full FEA model and has not been experimentally validated.



\---



\## System Workflow



```text

SolidWorks CAD Model

&#x20;       ↓

Reduced-Order Physics Model

&#x20;       ↓

Vibration Signal Simulation

&#x20;       ↓

Healthy / Slight / Severe Misalignment

&#x20;       ↓

FFT Analysis

&#x20;       ↓

Feature Extraction

&#x20;       ↓

Random Forest Classifier

&#x20;       ↓

Condition Prediction

