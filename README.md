# Loop to Python Adaptive

Two-layer adaptive insulin delivery algorithm, faithfully implementing oref0. For use on LoopAlgorithm via LoopAlgorithmToPython 

## Overview

This library provides adaptive parameter tuning for closed-loop insulin delivery systems:

- **Layer 1: Autotune** — Daily optimization of ISF, Carb Ratio, and Basal Rate
- **Layer 2: Autosens** — Real-time sensitivity adjustments (every 5 minutes)

Based on [openaps/oref0](https://github.com/openaps/oref0).
