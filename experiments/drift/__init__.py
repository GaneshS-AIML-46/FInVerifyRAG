"""
experiments/drift/__init__.py
-----------------------------
Standalone drift experiment — NOT part of the main RegGuard pipeline or paper claims.
See README.md for status and known limitations.
"""
from experiments.drift.drift_exp import (
    run_drift_experiment,
    DriftExperimentResult,
    DriftWindowResult,
    calibrate_threshold,
)

__all__ = [
    "run_drift_experiment",
    "DriftExperimentResult",
    "DriftWindowResult",
    "calibrate_threshold",
]
