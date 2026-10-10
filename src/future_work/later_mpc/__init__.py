"""Nominal MPC baseline for lateral vehicle trajectory tracking.

This package is deliberately isolated from the paper implementation.  Its
prediction-model and plant-model abstractions will be introduced in later
implementation stages.
"""

from .config import MPCConfig, VehicleParameters, load_fitted_vehicle_parameters
from .types import (
    MPCSolution,
    ReferencePoint,
    ReferenceTrajectory,
    SimulationResult,
    VehicleErrorState,
    VehicleState,
)

__all__ = [
    "MPCConfig",
    "MPCSolution",
    "ReferencePoint",
    "ReferenceTrajectory",
    "SimulationResult",
    "VehicleErrorState",
    "VehicleParameters",
    "VehicleState",
    "load_fitted_vehicle_parameters",
]
