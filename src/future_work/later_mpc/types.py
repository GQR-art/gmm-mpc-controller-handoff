"""Shared data contracts for the nominal MPC research module.

These are intentionally solver- and plant-agnostic.  Later stages use the same
contracts to connect the controller, prediction model, plant model, simulator,
and future GMR-augmented prediction model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VehicleState:
    """Global dynamic-vehicle state used at the plant/controller boundary."""

    X_m: float
    Y_m: float
    psi_rad: float
    vy_mps: float
    yaw_rate_radps: float


@dataclass(frozen=True)
class ReferencePoint:
    """One path-coordinate reference point at arc-length station ``s_m``."""

    s_m: float
    X_m: float
    Y_m: float
    psi_rad: float
    curvature_per_m: float


@dataclass(frozen=True)
class VehicleErrorState:
    """Path-relative controller state derived from one global plant state."""

    ey_m: float
    epsi_rad: float
    vy_mps: float
    yaw_rate_radps: float
    reference: ReferencePoint

    def as_vector(self) -> np.ndarray:
        """Return the MPC state vector ``[ey, epsi, vy, r]``."""
        return np.array(
            [self.ey_m, self.epsi_rad, self.vy_mps, self.yaw_rate_radps], dtype=float
        )


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Sampled reference path/preview with mutually aligned arrays."""

    X_m: np.ndarray
    Y_m: np.ndarray
    psi_rad: np.ndarray
    curvature_per_m: np.ndarray
    arc_length_m: np.ndarray
    name: str = "reference"

    def __post_init__(self) -> None:
        lengths = {
            len(self.X_m),
            len(self.Y_m),
            len(self.psi_rad),
            len(self.curvature_per_m),
            len(self.arc_length_m),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
            raise ValueError("Reference trajectory arrays must share a length of at least two.")


@dataclass(frozen=True)
class MPCSolution:
    """Solver output for one receding-horizon optimization step."""

    steering_sequence_rad: np.ndarray
    predicted_error_states: np.ndarray
    objective_value: float
    status: str
    solve_time_s: float
    success: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_sequence(self) -> np.ndarray:
        """Alias using the MPC notation requested in experiment specifications."""
        return self.steering_sequence_rad

    @property
    def optimal_cost(self) -> float:
        """Alias for the optimal finite-horizon objective value."""
        return self.objective_value

    @property
    def solve_status(self) -> str:
        """Alias for the solver status string."""
        return self.status

    @property
    def solve_time(self) -> float:
        """Alias for solver wall-clock time in seconds."""
        return self.solve_time_s

    @property
    def solver_information(self) -> dict[str, Any]:
        """Alias for structured solver diagnostics."""
        return self.diagnostics


@dataclass(frozen=True)
class SimulationResult:
    """Recorded closed-loop simulation signals and aggregate diagnostics."""

    time_s: np.ndarray
    plant_states: list[VehicleState]
    error_states: np.ndarray
    steering_rad: np.ndarray
    solve_times_s: np.ndarray
    solver_statuses: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
