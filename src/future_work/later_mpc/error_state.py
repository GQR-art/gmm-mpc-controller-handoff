"""Conversion from global plant states to path-relative MPC error states."""

from __future__ import annotations

from math import atan2, cos, pi, sin

from .reference_path import ReferencePath
from .types import VehicleErrorState, VehicleState


def wrap_to_pi(angle_rad: float) -> float:
    """Normalize an angle to the half-open interval ``[-pi, pi)``."""
    return (float(angle_rad) + pi) % (2.0 * pi) - pi


def global_state_to_error_state(
    vehicle_state: VehicleState,
    reference_path: ReferencePath,
) -> VehicleErrorState:
    """Project a global vehicle state onto a path and form ``[ey, epsi, vy, r]``.

    The signed lateral error is the projection of the vehicle-to-reference
    displacement onto the reference path's left normal ``[-sin(psi_ref),
    cos(psi_ref)]``.  Thus a vehicle on the left side of travel has ``ey > 0``.
    """
    reference = reference_path.project(vehicle_state.X_m, vehicle_state.Y_m)
    dX = vehicle_state.X_m - reference.X_m
    dY = vehicle_state.Y_m - reference.Y_m
    ey_m = -sin(reference.psi_rad) * dX + cos(reference.psi_rad) * dY
    epsi_rad = wrap_to_pi(vehicle_state.psi_rad - reference.psi_rad)
    return VehicleErrorState(
        ey_m=float(ey_m),
        epsi_rad=epsi_rad,
        vy_mps=vehicle_state.vy_mps,
        yaw_rate_radps=vehicle_state.yaw_rate_radps,
        reference=reference,
    )
