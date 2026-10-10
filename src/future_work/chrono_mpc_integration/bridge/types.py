"""Data contracts owned by the Chrono bridge.

The bridge intentionally does not define a second ``VehicleState``.  Its
boundary adapter returns the frozen contract from ``future_work.later_mpc``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ChronoRawState:
    """Vehicle chassis state measured directly from Project Chrono.

    ``X``, ``Y``, ``vx_world`` and ``vy_world`` use Chrono's world frame.
    ``vx_body``, ``vy_body`` and ``yaw_rate_body`` use the chassis body frame
    (X forward, Y left, Z up).  All values use SI units; angles are radians.
    """

    time: float
    X: float
    Y: float
    psi: float
    vx_world: float
    vy_world: float
    vx_body: float
    vy_body: float
    yaw_rate_body: float

    def __post_init__(self) -> None:
        values = (
            self.time,
            self.X,
            self.Y,
            self.psi,
            self.vx_world,
            self.vy_world,
            self.vx_body,
            self.vy_body,
            self.yaw_rate_body,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ChronoRawState values must all be finite.")


@dataclass(frozen=True, slots=True)
class ChronoVehicleMassProperties:
    """Aggregate vehicle properties read from the initialized Chrono plant.

    These values are deliberately separate from ``ChronoRawState``.  They are
    static plant evidence used by replay-validation reports, not controller
    state and not a duplicate of the frozen MPC ``VehicleState``.
    """

    mass_kg: float
    com_x_m: float
    com_y_m: float
    com_z_m: float
    inertia_xx_kg_m2: float
    inertia_yy_kg_m2: float
    inertia_zz_kg_m2: float

    def __post_init__(self) -> None:
        values = (
            self.mass_kg,
            self.com_x_m,
            self.com_y_m,
            self.com_z_m,
            self.inertia_xx_kg_m2,
            self.inertia_yy_kg_m2,
            self.inertia_zz_kg_m2,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ChronoVehicleMassProperties values must be finite.")
        if self.mass_kg <= 0.0 or min(
            self.inertia_xx_kg_m2,
            self.inertia_yy_kg_m2,
            self.inertia_zz_kg_m2,
        ) <= 0.0:
            raise ValueError("Chrono vehicle mass and principal inertias must be positive.")


@dataclass(frozen=True, slots=True)
class ChronoWheelDiagnostic:
    """Read-only state of one initialized Chrono wheel/tire pair.

    Tire-frame forces use the contact frame documented by ``ChTire``: X is
    the tire heading direction, Z is the terrain normal, and Y completes the
    right-handed frame.  Body-frame forces are the same reported terrain force
    rotated into the chassis frame.  No value in this contract is fed back to
    the vehicle simulation.
    """

    wheel_id: str
    axle_index: int
    side: str
    slip_angle_rad: float
    tmeasy_slip_angle_rad: float
    lateral_slip_sy: float
    longitudinal_slip_sx: float
    force_x_tire_N: float
    force_y_tire_N: float
    force_z_tire_N: float
    force_x_body_N: float
    force_y_body_N: float
    force_z_body_N: float
    wheel_omega_radps: float
    effective_radius_m: float
    road_wheel_angle_rad: float

    def __post_init__(self) -> None:
        values = (
            self.slip_angle_rad,
            self.tmeasy_slip_angle_rad,
            self.lateral_slip_sy,
            self.longitudinal_slip_sx,
            self.force_x_tire_N,
            self.force_y_tire_N,
            self.force_z_tire_N,
            self.force_x_body_N,
            self.force_y_body_N,
            self.force_z_body_N,
            self.wheel_omega_radps,
            self.effective_radius_m,
            self.road_wheel_angle_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ChronoWheelDiagnostic numeric values must be finite.")
        if self.axle_index < 0 or self.side not in {"left", "right"}:
            raise ValueError("Invalid wheel identity in ChronoWheelDiagnostic.")
        if self.effective_radius_m <= 0.0:
            raise ValueError("Wheel effective radius must be positive.")


@dataclass(frozen=True, slots=True)
class ChronoWheelSpeedInitialization:
    """Evidence from a one-time physically consistent wheel-speed setup."""

    wheel_id: str
    effective_radius_m: float
    requested_omega_radps: float
    actual_omega_radps: float

    def __post_init__(self) -> None:
        values = (
            self.effective_radius_m,
            self.requested_omega_radps,
            self.actual_omega_radps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Wheel-speed initialization values must be finite.")
        if self.effective_radius_m <= 0.0:
            raise ValueError("Effective rolling radius must be positive.")


@dataclass(frozen=True, slots=True)
class ChronoSuspensionDiagnostic:
    """Read-only suspension/contact evidence for one wheel station.

    ``spindle_*`` locations are expressed in the moving chassis reference
    frame.  ``spring_deformation_m`` follows Chrono's ``ChLinkTSDA`` sign
    convention (current length minus rest length), so compression is negative.
    ``tire_deflection_m`` is meaningful only while ``contact`` is true; the
    force-element tire can retain a signed kinematic value after contact loss.
    """

    wheel_id: str
    axle_index: int
    side: str
    track_m: float
    spindle_x_m: float
    spindle_y_m: float
    spindle_z_m: float
    spring_length_m: float
    spring_deformation_m: float
    spring_force_N: float
    shock_length_m: float
    shock_velocity_mps: float
    tire_deflection_m: float
    tire_effective_radius_m: float
    contact: bool
    contact_depth_m: float
    contact_normal_force_N: float

    def __post_init__(self) -> None:
        values = (
            self.track_m,
            self.spindle_x_m,
            self.spindle_y_m,
            self.spindle_z_m,
            self.spring_length_m,
            self.spring_deformation_m,
            self.spring_force_N,
            self.shock_length_m,
            self.shock_velocity_mps,
            self.tire_deflection_m,
            self.tire_effective_radius_m,
            self.contact_depth_m,
            self.contact_normal_force_N,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ChronoSuspensionDiagnostic values must be finite.")
        if self.axle_index < 0 or self.side not in {"left", "right"}:
            raise ValueError("Invalid wheel identity in suspension diagnostic.")
        if self.track_m <= 0.0 or self.tire_effective_radius_m <= 0.0:
            raise ValueError("Track and tire effective radius must be positive.")
        if self.contact_depth_m < 0.0 or self.contact_normal_force_N < 0.0:
            raise ValueError("Contact depth and normal force cannot be negative.")
