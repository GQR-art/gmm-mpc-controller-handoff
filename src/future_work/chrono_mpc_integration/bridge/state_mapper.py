"""Pure adapter from Chrono raw state to frozen Nominal MPC state contract."""

from __future__ import annotations

from future_work.later_mpc.types import VehicleState

from .types import ChronoRawState


def chrono_raw_to_vehicle_state(raw_state: ChronoRawState) -> VehicleState:
    """Map one Chrono measurement to the frozen ``later_mpc`` contract.

    All selected fields are already SI/radian quantities.  In particular,
    ``vy_mps`` is **only** the chassis body-frame lateral velocity; the
    world-frame ``vy_world`` is intentionally not used.
    """
    return VehicleState(
        X_m=raw_state.X,
        Y_m=raw_state.Y,
        psi_rad=raw_state.psi,
        vy_mps=raw_state.vy_body,
        yaw_rate_radps=raw_state.yaw_rate_body,
    )
