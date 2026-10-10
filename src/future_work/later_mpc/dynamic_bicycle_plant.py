"""Independent closed-loop plant based on the paper bicycle dynamics equation.

``DynamicBicycleModel`` is used here solely as the source of the lateral
differential equation through ``state_derivative``.  This plant owns its state,
global kinematics, and RK4 integration; it deliberately never calls
``DynamicBicycleModel.simulate``.
"""

from __future__ import annotations

from dataclasses import replace
from math import cos, sin

import numpy as np

from bicycle_model import DynamicBicycleModel

from .config import VehicleParameters, load_fitted_vehicle_parameters
from .types import VehicleState


class DynamicBicyclePlant:
    """Five-state vehicle-response plant for closed-loop simulations.

    State ordering is ``[X, Y, psi, vy, r]``.  The plant is intentionally not a
    prediction model: its only public propagation operation is a real-response
    RK4 step under an applied steering input.
    """

    def __init__(
        self,
        vehicle_parameters: VehicleParameters | None = None,
        initial_state: VehicleState | None = None,
    ) -> None:
        self._parameters = vehicle_parameters or load_fitted_vehicle_parameters()
        # This object supplies only state_derivative; simulate() is never used.
        self._lateral_dynamics = DynamicBicycleModel(self._parameters.as_dict())
        self._state = initial_state or VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)

    def reset(self, initial_state: VehicleState) -> None:
        """Replace the plant state without modifying the dynamics model."""
        self._state = initial_state

    def get_state(self) -> VehicleState:
        """Return an immutable snapshot of the current global plant state."""
        return replace(self._state)

    def lateral_dynamics_derivative(
        self,
        vy_mps: float,
        yaw_rate_radps: float,
        delta_rad: float,
        vx_mps: float,
    ) -> tuple[float, float]:
        """Return ``(vy_dot, r_dot)`` by delegating to the paper equation only."""
        if vx_mps <= 0.0:
            raise ValueError("vx_mps must be positive for DynamicBicyclePlant.")
        vy_dot, yaw_rate_dot = self._lateral_dynamics.state_derivative(
            0.0,
            [float(vy_mps), float(yaw_rate_radps)],
            float(delta_rad),
            float(vx_mps),
        )
        return float(vy_dot), float(yaw_rate_dot)

    def _full_state_derivative(
        self,
        state: np.ndarray,
        delta_rad: float,
        vx_mps: float,
    ) -> np.ndarray:
        """Evaluate global kinematics plus delegated lateral dynamics."""
        X_m, Y_m, psi_rad, vy_mps, yaw_rate_radps = state
        vy_dot, yaw_rate_dot = self.lateral_dynamics_derivative(
            vy_mps, yaw_rate_radps, delta_rad, vx_mps
        )
        X_dot = vx_mps * cos(psi_rad) - vy_mps * sin(psi_rad)
        Y_dot = vx_mps * sin(psi_rad) + vy_mps * cos(psi_rad)
        psi_dot = yaw_rate_radps
        return np.array([X_dot, Y_dot, psi_dot, vy_dot, yaw_rate_dot], dtype=float)

    def step(self, delta: float, vx: float, dt: float) -> VehicleState:
        """Advance the complete plant state one fixed-input RK4 step.

        The applied road-wheel angle ``delta`` and longitudinal speed ``vx`` are
        held constant during the integration interval ``dt``.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if vx <= 0.0:
            raise ValueError("vx must be positive.")

        state = np.array(
            [
                self._state.X_m,
                self._state.Y_m,
                self._state.psi_rad,
                self._state.vy_mps,
                self._state.yaw_rate_radps,
            ],
            dtype=float,
        )
        k1 = self._full_state_derivative(state, delta, vx)
        k2 = self._full_state_derivative(state + 0.5 * dt * k1, delta, vx)
        k3 = self._full_state_derivative(state + 0.5 * dt * k2, delta, vx)
        k4 = self._full_state_derivative(state + dt * k3, delta, vx)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self._state = VehicleState(
            X_m=float(next_state[0]),
            Y_m=float(next_state[1]),
            psi_rad=float(next_state[2]),
            vy_mps=float(next_state[3]),
            yaw_rate_radps=float(next_state[4]),
        )
        return self.get_state()
