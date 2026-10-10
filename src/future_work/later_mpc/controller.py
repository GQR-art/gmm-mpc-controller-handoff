"""Nominal MPC controller: path-relative state estimation plus QP invocation."""

from __future__ import annotations

from dataclasses import dataclass

from .error_state import global_state_to_error_state
from .mpc_optimizer import NominalMPCOptimizer
from .nominal_bicycle_predictor import NominalBicyclePredictor
from .reference_path import ReferencePath
from .types import MPCSolution, ReferenceTrajectory, VehicleErrorState, VehicleState


@dataclass(frozen=True)
class ControllerOutput:
    """One controller update, including its state/path context for logging."""

    delta_rad: float
    error_state: VehicleErrorState
    reference_preview: ReferenceTrajectory
    solution: MPCSolution


class NominalMPCController:
    """Connect error-state calculation, nominal predictor, and QP optimizer.

    This class does not contain vehicle dynamics or integrate vehicle state.
    """

    def __init__(
        self,
        prediction_model: NominalBicyclePredictor | None = None,
        optimizer: NominalMPCOptimizer | None = None,
    ) -> None:
        self._prediction_model = prediction_model or NominalBicyclePredictor()
        self._optimizer = optimizer or NominalMPCOptimizer()

    @property
    def optimizer(self) -> NominalMPCOptimizer:
        return self._optimizer

    def compute_control(
        self,
        vehicle_state: VehicleState,
        reference_path: ReferencePath,
        previous_delta: float,
    ) -> ControllerOutput:
        """Compute the next applied steering angle from the present vehicle state."""
        error_state = global_state_to_error_state(vehicle_state, reference_path)
        config = self._optimizer.config
        preview = reference_path.generate_horizon(
            current_position=error_state.reference.s_m,
            N=config.horizon_steps,
            Ts=config.sample_time_s,
            vx=config.longitudinal_speed_mps,
        )
        solution = self._optimizer.solve(
            current_state=error_state,
            prediction_model=self._prediction_model,
            reference_preview=preview,
            previous_delta=previous_delta,
        )
        delta_rad = float(solution.delta_sequence[0])
        return ControllerOutput(delta_rad, error_state, preview, solution)
