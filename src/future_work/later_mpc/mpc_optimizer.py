"""OSQP-based nominal MPC quadratic-program construction and solution.

This module has no knowledge of the closed-loop plant.  It obtains all discrete
prediction matrices exclusively through the supplied :class:`PredictionModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy import sparse

try:
    import osqp
except ImportError as exc:  # pragma: no cover - depends on the selected interpreter
    raise ImportError(
        "OSQP is required for NominalMPCOptimizer. Use the isolated interpreter "
        "D:\\VehicleResearch_Env\\mpc_env\\Scripts\\python.exe."
    ) from exc

from .config import MPCConfig
from .prediction_model import PredictionModel
from .types import MPCSolution, ReferenceTrajectory, VehicleErrorState


@dataclass(frozen=True)
class _CondensedPrediction:
    """Affine state maps ``x_i = state_offsets[i] + state_maps[i] U``."""

    state_maps: tuple[np.ndarray, ...]
    state_offsets: tuple[np.ndarray, ...]


class NominalMPCOptimizer:
    """Finite-horizon constrained QP optimizer for a supplied prediction model."""

    def __init__(self, config: MPCConfig | None = None) -> None:
        self._config = config or MPCConfig()

    @property
    def config(self) -> MPCConfig:
        """Return immutable MPC configuration used to form the QP."""
        return self._config

    @staticmethod
    def _state_vector(current_state: np.ndarray | Sequence[float] | VehicleErrorState) -> np.ndarray:
        if isinstance(current_state, VehicleErrorState):
            state = current_state.as_vector()
        else:
            state = np.asarray(current_state, dtype=float)
        if state.shape != (4,):
            raise ValueError("current_state must have shape (4,) for [ey, epsi, vy, r].")
        return state

    def _curvature_preview(
        self,
        reference_preview: ReferenceTrajectory | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(reference_preview, ReferenceTrajectory):
            curvatures = np.asarray(reference_preview.curvature_per_m, dtype=float)
        else:
            curvatures = np.asarray(reference_preview, dtype=float)
        curvatures = curvatures.reshape(-1)
        horizon = self._config.horizon_steps
        if curvatures.size == horizon + 1:
            return curvatures[:horizon]
        if curvatures.size == horizon:
            return curvatures
        raise ValueError(f"reference_preview must contain {horizon} or {horizon + 1} curvatures.")

    def _condense_prediction(
        self,
        current_state: np.ndarray,
        prediction_model: PredictionModel,
        curvatures: np.ndarray,
    ) -> _CondensedPrediction:
        """Compose state maps solely from PredictionModel-provided discretizations."""
        horizon = self._config.horizon_steps
        maps: list[np.ndarray] = [np.zeros((4, horizon), dtype=float)]
        offsets: list[np.ndarray] = [current_state]
        for index, curvature in enumerate(curvatures):
            continuous = prediction_model.linearize(
                current_state,
                self._config.longitudinal_speed_mps,
                float(curvature),
            )
            discrete = prediction_model.discretize(continuous, self._config.sample_time_s)
            state_map = discrete.Ad @ maps[-1]
            state_map[:, index] += discrete.Bd[:, 0]
            state_offset = discrete.Ad @ offsets[-1] + discrete.gd
            maps.append(state_map)
            offsets.append(state_offset)
        return _CondensedPrediction(tuple(maps), tuple(offsets))

    def _cost_matrices(
        self,
        condensed: _CondensedPrediction,
        previous_delta_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct ``U.T H U + f.T U`` excluding only constant terms."""
        horizon = self._config.horizon_steps
        weights = self._config.state_weights
        input_weights = self._config.input_weights
        Q = np.diag([weights.ey, weights.epsi, weights.vy, weights.r])
        Q_terminal = weights.terminal_scale * Q

        H = np.zeros((horizon, horizon), dtype=float)
        f = np.zeros(horizon, dtype=float)
        # x_0 is fixed, so it does not affect the minimizer. Penalize x_1...x_N.
        for index in range(1, horizon + 1):
            state_weight = Q_terminal if index == horizon else Q
            state_map = condensed.state_maps[index]
            state_offset = condensed.state_offsets[index]
            H += state_map.T @ state_weight @ state_map
            f += 2.0 * state_map.T @ state_weight @ state_offset

        H += input_weights.delta * np.eye(horizon)
        rate_difference = np.eye(horizon)
        rate_difference[1:, :-1] -= np.eye(horizon - 1)
        previous_offset = np.zeros(horizon, dtype=float)
        previous_offset[0] = -float(previous_delta_rad)
        H += input_weights.delta_rate * (rate_difference.T @ rate_difference)
        f += 2.0 * input_weights.delta_rate * (rate_difference.T @ previous_offset)
        return H, f

    def _constraint_matrices(self, previous_delta_rad: float) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
        """Build steering-amplitude and steering-increment hard constraints."""
        horizon = self._config.horizon_steps
        constraints = self._config.steering_constraints
        identity = np.eye(horizon)
        difference = np.eye(horizon)
        difference[1:, :-1] -= np.eye(horizon - 1)
        difference_offset = np.zeros(horizon, dtype=float)
        difference_offset[0] = -float(previous_delta_rad)

        A = sparse.csc_matrix(np.vstack((identity, difference)))
        lower = np.concatenate(
            (
                np.full(horizon, constraints.delta_min_rad),
                np.full(horizon, constraints.delta_rate_min_rad_per_sample) - difference_offset,
            )
        )
        upper = np.concatenate(
            (
                np.full(horizon, constraints.delta_max_rad),
                np.full(horizon, constraints.delta_rate_max_rad_per_sample) - difference_offset,
            )
        )
        return A, lower, upper

    @staticmethod
    def _states_from_solution(condensed: _CondensedPrediction, controls: np.ndarray) -> np.ndarray:
        return np.vstack(
            [offset + state_map @ controls for state_map, offset in zip(condensed.state_maps, condensed.state_offsets)]
        )

    def _objective_value(self, states: np.ndarray, controls: np.ndarray, previous_delta_rad: float) -> float:
        weights = self._config.state_weights
        input_weights = self._config.input_weights
        Q = np.diag([weights.ey, weights.epsi, weights.vy, weights.r])
        Q_terminal = weights.terminal_scale * Q
        value = sum(float(state @ Q @ state) for state in states[1:-1])
        value += float(states[-1] @ Q_terminal @ states[-1])
        value += input_weights.delta * float(controls @ controls)
        increments = np.diff(np.concatenate(([float(previous_delta_rad)], controls)))
        value += input_weights.delta_rate * float(increments @ increments)
        return value

    def evaluate_cost(
        self,
        current_state: np.ndarray | Sequence[float] | VehicleErrorState,
        prediction_model: PredictionModel,
        reference_preview: ReferenceTrajectory | Sequence[float] | np.ndarray,
        previous_delta: float,
        control_sequence_rad: Sequence[float] | np.ndarray,
    ) -> float:
        """Evaluate the configured finite-horizon objective without solving a QP."""
        state = self._state_vector(current_state)
        controls = np.asarray(control_sequence_rad, dtype=float).reshape(-1)
        if controls.size != self._config.horizon_steps:
            raise ValueError("control_sequence_rad length must equal the MPC horizon.")
        condensed = self._condense_prediction(state, prediction_model, self._curvature_preview(reference_preview))
        states = self._states_from_solution(condensed, controls)
        return self._objective_value(states, controls, previous_delta)

    def solve(
        self,
        current_state: np.ndarray | Sequence[float] | VehicleErrorState,
        prediction_model: PredictionModel,
        reference_preview: ReferenceTrajectory | Sequence[float] | np.ndarray,
        previous_delta: float,
    ) -> MPCSolution:
        """Solve the constrained nominal MPC QP and return its optimal sequence."""
        if not isinstance(prediction_model, PredictionModel):
            raise TypeError("prediction_model must implement PredictionModel.")
        state = self._state_vector(current_state)
        curvatures = self._curvature_preview(reference_preview)
        condensed = self._condense_prediction(state, prediction_model, curvatures)
        H, f = self._cost_matrices(condensed, previous_delta)
        A, lower, upper = self._constraint_matrices(previous_delta)

        solver = osqp.OSQP()
        started_at = perf_counter()
        solver.setup(
            P=sparse.csc_matrix(2.0 * H),
            q=f,
            A=A,
            l=lower,
            u=upper,
            verbose=False,
            # The problem is small and already solved to tight tolerance; avoid
            # per-step console messages from unnecessary OSQP polishing.
            polishing=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
        )
        result = solver.solve()
        elapsed_s = perf_counter() - started_at
        status = str(result.info.status)
        success = status.lower().startswith("solved") and result.x is not None
        controls = np.asarray(result.x if success else np.zeros(self._config.horizon_steps), dtype=float)
        states = self._states_from_solution(condensed, controls)
        diagnostics = {
            "solver": "OSQP",
            "status_val": int(result.info.status_val),
            "iterations": int(result.info.iter),
            "run_time_s": float(result.info.run_time),
            "setup_time_s": float(result.info.setup_time),
            "solve_time_s": float(result.info.solve_time),
            "objective_without_constants": float(result.info.obj_val),
        }
        return MPCSolution(
            steering_sequence_rad=controls,
            predicted_error_states=states,
            objective_value=self._objective_value(states, controls, previous_delta),
            status=status,
            solve_time_s=elapsed_s,
            success=success,
            diagnostics=diagnostics,
        )
