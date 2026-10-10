"""Minimal time-varying-speed wrapper around the frozen nominal MPC ingredients."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math

import numpy as np
from scipy import sparse
import osqp

from future_work.later_mpc.controller import ControllerOutput
from future_work.later_mpc.error_state import global_state_to_error_state
from future_work.later_mpc.mpc_optimizer import NominalMPCOptimizer, _CondensedPrediction
from future_work.later_mpc.nominal_bicycle_predictor import NominalBicyclePredictor
from future_work.later_mpc.reference_path import ReferencePath
from future_work.later_mpc.types import MPCSolution, ReferenceTrajectory, VehicleState
from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state


@dataclass(frozen=True)
class SpeedProfile:
    station_m: np.ndarray
    speed_mps: np.ndarray
    name: str

    def __post_init__(self) -> None:
        s = np.asarray(self.station_m, float)
        v = np.asarray(self.speed_mps, float)
        if s.ndim != 1 or s.size < 2 or s.shape != v.shape:
            raise ValueError("speed profile arrays must be aligned one-dimensional arrays")
        if not np.all(np.diff(s) > 0) or not np.isfinite(np.c_[s, v]).all() or np.any(v <= 0):
            raise ValueError("speed profile must have finite positive speed and increasing station")

    def sample(self, station_m):
        return np.interp(np.asarray(station_m, float), self.station_m, self.speed_mps)

    def preview(self, current_station_m: float, steps: int, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Known-reference approximation; never reads future Chrono state."""
        stations = np.empty(steps + 1)
        speeds = np.empty(steps)
        stations[0] = float(current_station_m)
        for i in range(steps):
            speeds[i] = float(self.sample(stations[i]))
            stations[i + 1] = stations[i] + speeds[i] * dt_s
        return stations, speeds


class VariableSpeedNominalMPCOptimizer(NominalMPCOptimizer):
    """Same QP as frozen optimizer, with per-step vx in the affine model."""

    def condense_variable(self, current_state, model, curvatures, speeds) -> _CondensedPrediction:
        state = self._state_vector(current_state)
        n = self.config.horizon_steps
        kappa = np.asarray(curvatures, float).reshape(-1)[:n]
        vx = np.asarray(speeds, float).reshape(-1)
        if kappa.size != n or vx.size != n or np.any(vx <= 0):
            raise ValueError("curvature and speed previews must have one positive value per horizon step")
        maps = [np.zeros((4, n), float)]
        offsets = [state]
        for i in range(n):
            continuous = model.linearize(offsets[-1], float(vx[i]), float(kappa[i]))
            discrete = model.discretize(continuous, self.config.sample_time_s)
            state_map = discrete.Ad @ maps[-1]
            state_map[:, i] += discrete.Bd[:, 0]
            offsets.append(discrete.Ad @ offsets[-1] + discrete.gd)
            maps.append(state_map)
        return _CondensedPrediction(tuple(maps), tuple(offsets))

    def solve_variable(self, current_state, model, preview, previous_delta, speed_preview) -> MPCSolution:
        condensed = self.condense_variable(
            current_state, model, np.asarray(preview.curvature_per_m)[: self.config.horizon_steps], speed_preview
        )
        H, f = self._cost_matrices(condensed, previous_delta)
        A, lower, upper = self._constraint_matrices(previous_delta)
        solver = osqp.OSQP()
        started = perf_counter()
        solver.setup(P=sparse.csc_matrix(2 * H), q=f, A=A, l=lower, u=upper,
                     verbose=False, polishing=False, eps_abs=1e-8, eps_rel=1e-8)
        result = solver.solve()
        elapsed = perf_counter() - started
        status = str(result.info.status)
        success = status.lower().startswith("solved") and result.x is not None
        controls = np.asarray(result.x if success else np.zeros(self.config.horizon_steps), float)
        states = self._states_from_solution(condensed, controls)
        return MPCSolution(controls, states, self._objective_value(states, controls, previous_delta),
                           status, elapsed, success,
                           {"iterations": int(result.info.iter), "objective_without_constants": float(result.info.obj_val),
                            "speed_preview_mps": np.asarray(speed_preview, float).tolist()})


class VariableSpeedNominalController:
    def __init__(self, speed_profile: SpeedProfile):
        self.speed_profile = speed_profile
        self._prediction_model = NominalBicyclePredictor()
        self._optimizer = VariableSpeedNominalMPCOptimizer()

    @property
    def optimizer(self):
        return self._optimizer

    def compute_control(self, vehicle_state: VehicleState, reference_path: ReferencePath, previous_delta: float):
        error = global_state_to_error_state(vehicle_state, reference_path)
        cfg = self._optimizer.config
        stations, speeds = self.speed_profile.preview(error.reference.s_m, cfg.horizon_steps, cfg.sample_time_s)
        points = [reference_path.sample(float(s)) for s in stations]
        preview = ReferenceTrajectory(
            X_m=np.array([p.X_m for p in points]), Y_m=np.array([p.Y_m for p in points]),
            psi_rad=np.array([p.psi_rad for p in points]),
            curvature_per_m=np.array([p.curvature_per_m for p in points]),
            arc_length_m=stations, name=f"{reference_path.name}_{self.speed_profile.name}")
        solution = self._optimizer.solve_variable(error, self._prediction_model, preview, previous_delta, speeds)
        return ControllerOutput(float(solution.delta_sequence[0]), error, preview, solution)


class VariableSpeedNominalBridge:
    def __init__(self, reference_path: ReferencePath, speed_profile: SpeedProfile):
        self.reference_path = reference_path
        self.controller = VariableSpeedNominalController(speed_profile)
        self._previous_delta_rad = 0.0

    @property
    def previous_delta_rad(self):
        return self._previous_delta_rad

    def reset(self, previous_delta_rad=0.0):
        self._previous_delta_rad = float(previous_delta_rad)

    def compute_control(self, raw_state):
        out = self.controller.compute_control(chrono_raw_to_vehicle_state(raw_state), self.reference_path,
                                              self._previous_delta_rad)
        self._previous_delta_rad = float(out.delta_rad)
        return out

