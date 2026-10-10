"""Variable-speed composition of the frozen single-pass Full Mean-GMR controller.

The source GMR, cost weights, constraints, horizon, propagation rule and
single-query semantics are unchanged.  This module only supplies the same
20-step Stanford speed preview used by the frozen FT03 Variable-Speed Nominal
controller to the Bicycle rollout and condensed prediction matrices.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
import math

import numpy as np

from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state
from future_work.later_mpc.controller import ControllerOutput
from future_work.later_mpc.types import MPCSolution
from future_work.short_horizon_output_residual.chrono_full_mean_gmr_mpc_smoke.bridge.chrono_full_mean_gmr_bridge import (
    FROZEN_GMR_SHA256,
    FullMeanGMRBridgeDiagnostics,
)
from future_work.short_horizon_output_residual.chrono_full_mean_gmr_mpc_smoke.cached_inference import (
    CachedRolloutAlignedLayer,
)
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import (
    VariableSpeedNominalController,
)
from future_work.short_horizon_output_residual.mpc_integration.full_residual_propagated_gmr_mpc.adapter import (
    FullPropagationIteration,
    FullResidualPropagatedMeanGMRMPC,
    cost_breakdown,
    corrected_states,
)
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import MODEL


class VariableSpeedFullMeanGMRAdapter(FullResidualPropagatedMeanGMRMPC):
    """Frozen Full Mean-GMR equations evaluated with a per-step speed preview."""

    def _maps_variable(self, error_state, curvature, speeds):
        n = self.config.horizon_steps
        mapping = [np.zeros((4, n))]
        offset = [np.asarray(error_state, float)]
        for index, (kappa, vx) in enumerate(zip(curvature[:n], speeds[:n])):
            discrete = self.bicycle.discretize(
                self.bicycle.linearize(offset[-1], float(vx), float(kappa)),
                self.config.sample_time_s,
            )
            state_map = discrete.Ad @ mapping[-1]
            state_map[:, index] += discrete.Bd[:, 0]
            mapping.append(state_map)
            offset.append(discrete.Ad @ offset[-1] + discrete.gd)
        return mapping[1:], offset[1:]

    def _nominal_states_variable(self, error_state, controls, curvature, speeds):
        states = [np.asarray(error_state, float)]
        for delta, kappa, vx in zip(controls, curvature, speeds):
            discrete = self.bicycle.discretize(
                self.bicycle.linearize(states[-1], float(vx), float(kappa)),
                self.config.sample_time_s,
            )
            states.append(discrete.Ad @ states[-1] + discrete.Bd[:, 0] * float(delta) + discrete.gd)
        return np.asarray(states)

    def _quadratic_variable(self, error_state, curvature, speeds, previous_delta, correction):
        n = self.config.horizon_steps
        maps, base = self._maps_variable(error_state, curvature, speeds)
        weights = self.config.state_weights
        q_base = np.array([weights.ey, weights.epsi, weights.vy, weights.r], float)
        hessian = np.zeros((n, n))
        gradient = np.zeros(n)
        for index, (matrix, bias) in enumerate(zip(maps, base)):
            q = q_base * (weights.terminal_scale if index == n - 1 else 1.0)
            adjusted = bias + correction[index]
            hessian += matrix.T @ (q[:, None] * matrix)
            gradient += 2.0 * matrix.T @ (q * adjusted)
        hessian += self.config.input_weights.delta * np.eye(n)
        difference = np.eye(n)
        difference[1:, :-1] -= np.eye(n - 1)
        prior = np.zeros(n)
        prior[0] = -float(previous_delta)
        hessian += self.config.input_weights.delta_rate * difference.T @ difference
        gradient += 2.0 * self.config.input_weights.delta_rate * difference.T @ prior
        return hessian, gradient

    def iterate_variable(self, vehicle_state, error_state, curvature_preview, speed_preview,
                         previous_delta, initial_controls):
        controls = np.asarray(initial_controls, float).reshape(-1)
        curvature = np.asarray(curvature_preview, float).reshape(-1)[: self.config.horizon_steps]
        speeds = np.asarray(speed_preview, float).reshape(-1)[: self.config.horizon_steps]
        state = np.asarray(error_state, float).reshape(4)
        if controls.size != 20 or curvature.size != 20 or speeds.size != 20:
            raise ValueError("FT03 frozen GMR requires aligned 20-step previews")
        if np.any(speeds <= 0) or not np.isfinite(np.c_[controls, curvature, speeds]).all():
            raise ValueError("Non-finite or non-positive variable-speed preview")

        prediction = self.layer.predict(
            vehicle_state,
            {"vx": float(speeds[0]), "steering": float(previous_delta)},
            controls,
            vx_preview=speeds,
        )
        nominal = self._nominal_states_variable(state, controls, curvature, speeds)
        corrected, delta_epsi, delta_ey = corrected_states(
            nominal, prediction.mu_deltaY[:, 1], include_ey=True
        )
        correction = corrected - nominal[1:]
        hessian, gradient = self._quadratic_variable(
            state, curvature, speeds, previous_delta, correction
        )
        updated, elapsed, qp = self._solve(
            hessian, gradient, previous_delta, controls, include_diagnostics=True
        )
        nn_mean, low_density, range_ood, likelihood_mean = self._density(prediction)
        record = FullPropagationIteration(
            iteration=0,
            controls=controls.copy(),
            corrected_cost=float(cost_breakdown(corrected, controls, previous_delta, self.config).total),
            nominal_cost=float(cost_breakdown(nominal[1:], controls, previous_delta, self.config).total),
            control_change_norm=float(np.linalg.norm(updated - controls)),
            mu_deltaY=prediction.mu_deltaY.copy(),
            delta_epsi=delta_epsi.copy(),
            delta_ey=delta_ey.copy(),
            corrected_states=corrected.copy(),
            solve_time_s=elapsed,
            prediction=prediction,
            feature_nn_distance_mean=nn_mean,
            feature_low_density_ratio=low_density,
            feature_range_ood_ratio=range_ood,
            feature_input_log_likelihood_mean=likelihood_mean,
            qp_diagnostics=qp,
        )
        return [record], updated


class VariableSpeedFullMeanGMRBridge:
    """CLASS-B bridge: query once at U0, then freeze correction in one QP."""

    def __init__(self, reference_path, speed_profile, model_path=MODEL):
        self.reference_path = reference_path
        self.controller = VariableSpeedNominalController(speed_profile)
        self.model_path = Path(model_path)
        digest = sha256(self.model_path.read_bytes()).hexdigest()
        if digest.lower() != FROZEN_GMR_SHA256:
            raise RuntimeError(f"Frozen GMR artifact hash mismatch: {digest}")
        self.artifact_sha256 = digest
        self.full_adapter = VariableSpeedFullMeanGMRAdapter(
            self.controller.optimizer.config,
            layer=CachedRolloutAlignedLayer(model_path=self.model_path),
            max_iterations=1,
        )
        self._previous_delta_rad = 0.0
        self.last_diagnostics = None

    @property
    def previous_delta_rad(self):
        return self._previous_delta_rad

    def reset(self, previous_delta_rad=0.0):
        if not math.isfinite(previous_delta_rad):
            raise ValueError("previous steering must be finite")
        self._previous_delta_rad = float(previous_delta_rad)
        self.last_diagnostics = None

    def _validate(self, controls):
        u = np.asarray(controls, float)
        c = self.controller.optimizer.config.steering_constraints
        du = np.diff(np.r_[self._previous_delta_rad, u])
        if u.shape != (20,) or not np.isfinite(u).all():
            raise RuntimeError("Invalid GMR control sequence")
        if np.any(u < c.delta_min_rad - 1e-7) or np.any(u > c.delta_max_rad + 1e-7):
            raise RuntimeError("GMR steering bound violation")
        if np.any(du < c.delta_rate_min_rad_per_sample - 1e-7) or np.any(du > c.delta_rate_max_rad_per_sample + 1e-7):
            raise RuntimeError("GMR steering-rate bound violation")

    def compute_control(self, raw_state):
        started = perf_counter()
        vehicle_state = chrono_raw_to_vehicle_state(raw_state)
        nominal_started = perf_counter()
        nominal = self.controller.compute_control(
            vehicle_state, self.reference_path, self._previous_delta_rad
        )
        nominal_elapsed = perf_counter() - nominal_started
        if not nominal.solution.success:
            raise RuntimeError(f"Nominal QP failed: {nominal.solution.status}")
        u0 = np.asarray(nominal.solution.delta_sequence, float)
        speeds = np.asarray(nominal.solution.diagnostics["speed_preview_mps"], float)
        adapter_started = perf_counter()
        records, ugmr = self.full_adapter.iterate_variable(
            vehicle_state={"vy": raw_state.vy_body, "r": raw_state.yaw_rate_body},
            error_state=nominal.error_state.as_vector(),
            curvature_preview=nominal.reference_preview.curvature_per_m,
            speed_preview=speeds,
            previous_delta=self._previous_delta_rad,
            initial_controls=u0,
        )
        adapter_elapsed = perf_counter() - adapter_started
        iteration = records[0]
        qp = iteration.qp_diagnostics
        if qp is None or not qp.success:
            raise RuntimeError("Frozen GMR external QP did not solve")
        self._validate(ugmr)
        total = perf_counter() - started
        solution = MPCSolution(
            steering_sequence_rad=np.asarray(ugmr, float),
            predicted_error_states=np.asarray(iteration.corrected_states, float),
            objective_value=float(iteration.corrected_cost),
            status=qp.solver_status,
            solve_time_s=total,
            success=True,
            diagnostics={
                "speed_preview_mps": speeds.tolist(),
                "nominal_solver_status": nominal.solution.status,
                "external_qp_status": qp.solver_status,
                "single_pass": True,
                "candidate_control_class": "CLASS_B",
            },
        )
        output = ControllerOutput(float(ugmr[0]), nominal.error_state, nominal.reference_preview, solution)
        self._previous_delta_rad = output.delta_rad
        self.last_diagnostics = FullMeanGMRBridgeDiagnostics(
            raw_state=raw_state,
            nominal_output=nominal,
            nominal_controls=u0.copy(),
            gmr_controls=np.asarray(ugmr, float).copy(),
            iteration=iteration,
            external_qp=qp,
            artifact_sha256=self.artifact_sha256,
            nominal_controller_time_s=float(nominal_elapsed),
            gmr_inference_and_propagation_time_s=max(0.0, adapter_elapsed - iteration.solve_time_s),
            external_qp_time_s=float(iteration.solve_time_s),
            total_controller_time_s=float(total),
        )
        return output
