"""External full-trajectory Mean-GMR MPC adapter.

The frozen nominal MPC state model, weights, and constraints are never changed.
For a candidate steering sequence, GMR's additive ``delta_r`` is propagated by
the already-audited later-MPC error kinematics.  The resulting ``delta_ey`` and
``delta_epsi`` are frozen offsets for one external QP update.  Sequential mode
recomputes those offsets for the newly selected candidate, at most three times.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import osqp
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from future_work.later_mpc.nominal_bicycle_predictor import NominalBicyclePredictor
from future_work.short_horizon_output_residual.residual_trajectory_propagation_audit.run_audit import (
    corrected_states,
    cost_breakdown,
)
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import (
    RolloutAlignedPrediction,
    RolloutAlignedSequentialLayer,
)


@dataclass(frozen=True)
class FullPropagationIteration:
    iteration: int
    controls: np.ndarray
    corrected_cost: float
    nominal_cost: float
    control_change_norm: float
    mu_deltaY: np.ndarray
    delta_epsi: np.ndarray
    delta_ey: np.ndarray
    corrected_states: np.ndarray
    solve_time_s: float
    prediction: RolloutAlignedPrediction
    feature_nn_distance_mean: float
    feature_low_density_ratio: float
    feature_range_ood_ratio: float
    feature_input_log_likelihood_mean: float
    qp_diagnostics: "ExternalQPSolveDiagnostics | None" = None


@dataclass(frozen=True)
class ExternalQPSolveDiagnostics:
    """OSQP metadata for the unchanged external residual-aware QP."""

    solver_status: str
    success: bool
    iteration_count: int | None
    objective_value: float | None


class FullResidualPropagatedMeanGMRMPC:
    """One-or-three external QP updates using full ``delta_r -> epsi -> ey`` offsets."""

    feature_names = ("vx", "r_bicycle", "ay_bicycle", "delta_future", "delta_dot_future", "prediction_time_s")

    def __init__(self, config, layer: RolloutAlignedSequentialLayer | None = None,
                 max_iterations: int = 1, threshold: float = 1e-4):
        if config.horizon_steps != 20:
            raise ValueError("The frozen rollout-aligned model requires 20 MPC steps")
        self.config = config
        self.layer = layer or RolloutAlignedSequentialLayer()
        self.bicycle = NominalBicyclePredictor()
        self.max_iterations = int(max_iterations)
        self.threshold = float(threshold)
        training = pd.read_csv(self.layer.training_path, usecols=list(self.feature_names)).to_numpy(float)
        self._feature_mean = training.mean(axis=0)
        self._feature_std = np.maximum(training.std(axis=0), 1e-9)
        standardized = (training - self._feature_mean) / self._feature_std
        self._feature_tree = cKDTree(standardized)
        # The train-to-train nearest-neighbour 99th percentile defines a
        # density warning; it is diagnostic only and never gates control.
        distances, _ = self._feature_tree.query(standardized, k=2)
        self._nn_reference_p99 = float(np.quantile(distances[:, 1], 0.99))
        self._input_mean = self.layer.model.scaler.mean_[:len(self.feature_names)]
        self._input_scale = self.layer.model.scaler.scale_[:len(self.feature_names)]
        self._gmm_weights = np.asarray(self.layer.model.gmm.weights_, float)
        self._gmm_input_means = np.asarray(self.layer.model.gmm.means_[:, :len(self.feature_names)], float)
        self._gmm_input_covariances = np.asarray(
            self.layer.model.gmm.covariances_[:, :len(self.feature_names), :len(self.feature_names)], float
        )
        # A train-only 1% lower-tail marginal GMM likelihood threshold is more
        # meaningful than an exact-nearest-row threshold for continuous MPC
        # rollouts.  It diagnoses density only; it is never used as a gate.
        self._input_likelihood_threshold = float(np.quantile(self._input_log_likelihood(training), 0.01))

    def _maps(self, error_state: np.ndarray, curvature: np.ndarray):
        n = self.config.horizon_steps
        mapping, offset = [np.zeros((4, n))], [np.asarray(error_state, float)]
        for index, kappa in enumerate(np.asarray(curvature, float)[:n]):
            discrete = self.bicycle.discretize(
                self.bicycle.linearize(offset[-1], self.config.longitudinal_speed_mps, float(kappa)),
                self.config.sample_time_s,
            )
            state_map = discrete.Ad @ mapping[-1]
            state_map[:, index] += discrete.Bd[:, 0]
            mapping.append(state_map)
            offset.append(discrete.Ad @ offset[-1] + discrete.gd)
        return mapping[1:], offset[1:]

    def _nominal_states(self, error_state: np.ndarray, controls: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        return self.bicycle.predict(
            error_state, controls, np.asarray(curvature, float)[:self.config.horizon_steps],
            self.config.longitudinal_speed_mps, self.config.sample_time_s,
        )

    def _input_log_likelihood(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, float) - self._input_mean) / self._input_scale
        terms = []
        d = standardized.shape[1]
        for weight, mean, covariance in zip(self._gmm_weights, self._gmm_input_means, self._gmm_input_covariances):
            covariance = covariance + np.eye(d) * 1e-12
            centered = standardized - mean
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise FloatingPointError("Non-positive GMM marginal covariance")
            mahalanobis = np.sum(np.linalg.solve(covariance, centered.T).T * centered, axis=1)
            terms.append(np.log(weight) - 0.5 * (d * np.log(2.0 * np.pi) + logdet + mahalanobis))
        return logsumexp(np.vstack(terms), axis=0)

    def _density(self, prediction: RolloutAlignedPrediction) -> tuple[float, float, float, float]:
        features = prediction.gmr_features
        distances, _ = self._feature_tree.query((features - self._feature_mean) / self._feature_std, k=1)
        likelihood = self._input_log_likelihood(features)
        range_audit = self.layer.audit_prediction_features(prediction)
        range_ratio = float(np.mean([
            range_audit[name]["ood_ratio"] for name in self.feature_names
        ]))
        return float(np.mean(distances)), float(np.mean(likelihood < self._input_likelihood_threshold)), range_ratio, float(np.mean(likelihood))

    def _quadratic(self, error_state: np.ndarray, curvature: np.ndarray, previous_delta: float,
                   correction: np.ndarray):
        n = self.config.horizon_steps
        maps, base = self._maps(error_state, curvature)
        weights = self.config.state_weights
        q_base = np.array([weights.ey, weights.epsi, weights.vy, weights.r], dtype=float)
        hessian = np.zeros((n, n)); gradient = np.zeros(n)
        for index, (matrix, bias) in enumerate(zip(maps, base)):
            q = q_base * (weights.terminal_scale if index == n - 1 else 1.0)
            adjusted_bias = bias + correction[index]
            hessian += matrix.T @ (q[:, None] * matrix)
            gradient += 2.0 * matrix.T @ (q * adjusted_bias)
        hessian += self.config.input_weights.delta * np.eye(n)
        difference = np.eye(n); difference[1:, :-1] -= np.eye(n - 1)
        prior = np.zeros(n); prior[0] = -float(previous_delta)
        hessian += self.config.input_weights.delta_rate * difference.T @ difference
        gradient += 2.0 * self.config.input_weights.delta_rate * difference.T @ prior
        return hessian, gradient

    def _solve(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        previous_delta: float,
        fallback: np.ndarray,
        *,
        include_diagnostics: bool = False,
    ):
        """Solve unchanged QP; diagnostics are optional to preserve legacy callers."""
        n = self.config.horizon_steps
        bounds = self.config.steering_constraints
        difference = np.eye(n); difference[1:, :-1] -= np.eye(n - 1)
        prior = np.zeros(n); prior[0] = -float(previous_delta)
        matrix = sparse.csc_matrix(np.vstack([np.eye(n), difference]))
        lower = np.r_[np.full(n, bounds.delta_min_rad), np.full(n, bounds.delta_rate_min_rad_per_sample) - prior]
        upper = np.r_[np.full(n, bounds.delta_max_rad), np.full(n, bounds.delta_rate_max_rad_per_sample) - prior]
        solver = osqp.OSQP(); started = perf_counter()
        solver.setup(P=sparse.csc_matrix(2.0 * hessian), q=gradient, A=matrix, l=lower, u=upper,
                     verbose=False, polishing=False, eps_abs=1e-8, eps_rel=1e-8)
        solved = solver.solve()
        controls = np.asarray(solved.x if solved.x is not None else fallback, dtype=float)
        elapsed = float(perf_counter() - started)
        if not include_diagnostics:
            return controls, elapsed
        status = str(solved.info.status)
        diagnostics = ExternalQPSolveDiagnostics(
            solver_status=status,
            success=bool(status.lower().startswith("solved") and solved.x is not None),
            iteration_count=(
                int(solved.info.iter) if getattr(solved.info, "iter", None) is not None else None
            ),
            objective_value=(
                float(solved.info.obj_val)
                if getattr(solved.info, "obj_val", None) is not None
                else None
            ),
        )
        return controls, elapsed, diagnostics

    def iterate(self, vehicle_state: dict, error_state: np.ndarray, curvature_preview: np.ndarray,
                previous_delta: float, initial_controls: np.ndarray) -> tuple[list[FullPropagationIteration], np.ndarray]:
        """Run full propagation with no plant look-ahead or state feedback."""
        controls = np.asarray(initial_controls, float).reshape(-1)
        if controls.size != self.config.horizon_steps:
            raise ValueError("initial_controls must contain 20 values")
        state = np.asarray(error_state, float).reshape(4)
        curvature = np.asarray(curvature_preview, float).reshape(-1)[:self.config.horizon_steps]
        condition = {"vx": self.config.longitudinal_speed_mps, "steering": float(previous_delta)}
        records: list[FullPropagationIteration] = []
        for index in range(self.max_iterations):
            prediction = self.layer.predict(vehicle_state, condition, controls)
            nominal_with_initial = self._nominal_states(state, controls, curvature)
            corrected, delta_epsi, delta_ey = corrected_states(
                nominal_with_initial, prediction.mu_deltaY[:, 1], include_ey=True
            )
            correction = corrected - nominal_with_initial[1:]
            hessian, gradient = self._quadratic(state, curvature, previous_delta, correction)
            corrected_cost = cost_breakdown(corrected, controls, previous_delta, self.config).total
            nominal_cost = cost_breakdown(nominal_with_initial[1:], controls, previous_delta, self.config).total
            updated, elapsed, qp_diagnostics = self._solve(
                hessian, gradient, previous_delta, controls, include_diagnostics=True
            )
            nn_mean, low_density, range_ood, likelihood_mean = self._density(prediction)
            records.append(FullPropagationIteration(
                index, controls.copy(), float(corrected_cost), float(nominal_cost),
                float(np.linalg.norm(updated - controls)), prediction.mu_deltaY.copy(), delta_epsi.copy(),
                delta_ey.copy(), corrected.copy(), elapsed, prediction, nn_mean, low_density, range_ood,
                likelihood_mean, qp_diagnostics,
            ))
            controls = updated
            if records[-1].control_change_norm < self.threshold:
                break
        return records, controls
