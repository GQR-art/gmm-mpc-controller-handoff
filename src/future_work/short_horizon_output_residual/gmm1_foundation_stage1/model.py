"""Single-query Stanford GMR output surrogate with explicit plan provenance.

No residual is recursively injected. MPC vy/r remain nominal, as in ClassB.
Delta-ay is diagnostic. This is not a learned Markov state-transition model.
"""
import numpy as np
from future_work.later_mpc.controller import ControllerOutput
from future_work.later_mpc.types import MPCSolution
from future_work.short_horizon_output_residual.ft03_fullterrain_frozen_gmr_exploratory_closed_loop.variable_speed_gmr import (
    VariableSpeedFullMeanGMRBridge, VariableSpeedFullMeanGMRAdapter,
)
from future_work.short_horizon_output_residual.mpc_integration.full_residual_propagated_gmr_mpc.adapter import (
    FullPropagationIteration, cost_breakdown,
)


def propagate(nominal, delta_r, speeds, ts):
    """Legacy right-endpoint yaw quadrature, with per-interval preview speed.

    delta_r is rad/s at future endpoints, never a state increment. Position
    uses the existing trapezoidal kinematic velocity difference.
    """
    dpsi = np.r_[0., np.cumsum(delta_r) * ts]
    dey = []
    accumulated = 0.
    for i, speed in enumerate(speeds):
        velocity = []
        for j in (i, i + 1):
            epsi, vy = nominal[j, 1:3]
            velocity.append(speed * (np.sin(epsi + dpsi[j]) - np.sin(epsi))
                            + vy * (np.cos(epsi + dpsi[j]) - np.cos(epsi)))
        accumulated += .5 * ts * sum(velocity)
        dey.append(accumulated)
    correction = np.zeros((len(delta_r), 4))
    correction[:, 0] = dey
    correction[:, 1] = dpsi[1:]
    return correction


class FoundationAdapter(VariableSpeedFullMeanGMRAdapter):
    def iterate_variable(self, vehicle_state, error_state, curvature_preview,
                         speed_preview, previous_delta, initial_controls):
        u0 = np.asarray(initial_controls, float)
        speeds = np.asarray(speed_preview, float)[:20]
        curvature = np.asarray(curvature_preview, float)[:20]
        prediction = self.layer.predict(vehicle_state,
            {"vx": float(speeds[0]), "steering": float(previous_delta)}, u0,
            vx_preview=speeds)
        nominal0 = self._nominal_states_variable(error_state, u0, curvature, speeds)
        correction = propagate(nominal0, prediction.mu_deltaY[:, 1], speeds,
                               self.config.sample_time_s)
        corrected0 = nominal0[1:] + correction
        hessian, gradient = self._quadratic_variable(
            error_state, curvature, speeds, previous_delta, correction)
        u1, elapsed, qp = self._solve(hessian, gradient, previous_delta, u0,
                                     include_diagnostics=True)
        nn, low, ood, likelihood = self._density(prediction)
        record = FullPropagationIteration(
            0, u0.copy(), cost_breakdown(corrected0, u0, previous_delta, self.config).total,
            cost_breakdown(nominal0[1:], u0, previous_delta, self.config).total,
            float(np.linalg.norm(u1-u0)), prediction.mu_deltaY.copy(),
            correction[:, 1], correction[:, 0], corrected0, elapsed, prediction,
            nn, low, ood, likelihood, qp)
        self.nominal0 = nominal0
        self.correction = correction
        self.nominal1 = self._nominal_states_variable(error_state, u1, curvature, speeds)
        self.selected_prediction = self.nominal1[1:] + correction
        return [record], u1


class FoundationBridge(VariableSpeedFullMeanGMRBridge):
    def __init__(self, path, profile):
        super().__init__(path, profile)
        # Preserve the initialized model/cache; no extra inference or training.
        self.full_adapter.__class__ = FoundationAdapter

    def compute_control(self, raw_state):
        previous = self.previous_delta_rad
        output = super().compute_control(raw_state)
        states = self.full_adapter.selected_prediction.copy()
        # Superclass reports U0's states beside U1; record actual QP prediction.
        solution = MPCSolution(np.asarray(output.solution.delta_sequence), states,
            cost_breakdown(states, output.solution.delta_sequence, previous,
                           self.controller.optimizer.config).total,
            output.solution.status, output.solution.solve_time_s, output.solution.success,
            {**output.solution.diagnostics, "prediction_semantics": "F0(U1)+C(U0)"})
        return ControllerOutput(output.delta_rad, output.error_state, output.reference_preview, solution)
