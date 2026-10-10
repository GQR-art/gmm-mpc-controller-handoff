"""Chrono-to-frozen Full Mean-GMR MPC boundary adapter.

This module owns no plant, path, bicycle, GMR, or QP algorithm.  It composes
the frozen NominalMPCController with the already-audited external full
residual-propagated QP and exposes the same ``compute_control`` shape as the
existing Chrono NominalMPCBridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state
from future_work.chrono_mpc_integration.bridge.types import ChronoRawState
from future_work.later_mpc.controller import ControllerOutput, NominalMPCController
from future_work.later_mpc.reference_path import ReferencePath
from future_work.later_mpc.types import MPCSolution
from future_work.short_horizon_output_residual.mpc_integration.full_residual_propagated_gmr_mpc.adapter import (
    ExternalQPSolveDiagnostics,
    FullPropagationIteration,
    FullResidualPropagatedMeanGMRMPC,
)
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import MODEL


FROZEN_GMR_SHA256 = "aa763629ba6c395c76dbbe96a005f98e03336c3a246922dbafbbe191f60c8659"


@dataclass(frozen=True, slots=True)
class FullMeanGMRBridgeDiagnostics:
    """Per-update evidence kept outside the frozen controller contracts."""

    raw_state: ChronoRawState
    nominal_output: ControllerOutput
    nominal_controls: np.ndarray
    gmr_controls: np.ndarray
    iteration: FullPropagationIteration
    external_qp: ExternalQPSolveDiagnostics
    artifact_sha256: str
    nominal_controller_time_s: float
    gmr_inference_and_propagation_time_s: float
    external_qp_time_s: float
    total_controller_time_s: float


class FullMeanGMRChronoBridge:
    """Single-pass Full Mean-GMR controller output for the existing runner.

    The nominal controller is run once to provide ``U0`` and its frozen
    path-relative state/reference-preview contracts.  The Full adapter then
    performs exactly one external QP update.  If either solver fails, this
    bridge raises instead of silently falling back to Nominal MPC.
    """

    def __init__(
        self,
        *,
        reference_path: ReferencePath,
        controller: NominalMPCController | None = None,
        full_adapter: FullResidualPropagatedMeanGMRMPC | None = None,
        initial_previous_delta_rad: float = 0.0,
        model_path: Path | str = MODEL,
    ) -> None:
        if not isinstance(reference_path, ReferencePath):
            raise TypeError("reference_path must be a later_mpc ReferencePath.")
        if not math.isfinite(initial_previous_delta_rad):
            raise ValueError("initial_previous_delta_rad must be finite.")

        self.reference_path = reference_path
        self.controller = controller or NominalMPCController()
        self.model_path = Path(model_path)
        self.artifact_sha256 = self._verify_frozen_artifact(self.model_path)
        self.full_adapter = full_adapter or FullResidualPropagatedMeanGMRMPC(
            self.controller.optimizer.config, max_iterations=1
        )
        if self.full_adapter.max_iterations != 1:
            raise ValueError("Chrono smoke test is frozen to single-pass Mean-GMR MPC.")
        if self.full_adapter.config is not self.controller.optimizer.config:
            # Equal values alone are insufficient: this ensures the adapter and
            # nominal controller read precisely the same frozen configuration.
            raise ValueError("Full adapter must use the nominal controller's MPCConfig object.")
        self._previous_delta_rad = float(initial_previous_delta_rad)
        self.last_diagnostics: FullMeanGMRBridgeDiagnostics | None = None

    @staticmethod
    def _verify_frozen_artifact(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen GMR artifact is missing: {path}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest.lower() != FROZEN_GMR_SHA256:
            raise RuntimeError(
                "Frozen GMR artifact hash mismatch: expected "
                f"{FROZEN_GMR_SHA256}, got {digest}."
            )
        return digest

    @property
    def previous_delta_rad(self) -> float:
        """Previously applied MPC road-wheel command, in radians."""
        return self._previous_delta_rad

    def reset(self, previous_delta_rad: float = 0.0) -> None:
        if not math.isfinite(previous_delta_rad):
            raise ValueError("previous_delta_rad must be finite.")
        self._previous_delta_rad = float(previous_delta_rad)
        self.last_diagnostics = None

    def _validate_road_wheel_controls(self, controls: np.ndarray) -> None:
        constraints = self.controller.optimizer.config.steering_constraints
        u = np.asarray(controls, dtype=float).reshape(-1)
        if u.shape != (self.controller.optimizer.config.horizon_steps,) or not np.isfinite(u).all():
            raise RuntimeError("Full Mean-GMR QP returned non-finite or wrongly sized controls.")
        tolerance = 1e-7  # OSQP numerical feasibility tolerance; no constraint relaxation.
        if np.any(u < constraints.delta_min_rad - tolerance) or np.any(u > constraints.delta_max_rad + tolerance):
            raise RuntimeError("Full Mean-GMR control violates frozen road-wheel bounds.")
        increments = np.diff(np.r_[self._previous_delta_rad, u])
        if (
            np.any(increments < constraints.delta_rate_min_rad_per_sample - tolerance)
            or np.any(increments > constraints.delta_rate_max_rad_per_sample + tolerance)
        ):
            raise RuntimeError("Full Mean-GMR control violates frozen road-wheel rate bounds.")

    def compute_control(self, raw_state: ChronoRawState) -> ControllerOutput:
        """Return ``U_GMR[0]``; no future Chrono state is accepted or used."""
        started = perf_counter()
        vehicle_state = chrono_raw_to_vehicle_state(raw_state)
        nominal_started = perf_counter()
        nominal = self.controller.compute_control(
            vehicle_state=vehicle_state,
            reference_path=self.reference_path,
            previous_delta=self._previous_delta_rad,
        )
        nominal_elapsed = float(perf_counter() - nominal_started)
        if not nominal.solution.success or not nominal.solution.status.lower().startswith("solved"):
            raise RuntimeError(f"Frozen Nominal MPC did not solve: {nominal.solution.status}")
        nominal_controls = np.asarray(nominal.solution.delta_sequence, dtype=float).copy()

        adapter_started = perf_counter()
        records, gmr_controls = self.full_adapter.iterate(
            vehicle_state={"vy": raw_state.vy_body, "r": raw_state.yaw_rate_body},
            error_state=nominal.error_state.as_vector(),
            curvature_preview=nominal.reference_preview.curvature_per_m,
            previous_delta=self._previous_delta_rad,
            initial_controls=nominal_controls,
        )
        adapter_elapsed = float(perf_counter() - adapter_started)
        if len(records) != 1:
            raise RuntimeError("Single-pass smoke bridge received an unexpected iteration count.")
        iteration = records[0]
        external_qp = iteration.qp_diagnostics
        if external_qp is None or not external_qp.success:
            status = "missing diagnostics" if external_qp is None else external_qp.solver_status
            raise RuntimeError(f"Full Mean-GMR external QP did not solve: {status}")
        self._validate_road_wheel_controls(gmr_controls)
        total_elapsed = float(perf_counter() - started)

        solution = MPCSolution(
            steering_sequence_rad=np.asarray(gmr_controls, dtype=float).copy(),
            predicted_error_states=np.asarray(iteration.corrected_states, dtype=float).copy(),
            objective_value=float(iteration.corrected_cost),
            status=external_qp.solver_status,
            solve_time_s=total_elapsed,
            success=True,
            diagnostics={
                "nominal_solver_status": nominal.solution.status,
                "nominal_solver_success": nominal.solution.success,
                "nominal_objective_value": nominal.solution.objective_value,
                "nominal_controller_time_s": nominal_elapsed,
                "external_qp_status": external_qp.solver_status,
                "external_qp_success": external_qp.success,
                "external_qp_iteration_count": external_qp.iteration_count,
                "external_qp_objective_value": external_qp.objective_value,
                "external_qp_time_s": iteration.solve_time_s,
                "single_pass": True,
            },
        )
        output = ControllerOutput(
            delta_rad=float(gmr_controls[0]),
            error_state=nominal.error_state,
            reference_preview=nominal.reference_preview,
            solution=solution,
        )
        self._previous_delta_rad = output.delta_rad
        self.last_diagnostics = FullMeanGMRBridgeDiagnostics(
            raw_state=raw_state,
            nominal_output=nominal,
            nominal_controls=nominal_controls,
            gmr_controls=np.asarray(gmr_controls, dtype=float).copy(),
            iteration=iteration,
            external_qp=external_qp,
            artifact_sha256=self.artifact_sha256,
            nominal_controller_time_s=nominal_elapsed,
            gmr_inference_and_propagation_time_s=max(
                0.0, adapter_elapsed - float(iteration.solve_time_s)
            ),
            external_qp_time_s=float(iteration.solve_time_s),
            total_controller_time_s=total_elapsed,
        )
        return output
