"""Run the formal minimal Chrono Full Mean-GMR MPC closed-loop smoke test.

No model is trained or altered.  Each controller gets an independently
initialized plant under the already-confirmed historical protocol: direct for
Straight, and settling-probe plus inverse pose alignment for R80/DLC.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
from time import strftime
from typing import Any

import numpy as np

from future_work.chrono_mpc_integration.baseline_reproduction_audit.historical_protocol_wrapper import (
    CALIBRATION_PATH,
    PHYSICS_DT_S,
    MPC_DT_S,
    TARGET_SPEED_MPS,
    InitializationAudit,
    Pose2D,
    _new_initialized_plant,
    _raw_pose,
    _runner_config,
    _settle,
    inverse_settling_pose,
)
from future_work.chrono_mpc_integration.bridge.chrono_mpc_controller import NominalMPCBridge
from future_work.chrono_mpc_integration.bridge.runner import MultiRateChronoMPCRunner
from future_work.chrono_mpc_integration.bridge.speed_controller import LongitudinalSpeedPI
from future_work.chrono_mpc_integration.bridge.steering_adapter import SteeringCalibration
from future_work.chrono_mpc_integration.scenarios.constant_radius import make_constant_radius_left_scenario
from future_work.chrono_mpc_integration.scenarios.double_lane_change import make_double_lane_change_scenario
from future_work.chrono_mpc_integration.scenarios.offset_recovery import make_offset_recovery_scenario
from future_work.short_horizon_output_residual.chrono_full_mean_gmr_mpc_smoke.bridge import (
    FullMeanGMRChronoBridge,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
SCORED_DURATION_S = 0.5
PROTOCOLS = {"Straight": "direct", "R80": "staged", "DLC": "staged"}


def _scenario_by_name(name: str):
    if name == "Straight":
        return replace(make_offset_recovery_scenario(), duration_s=SCORED_DURATION_S)
    if name == "R80":
        return replace(make_constant_radius_left_scenario(), duration_s=SCORED_DURATION_S)
    if name == "DLC":
        return replace(make_double_lane_change_scenario(), duration_s=SCORED_DURATION_S)
    raise ValueError(f"Unknown formal smoke scenario: {name}")


def _prepare_formal_plant(scenario, protocol: str):
    """Reuse the historical wrapper's exact direct/staged settling sequence."""
    calibration = SteeringCalibration.from_json(
        CALIBRATION_PATH, expected_vehicle_model="GolfLike3D_v1"
    )
    requested = Pose2D(scenario.initial_x_m, scenario.initial_y_m, scenario.initial_psi_rad)
    probe_pose = None
    probe_result = None
    formal_requested = requested
    if protocol == "staged":
        probe = _new_initialized_plant(requested, initial_speed_mps=scenario.initial_speed_mps)
        probe_result = _settle(probe, calibration, speed_mps=scenario.initial_speed_mps)
        probe_pose = _raw_pose(probe)
        formal_requested = inverse_settling_pose(probe_pose)
    elif protocol != "direct":
        raise ValueError(f"Unsupported historical protocol: {protocol}")
    plant = _new_initialized_plant(formal_requested, initial_speed_mps=scenario.initial_speed_mps)
    formal_result = _settle(plant, calibration, speed_mps=scenario.initial_speed_mps)
    initialization = InitializationAudit(
        protocol=protocol,
        requested_initial_pose=requested,
        probe_post_settling_pose=probe_pose,
        formal_requested_initial_pose=formal_requested,
        formal_post_settling_pose=_raw_pose(plant),
        probe_settling=None if probe_result is None else asdict(probe_result),
        formal_settling=asdict(formal_result),
    )
    return plant, calibration, initialization


def _finite_sample(sample) -> bool:
    values = (
        sample.X, sample.Y, sample.psi, sample.vx, sample.vy, sample.yaw_rate,
        sample.ey, sample.epsi, sample.reference_X, sample.reference_Y,
        sample.reference_psi, sample.reference_curvature, sample.delta_mpc_rad,
        sample.delta_increment, sample.steering_normalized, sample.throttle,
        sample.brake, sample.solver_time, sample.roll, sample.beta,
        sample.FL_Fz, sample.FR_Fz, sample.RL_Fz, sample.RR_Fz,
    )
    return all(math.isfinite(float(value)) for value in values)


def _array_json(values: np.ndarray) -> str:
    return json.dumps(np.asarray(values, float).tolist(), separators=(",", ":"))


def _gmr_log_row(sample, diagnostics) -> dict[str, Any]:
    iteration = diagnostics.iteration
    return {
        "time_s": sample.time,
        "X_m": sample.X, "Y_m": sample.Y, "psi_rad": sample.psi,
        "vx_mps": sample.vx, "vy_mps": sample.vy, "r_radps": sample.yaw_rate,
        "ey_m": sample.ey, "epsi_rad": sample.epsi,
        "kappa_ref_per_m": sample.reference_curvature,
        "nominal_u0_rad": _array_json(diagnostics.nominal_controls),
        "nominal_first_delta_rad": float(diagnostics.nominal_controls[0]),
        "gmr_u_rad": _array_json(diagnostics.gmr_controls),
        "gmr_first_delta_rad": float(diagnostics.gmr_controls[0]),
        "applied_delta_rad": sample.delta_mpc_rad,
        "mu_delta_ay": _array_json(iteration.mu_deltaY[:, 0]),
        "mu_delta_r": _array_json(iteration.mu_deltaY[:, 1]),
        "delta_epsi": _array_json(iteration.delta_epsi),
        "delta_ey": _array_json(iteration.delta_ey),
        "corrected_error_states": _array_json(iteration.corrected_states),
        "external_qp_status": diagnostics.external_qp.solver_status,
        "external_qp_success": diagnostics.external_qp.success,
        "external_qp_iteration_count": diagnostics.external_qp.iteration_count,
        "external_qp_objective_value": diagnostics.external_qp.objective_value,
        "external_qp_time_s": diagnostics.external_qp_time_s,
        "nominal_controller_time_s": diagnostics.nominal_controller_time_s,
        "gmr_inference_and_propagation_time_s": diagnostics.gmr_inference_and_propagation_time_s,
        "total_gmr_controller_time_s": diagnostics.total_controller_time_s,
        "observed_delta_r_next": None,
        "r_bicycle_one_step_radps": float(iteration.prediction.Y_bicycle[0, 1]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(name: str, controller_kind: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario_by_name(name)
    protocol = PROTOCOLS[name]
    plant, calibration, initialization = _prepare_formal_plant(scenario, protocol)
    if controller_kind == "nominal":
        bridge = NominalMPCBridge(reference_path=scenario.reference_path)
    elif controller_kind == "full_mean_gmr":
        bridge = FullMeanGMRChronoBridge(reference_path=scenario.reference_path)
    else:
        raise ValueError(controller_kind)
    runner = MultiRateChronoMPCRunner(
        plant=plant,
        mpc_bridge=bridge,  # compatible controller-boundary contract
        steering_calibration=calibration,
        speed_controller=LongitudinalSpeedPI(),
        config=_runner_config(),
    )
    rows: list[dict[str, Any]] = []
    failure: str | None = None

    def observe(sample) -> bool:
        nonlocal failure
        if not _finite_sample(sample):
            failure = "non_finite_closed_loop_sample"
            return False
        if sample.wheel_contact_count != 4:
            failure = f"wheel_contact_count={sample.wheel_contact_count}"
            return False
        if not sample.mpc_updated:
            return True
        if not sample.solver_status.lower().startswith("solved"):
            failure = f"solver_status={sample.solver_status}"
            return False
        if controller_kind == "full_mean_gmr":
            diagnostics = bridge.last_diagnostics
            if diagnostics is None:
                failure = "missing_full_mean_gmr_diagnostics"
                return False
            values = (diagnostics.gmr_controls, diagnostics.iteration.mu_deltaY,
                      diagnostics.iteration.delta_epsi, diagnostics.iteration.delta_ey,
                      diagnostics.iteration.corrected_states)
            if not diagnostics.external_qp.success or not all(np.isfinite(v).all() for v in values):
                failure = "non_finite_or_unsolved_external_gmr_qp"
                return False
            if rows:
                rows[-1]["observed_delta_r_next"] = sample.yaw_rate - rows[-1]["r_bicycle_one_step_radps"]
            rows.append(_gmr_log_row(sample, diagnostics))
        else:
            rows.append({
                "time_s": sample.time, "X_m": sample.X, "Y_m": sample.Y,
                "psi_rad": sample.psi, "vx_mps": sample.vx, "vy_mps": sample.vy,
                "r_radps": sample.yaw_rate, "ey_m": sample.ey, "epsi_rad": sample.epsi,
                "applied_delta_rad": sample.delta_mpc_rad, "solver_status": sample.solver_status,
                "solver_time_s": sample.solver_time,
            })
        return True

    try:
        samples = runner.run(scenario.duration_s, step_callback=observe)
    except Exception as exc:  # Record a stopped smoke case; never apply fallback control.
        samples = []
        failure = f"{type(exc).__name__}: {exc}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / f"{controller_kind}_mpc_cycles.csv", rows)
    mpc_rows = rows
    success = failure is None and bool(samples) and len(mpc_rows) == 11
    return {
        "controller": controller_kind,
        "success": success,
        "failure": failure,
        "protocol": protocol,
        "scored_duration_s": scenario.duration_s,
        "mpc_cycle_count": len(mpc_rows),
        "sample_count": len(samples),
        "initialization": asdict(initialization),
        "max_abs_steering_rad": max((abs(float(row["applied_delta_rad"])) for row in mpc_rows), default=None),
        "max_abs_ey_m": max((abs(float(row["ey_m"])) for row in mpc_rows), default=None),
        "max_abs_epsi_rad": max((abs(float(row["epsi_rad"])) for row in mpc_rows), default=None),
        "all_external_qp_solved": (
            None if controller_kind == "nominal" else all(bool(row["external_qp_success"]) for row in mpc_rows)
        ),
    }


def _final_report(summary: dict[str, Any]) -> str:
    gmr = [entry["full_mean_gmr"] for entry in summary["scenarios"].values()]
    all_success = all(entry["success"] for entry in gmr)
    all_solved = all(entry["all_external_qp_solved"] is True for entry in gmr)
    # Timing comes from per-cycle CSV; no performance conclusion is inferred here.
    verdict = "CHRONO_FULL_MEAN_GMR_SMOKE_PASS" if all_success and all_solved else "CHRONO_FULL_MEAN_GMR_SMOKE_FAIL"
    return "\n".join([
        "# Chrono Full Mean-GMR MPC minimal smoke report", "",
        f"1. Full Mean-GMR entered Chrono closed loop: {'yes' if all_success else 'not demonstrated'}.",
        "2. The bridge returns and applies U_GMR[0]; this is regression-tested and logged.",
        f"3. All three minimal scenario runs stable: {'yes' if all_success else 'no'}.",
        f"4. External QP solved at all logged updates: {'yes' if all_solved else 'no'}.",
        "5. Timing is recorded per cycle; this short smoke makes no control-performance claim.",
        "6. Source-GMR correction-floor evidence is recorded only as a next-cycle diagnostic.",
        "7. No frozen algorithm, model, vehicle, configuration, or scenario was modified.",
        "", verdict, "",
    ])


def run_smoke(*, output_root: Path | None = None) -> tuple[dict[str, Any], Path]:
    """Run paired nominal/GMR 0.5-s smokes under the formal protocols."""
    if not math.isclose(PHYSICS_DT_S, 0.002) or not math.isclose(MPC_DT_S, 0.05) or not math.isclose(TARGET_SPEED_MPS, 15.0):
        raise RuntimeError("Frozen historical timing/speed constants do not match the smoke protocol.")
    directory = (output_root or OUTPUT_ROOT) / strftime("%Y%m%d_%H%M%S")
    summary: dict[str, Any] = {
        "experiment": "chrono_full_mean_gmr_mpc_minimal_smoke",
        "scored_duration_s": SCORED_DURATION_S,
        "physics_dt_s": PHYSICS_DT_S,
        "mpc_dt_s": MPC_DT_S,
        "physics_steps_per_mpc_update": 25,
        "scenarios": {},
    }
    for name in ("Straight", "R80", "DLC"):
        scenario_dir = directory / name
        summary["scenarios"][name] = {
            "nominal": _run_one(name, "nominal", scenario_dir),
            "full_mean_gmr": _run_one(name, "full_mean_gmr", scenario_dir),
        }
    summary_path = directory / "summary.json"
    directory.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "final_report.md").write_text(_final_report(summary), encoding="utf-8")
    return summary, directory


def main() -> int:
    summary, directory = run_smoke()
    print(json.dumps({"output_directory": str(directory), "summary": summary}, indent=2, ensure_ascii=False))
    return 0 if all(item["full_mean_gmr"]["success"] for item in summary["scenarios"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
