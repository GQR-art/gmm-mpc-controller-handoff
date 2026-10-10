"""Execute the preregistered FT03 full-terrain Nominal/Frozen-GMR pair."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd

from future_work.chrono_mpc_integration.bridge.speed_controller import LongitudinalSpeedPI
from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state
from future_work.chrono_mpc_integration.scenarios import ChronoMPCScenario
from future_work.later_mpc.error_state import global_state_to_error_state
from future_work.short_horizon_output_residual.chrono_full_mean_gmr_mpc_smoke.cached_inference import CachedRolloutAlignedLayer
from future_work.short_horizon_output_residual.ft03_high_speed_yaw_dynamics_audit.run_diagnostic_replay import force_snapshot
from future_work.short_horizon_output_residual.ft03_original_speed_friction_sensitivity.run_experiment import (
    digest,
    initialized_plant,
    protected_hashes,
)
from future_work.short_horizon_output_residual.ft03_reference_nominal_failure_localization.reference_path import build_path
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.profiles import stanford_profile
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import VariableSpeedNominalBridge
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import MODEL, TRAINING

from .variable_speed_gmr import VariableSpeedFullMeanGMRBridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent / "ft03_terrain_coverage_rebuild"
TERRAIN = SOURCE / "configs/C-MU100-FULL_terrain.json"
HISTORICAL = SOURCE / "logs/C-MU100-FULL/physics_log.csv"
DT = 0.002
TS = 0.05
MAX_TIME_S = 35.0
WHEELS = ("FL", "FR", "RL", "RR")
EXPECTED_GMR_SHA256 = "AA763629BA6C395C76DBBE96A005F98E03336C3A246922DBAFBBE191F60C8659"
CORE_CHANNELS = (
    "time_s", "X_m", "Y_m", "psi_rad", "vx_mps", "vy_mps", "r_radps",
    "ey_m", "epsi_rad", "projection_station_m", "delta_command_rad",
    "delta_actual_rad", "throttle", "brake",
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf8")


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest().upper()


def frozen_hashes():
    values = protected_hashes()
    extra = [
        MODEL, TRAINING, TERRAIN,
        HERE.parent / "mpc_integration/full_residual_propagated_gmr_mpc/adapter.py",
        HERE.parent / "mpc_integration/rollout_aligned_sequential_layer.py",
        HERE.parent / "chrono_full_mean_gmr_mpc_smoke/cached_inference.py",
        HERE.parent / "chrono_full_mean_gmr_mpc_smoke/bridge/chrono_full_mean_gmr_bridge.py",
        HERE.parent / "ft03_variable_speed_nominal_validation/variable_speed_mpc.py",
    ]
    values.update({str(p): _sha(p) for p in extra})
    return values


def prepare():
    for name in ("audits", "configs", "scripts", "logs", "metrics", "figures", "reports", "manifests", "visualization"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    terrain = json.loads(TERRAIN.read_text(encoding="utf8"))
    geometry = terrain["Patches"][0]["Geometry"]["Dimensions"]
    mu = terrain["Patches"][0]["Contact Material"]["Coefficient of Friction"]
    if geometry != [480, 200, 2] or mu != 1.0:
        raise RuntimeError("Verified full terrain is not 480x200x2 m at mu=1.0")
    if _sha(MODEL) != EXPECTED_GMR_SHA256:
        raise RuntimeError("BLOCKED_BY_GMR_INTERFACE_MISMATCH: artifact hash")
    before = frozen_hashes()
    before_path = HERE / "manifests/artifact_hashes_before.json"
    if before_path.exists() and json.loads(before_path.read_text(encoding="utf8")) != before:
        raise RuntimeError("Frozen inputs changed since preregistration")
    _json(before_path, before)
    manifest = {
        "status": "PREREGISTERED_BEFORE_NEW_OUTCOME_RUNS",
        "experiment": "FT03_FULLTERRAIN_ORIGINAL_SPEED_FROZEN_GMR_EXPLORATORY_CLOSED_LOOP",
        "cases": ["FT03_FULL_MU100_NOMINAL", "FT03_FULL_MU100_FROZEN_GMR"],
        "only_design_difference": "frozen GMR mean correction disabled/enabled",
        "terrain": {"path": str(TERRAIN), "sha256": _sha(TERRAIN), "geometry_m": geometry, "mu": mu},
        "gmr": {"path": str(MODEL), "sha256": _sha(MODEL), "K": 3,
                "features": ["vx", "r_bicycle", "ay_bicycle", "delta_future", "delta_dot_future", "prediction_time_s"],
                "targets": ["delta_ay", "delta_r"], "correction_sign": "additive", "covariance_control_use": False},
        "controller_semantics": "CLASS_B: one query at Nominal U0 per cycle; correction frozen in external QP",
        "timing": {"physics_dt_s": DT, "control_dt_s": TS, "horizon_steps": 20, "max_time_s": MAX_TIME_S},
        "constraints": {"delta_rad": [-0.35, 0.35], "delta_increment_rad_per_sample": [-0.03, 0.03]},
        "stop_rules": ["artifact/interface mismatch", "nonfinite GMR", "Nominal reproduction drift", "existing plant/control safety stop"],
        "no_tuning": True,
    }
    _json(HERE / "manifests/experiment_manifest.json", manifest)
    return manifest


def _terrain_fields(plant, fields):
    config = json.loads(TERRAIN.read_text(encoding="utf8"))
    cx, cy, _ = config["Patches"][0]["Location"]
    sx, sy, _ = config["Patches"][0]["Geometry"]["Dimensions"]
    for wheel in WHEELS:
        x, y = float(fields[f"{wheel}_point_X_m"]), float(fields[f"{wheel}_point_Y_m"])
        inside = cx - sx / 2 <= x <= cx + sx / 2 and cy - sy / 2 <= y <= cy + sy / 2
        queried = float(plant._terrain.GetCoefficientFriction(plant._chrono.ChVector3d(x, y, 0.0)))
        fields[f"{wheel}_terrain_inside"] = inside
        fields[f"{wheel}_terrain_queried_mu"] = queried
        fields[f"{wheel}_terrain_mu_matches"] = bool(inside and abs(queried - 1.0) < 1e-6)
    return fields


def _prediction_rows(time_s, station_s, raw, diagnostics):
    pred = diagnostics.iteration.prediction
    covariance = pred.Sigma_deltaY
    rows = []
    for h in range(20):
        rows.append({
            "time": time_s, "station_s": station_s, "horizon_step": h + 1,
            "tau": float(pred.prediction_time_s[h]), "vx": float(pred.gmr_features[h, 0]),
            "r_bicycle": float(pred.Y_bicycle[h, 1]), "ay_bicycle": float(pred.Y_bicycle[h, 0]),
            "delta": float(pred.gmr_features[h, 3]), "delta_dot": float(pred.gmr_features[h, 4]),
            "mu_Delta_r": float(pred.mu_deltaY[h, 1]), "mu_Delta_ay": float(pred.mu_deltaY[h, 0]),
            "sigma_Delta_r": float(math.sqrt(max(0.0, covariance[h, 1, 1]))),
            "sigma_Delta_ay": float(math.sqrt(max(0.0, covariance[h, 0, 0]))),
            "r_corrected": float(pred.Y_corrected[h, 1]), "ay_corrected": float(pred.Y_corrected[h, 0]),
            "r_chrono": np.nan, "ay_chrono": np.nan, "Delta_r_target": np.nan, "Delta_ay_target": np.nan,
            "protocol": "same-closed-loop future observation; controls replan after h=1",
        })
    return rows


def run_case(method):
    if method not in {"Nominal", "GMR"}:
        raise ValueError(method)
    folder = HERE / "logs" / method
    if folder.exists():
        raise FileExistsError(f"Never overwrite prior run: {folder}")
    before = json.loads((HERE / "manifests/artifact_hashes_before.json").read_text(encoding="utf8"))
    if before != frozen_hashes():
        raise RuntimeError("Frozen artifact drift before run")
    built = build_path(77.5, 82.45)
    profile = stanford_profile(built.frame)
    scenario = ChronoMPCScenario(0.0, 0.0, 0.0, float(profile.sample(0.0)), MAX_TIME_S, built.path)
    plant, calibration, initialization = initialized_plant(scenario, TERRAIN)
    if plant.config.rigid_terrain_json.resolve() != TERRAIN.resolve():
        raise RuntimeError("Full-terrain config did not reach plant")
    bridge = VariableSpeedNominalBridge(built.path, profile) if method == "Nominal" else VariableSpeedFullMeanGMRBridge(built.path, profile)
    bridge.reset(0.0)
    pi = LongitudinalSpeedPI()
    pi.reset()
    ratio = int(round(TS / DT))
    normalized = 0.0
    delta_inc = 0.0
    output = None
    physics, predictions = [], []
    failure = None
    start = plant.get_raw_state().time
    for step in range(int(MAX_TIME_S / DT) + 1):
        raw = plant.get_raw_state()
        projected = built.path.project(raw.X, raw.Y)
        if step % ratio == 0:
            previous = bridge.previous_delta_rad
            try:
                output = bridge.compute_control(raw)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                break
            delta_inc = output.delta_rad - previous
            normalized = calibration.delta_rad_to_normalized(output.delta_rad)
            if method == "GMR":
                diag = bridge.last_diagnostics
                if diag is None or not np.isfinite(diag.iteration.prediction.mu_deltaY).all():
                    failure = "nonfinite_or_missing_gmr_diagnostics"
                    break
                predictions.extend(_prediction_rows(raw.time - start, projected.s_m, raw, diag))
        else:
            delta_inc = 0.0
        speed_ref = float(profile.sample(projected.s_m))
        lon = pi.update(vx_reference_mps=speed_ref, vx_measured_mps=raw.vx_body, dt_s=DT)
        err = global_state_to_error_state(chrono_raw_to_vehicle_state(raw), built.path)
        wheels = plant.get_wheel_diagnostics()
        suspension = plant.get_suspension_diagnostics()
        wmap = {w.wheel_id: w for w in wheels}
        front_angles = [w.road_wheel_angle_rad for w in wheels if w.axle_index == 0]
        row = {
            "time_s": raw.time - start, "X_m": raw.X, "Y_m": raw.Y, "psi_rad": raw.psi,
            "vx_mps": raw.vx_body, "vy_mps": raw.vy_body, "r_radps": raw.yaw_rate_body,
            "ey_m": err.ey_m, "epsi_rad": err.epsi_rad, "projection_station_m": projected.s_m,
            "remaining_path_m": built.path.length_m - projected.s_m,
            "reference_X_m": projected.X_m, "reference_Y_m": projected.Y_m,
            "reference_psi_rad": projected.psi_rad, "reference_curvature_per_m": projected.curvature_per_m,
            "speed_reference_mps": speed_ref, "reference_r_radps": speed_ref * projected.curvature_per_m,
            "actual_path_r_demand_radps": raw.vx_body * projected.curvature_per_m,
            "ay_reference_demand_mps2": speed_ref ** 2 * projected.curvature_per_m,
            "delta_command_rad": output.delta_rad, "delta_actual_rad": float(math.atan(np.mean(np.tan(front_angles)))),
            "delta_increment_rad": delta_inc, "steering_normalized": normalized,
            "throttle": lon.throttle, "brake": lon.braking,
            "solver_status": output.solution.status, "solver_time_s": output.solution.solve_time_s,
            "mpc_updated": step % ratio == 0, "roll_rad": plant.get_chassis_roll_pitch_rad()[0],
            "beta_rad": math.atan2(raw.vy_body, raw.vx_body),
            "ay_com_mps2": plant.get_aggregate_com_acceleration_body_mps2()[1],
            "wheel_contact_count": sum(x.contact for x in suspension),
            "formal_interval": built.formal_start_s <= projected.s_m <= built.formal_end_s,
            "requested_mu": 1.0, "loaded_terrain_config_mu": 1.0, "controller": method,
        }
        for wid in WHEELS:
            w = wmap[wid]
            row.update({
                f"{wid}_Fz_N": w.force_z_tire_N, f"{wid}_slip_angle_rad": w.slip_angle_rad,
                f"{wid}_lateral_slip_sy": w.lateral_slip_sy, f"{wid}_longitudinal_slip_sx": w.longitudinal_slip_sx,
                f"{wid}_contact": bool(next(x.contact for x in suspension if x.wheel_id == wid)),
            })
        row.update(_terrain_fields(plant, force_snapshot(plant)))
        if step % ratio == 0:
            controls = np.asarray(output.solution.delta_sequence, float)
            speeds = np.asarray(output.solution.diagnostics["speed_preview_mps"], float)
            for j in range(20):
                row[f"candidate_delta_{j + 1:02d}_rad"] = controls[j]
                row[f"preview_speed_{j + 1:02d}_mps"] = speeds[j]
            if method == "GMR":
                diag = bridge.last_diagnostics
                row.update({
                    "nominal_first_delta_rad": float(diag.nominal_controls[0]),
                    "gmr_first_delta_rad": float(diag.gmr_controls[0]),
                    "gmr_control_change_rad": float(diag.gmr_controls[0] - diag.nominal_controls[0]),
                    "gmr_feature_low_density_ratio": float(diag.iteration.feature_low_density_ratio),
                    "gmr_feature_range_ood_ratio": float(diag.iteration.feature_range_ood_ratio),
                    "external_qp_success": bool(diag.external_qp.success),
                })
        physics.append(row)
        if projected.s_m >= built.formal_end_s or step == int(MAX_TIME_S / DT):
            break
        try:
            plant.advance(DT, normalized_steering=normalized, throttle=lon.throttle, braking=lon.braking)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
    frame = pd.DataFrame(physics)
    frame["delta_actual_dot_radps"] = np.gradient(frame.delta_actual_rad.to_numpy(float), DT, edge_order=2)
    frame["ay_kin_mps2"] = np.gradient(frame.vy_mps.to_numpy(float), DT, edge_order=2) + frame.vx_mps * frame.r_radps
    folder.mkdir(parents=True)
    frame.to_csv(folder / "physics_log.csv", index=False)
    frame.loc[frame.mpc_updated].to_csv(folder / "mpc_log.csv", index=False)
    if method == "GMR":
        point = pd.DataFrame(predictions)
        times = frame.time_s.to_numpy(float)
        for index, pred in point.iterrows():
            target_t = float(pred.time + pred.tau)
            if target_t <= times[-1] + DT / 2:
                j = int(np.argmin(np.abs(times - target_t)))
                point.at[index, "r_chrono"] = frame.r_radps.iloc[j]
                point.at[index, "ay_chrono"] = frame.ay_com_mps2.iloc[j]
                point.at[index, "Delta_r_target"] = frame.r_radps.iloc[j] - pred.r_bicycle
                point.at[index, "Delta_ay_target"] = frame.ay_com_mps2.iloc[j] - pred.ay_bicycle
        point.to_csv(folder / "pointwise_gmr_outputs.csv", index=False)
    all_inside = bool(frame[[f"{w}_terrain_inside" for w in WHEELS]].all().all())
    all_mu = bool(frame[[f"{w}_terrain_mu_matches" for w in WHEELS]].all().all())
    summary = {
        "case": f"FT03_FULL_MU100_{'NOMINAL' if method == 'Nominal' else 'FROZEN_GMR'}",
        "method": method, "status": "FINISHED" if failure is None else "FAILED", "failure": failure,
        "sample_count": len(frame), "simulated_time_s": float(frame.time_s.iloc[-1]),
        "terminal_projection_reached": bool(frame.projection_station_m.iloc[-1] >= built.formal_end_s),
        "terminal_ey_m": float(frame.ey_m.iloc[-1]), "terminal_epsi_rad": float(frame.epsi_rad.iloc[-1]),
        "solver_failures": int((~frame.loc[frame.mpc_updated, "solver_status"].str.lower().str.startswith("solved")).sum()),
        "min_contact_count": int(frame.wheel_contact_count.min()), "all_core_finite": bool(np.isfinite(frame[list(CORE_CHANNELS)]).all().all()),
        "terrain_all_wheels_inside": all_inside, "terrain_all_mu_matches": all_mu,
        "gmr_control_status": "NOT_LOADED_DURING_CONTROL" if method == "Nominal" else "FROZEN_MEAN_ENABLED_COVARIANCE_DIAGNOSTIC_ONLY",
        "gmr_artifact_sha256": None if method == "Nominal" else bridge.artifact_sha256.upper(),
        "initialization": initialization,
    }
    _json(folder / "run_summary.json", summary)
    return summary


def nominal_reproduction_gate():
    historical = pd.read_csv(HISTORICAL)
    current = pd.read_csv(HERE / "logs/Nominal/physics_log.csv")
    result = {"passed": False, "historical_rows": len(historical), "current_rows": len(current), "absolute_tolerance": 1e-5}
    if len(historical) == len(current):
        differences = {c: float(np.max(np.abs(historical[c].to_numpy(float) - current[c].to_numpy(float)))) for c in CORE_CHANNELS}
        result.update({"max_abs_by_channel": differences, "passed": bool(all(v <= 1e-5 for v in differences.values()))})
    _json(HERE / "audits/nominal_reproduction_gate.json", result)
    return result


def pre_run_sanity():
    full = pd.read_csv(HERE / "logs/Nominal/physics_log.csv")
    frame = full.loc[full.mpc_updated].copy()
    # Frozen windows: formal-entry vicinity and the early 145-165 m yaw-mismatch region.
    selected = frame[(frame.projection_station_m.between(140, 165))].iloc[::5].copy()
    layer = CachedRolloutAlignedLayer()
    rows = []
    for _, r in selected.iterrows():
        controls = np.array([r[f"candidate_delta_{i:02d}_rad"] for i in range(1, 21)], float)
        speeds = np.array([r[f"preview_speed_{i:02d}_mps"] for i in range(1, 21)], float)
        steering_now = float(r.delta_command_rad - r.delta_increment_rad)
        pred = layer.predict(
            {"vy": r.vy_mps, "r": r.r_radps}, {"vx": speeds[0], "steering": steering_now},
            controls, vx_preview=speeds,
        )
        target_time = float(r.time_s + TS)
        target_index = int(np.argmin(np.abs(full.time_s.to_numpy(float) - target_time)))
        target_r = float(full.r_radps.iloc[target_index])
        target_ay = float(full.ay_com_mps2.iloc[target_index])
        rows.append({
            "time_s": r.time_s, "station_s": r.projection_station_m,
            "delta_r_target_one_step": target_r - pred.Y_bicycle[0, 1],
            "delta_ay_target_one_step": target_ay - pred.Y_bicycle[0, 0],
            "mu_delta_r_h1": pred.mu_deltaY[0, 1], "mu_delta_ay_h1": pred.mu_deltaY[0, 0],
            "max_abs_mu_delta_r": float(np.max(np.abs(pred.mu_deltaY[:, 1]))),
            "max_abs_mu_delta_ay": float(np.max(np.abs(pred.mu_deltaY[:, 0]))),
            "all_finite": bool(np.isfinite(np.r_[pred.mu_deltaY.ravel(), pred.Sigma_deltaY.ravel()]).all()),
        })
    data = pd.DataFrame(rows)
    data.to_csv(HERE / "metrics/gmr_pre_run_sanity.csv", index=False)
    passed = bool(len(data) and data.all_finite.all() and data.max_abs_mu_delta_r.max() < 5 and data.max_abs_mu_delta_ay.max() < 100)
    result = {"passed": passed, "sample_count": len(data), "finite": bool(data.all_finite.all()),
              "max_abs_mu_delta_r": float(data.max_abs_mu_delta_r.max()), "max_abs_mu_delta_ay": float(data.max_abs_mu_delta_ay.max()),
              "note": "Magnitude thresholds are interface-explosion guards only; they do not tune or gate based on accuracy."}
    _json(HERE / "audits/gmr_pre_run_sanity.json", result)
    return result


def finalize_hashes():
    before = json.loads((HERE / "manifests/artifact_hashes_before.json").read_text(encoding="utf8"))
    after = frozen_hashes()
    result = {"before": before, "after": after, "unchanged": before == after}
    _json(HERE / "artifact_hashes_before_after.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "sanity", "all"))
    parser.add_argument("--method", choices=("Nominal", "GMR"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), indent=2))
        return
    if args.action == "run":
        print(json.dumps(run_case(args.method), indent=2))
        return
    if args.action == "sanity":
        print(json.dumps(pre_run_sanity(), indent=2))
        return
    prepare()
    nominal = run_case("Nominal")
    gate = nominal_reproduction_gate()
    if not gate["passed"]:
        finalize_hashes()
        raise RuntimeError("NOMINAL_REPRODUCTION_DRIFT_STOP")
    sanity = pre_run_sanity()
    if not sanity["passed"]:
        finalize_hashes()
        raise RuntimeError("BLOCKED_BY_GMR_INTERFACE_MISMATCH")
    gmr = run_case("GMR")
    integrity = finalize_hashes()
    print(json.dumps({"nominal": nominal, "reproduction": gate, "sanity": sanity, "gmr": gmr, "integrity": integrity["unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
