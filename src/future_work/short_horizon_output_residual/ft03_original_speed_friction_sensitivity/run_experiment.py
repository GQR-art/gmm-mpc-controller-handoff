"""Pre-registered FT03 original-speed terrain-friction experiment.

Only the independent terrain JSON differs between groups.  All vehicle,
controller, speed-profile and staged-initialization code is imported unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

from future_work.chrono_mpc_integration.bridge.chrono_config import ChronoSedanConfig
from future_work.chrono_mpc_integration.bridge.chrono_vehicle import ChronoSedanPlant
from future_work.chrono_mpc_integration.bridge.speed_controller import LongitudinalSpeedPI
from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state
from future_work.chrono_mpc_integration.bridge.steering_adapter import SteeringCalibration
from future_work.chrono_mpc_integration.scenarios import ChronoMPCScenario
from future_work.chrono_mpc_integration.baseline_reproduction_audit.historical_protocol_wrapper import (
    CALIBRATION_PATH, Pose2D, _raw_pose, _settle, inverse_settling_pose,
)
from future_work.later_mpc.error_state import global_state_to_error_state
from future_work.short_horizon_output_residual.ft03_reference_nominal_failure_localization.reference_path import build_path
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.profiles import stanford_profile
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import VariableSpeedNominalBridge
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.run_experiment import hashes as prior_hashes
from future_work.short_horizon_output_residual.ft03_high_speed_yaw_dynamics_audit.run_diagnostic_replay import force_snapshot


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_TERRAIN = Path(r"D:\chrono_source\chrono-10.0.0\data\vehicle\terrain\RigidPlane.json")
SOURCE_C0 = HERE.parent / "ft03_high_speed_yaw_dynamics_audit" / "logs" / "diagnostic_c0" / "physics_log.csv"
GROUPS = {"C-mu090": 0.90, "C-mu100": 1.00, "C-mu110": 1.10, "C-mu120": 1.20}
DT = 0.002
TS = 0.05
MAX_TIME_S = 35.0
BASELINE_ABS_TOL = 1e-5  # frozen before outcomes


@dataclass(frozen=True, slots=True)
class FrictionConfig(ChronoSedanConfig):
    terrain_override: Path = Path()

    @property
    def rigid_terrain_json(self) -> Path:
        return self.terrain_override


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def protected_hashes() -> dict[str, str]:
    result = prior_hashes()
    extra = [SOURCE_TERRAIN,
             ROOT / "future_work/short_horizon_output_residual/control_aware_rollout_gmr/outputs/selected_rollout_aligned_B.joblib",
             ROOT / "future_work/short_horizon_output_residual/ft03_variable_speed_nominal_validation/variable_speed_mpc.py",
             ROOT / "future_work/short_horizon_output_residual/ft03_variable_speed_nominal_validation/profiles.py",
             ROOT / "future_work/short_horizon_output_residual/ft03_reference_nominal_failure_localization/reference_path.py",
             ROOT / "future_work/chrono_mpc_integration/vehicle_models/GolfLike/GolfLike_TMeasy_Front.json",
             ROOT / "future_work/chrono_mpc_integration/vehicle_models/GolfLike/GolfLike_TMeasy_Rear.json"]
    result.update({str(p): digest(p) for p in extra})
    return result


def preregister() -> None:
    """Materialize only new-directory configs, before any outcome run."""
    configs = HERE / "configs"
    manifests = HERE / "manifests"
    configs.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    raw = json.loads(SOURCE_TERRAIN.read_text(encoding="utf8"))
    source_mu = raw["Patches"][0]["Contact Material"]["Coefficient of Friction"]
    if source_mu != 0.9:
        raise RuntimeError(f"Frozen terrain source no longer has mu=0.9: {source_mu}")
    for group, mu in GROUPS.items():
        candidate = json.loads(json.dumps(raw))
        candidate["Patches"][0]["Contact Material"]["Coefficient of Friction"] = mu
        path = configs / f"{group}_terrain.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf8")) != candidate:
                raise FileExistsError(f"Pre-existing independent terrain config differs: {path}")
        else:
            path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf8")
        changed = []
        def walk(a, b, prefix=""):
            if isinstance(a, dict):
                for key in a:
                    walk(a[key], b[key], f"{prefix}/{key}")
            elif isinstance(a, list):
                for i, (left, right) in enumerate(zip(a, b)):
                    walk(left, right, f"{prefix}/{i}")
            elif a != b:
                changed.append(prefix)
        walk(raw, candidate)
        expected = [] if mu == source_mu else ["/Patches/0/Contact Material/Coefficient of Friction"]
        if changed != expected:
            raise AssertionError(f"Unexpected terrain differences: {changed}")
    manifest = {
        "preregistered_groups": GROUPS, "only_variable": "terrain contact-material coefficient of friction",
        "source_terrain": str(SOURCE_TERRAIN), "source_terrain_sha256": digest(SOURCE_TERRAIN),
        "config_sha256": {g: digest(configs / f"{g}_terrain.json") for g in GROUPS},
        "Stanford_source_session": "2023-11_11.csv", "formal_source_interval_s": [77.5, 82.45],
        "speed_profile": "same frozen Stanford station-speed profile for all groups",
        "max_simulation_time_s": MAX_TIME_S, "physics_dt_s": DT, "mpc_dt_s": TS, "horizon_steps": 20,
        "terminal_gate": {"projection_at_formal_end": True, "max_abs_ey_m": 0.5, "max_abs_epsi_rad": 0.1},
        "baseline_core_channel_abs_tolerance": BASELINE_ABS_TOL,
        "event_rules_source": "ft03_original_speed_closed_loop_recovery/analyze.py (frozen)",
        "no_GMR": True, "run_order": list(GROUPS),
    }
    path = manifests / "experiment_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf8")) != manifest:
            raise FileExistsError(f"Existing manifest differs: {path}")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    hash_path = manifests / "artifact_hashes_before.json"
    if not hash_path.exists():
        hash_path.write_text(json.dumps(protected_hashes(), indent=2) + "\n", encoding="utf8")


def new_plant(pose: Pose2D, speed: float, terrain_file: Path) -> ChronoSedanPlant:
    config = FrictionConfig(vehicle_model="GolfLike3D_v1", physics_dt_s=DT,
                            initial_x_m=pose.X_m, initial_y_m=pose.Y_m,
                            initial_yaw_rad=pose.psi_rad, initial_forward_speed_mps=speed,
                            terrain_override=terrain_file)
    plant = ChronoSedanPlant(config)
    plant.initialize()
    plant.initialize_wheel_angular_speeds(speed)
    return plant


def initialized_plant(scenario: ChronoMPCScenario, terrain_file: Path):
    calibration = SteeringCalibration.from_json(CALIBRATION_PATH, expected_vehicle_model="GolfLike3D_v1")
    requested = Pose2D(scenario.initial_x_m, scenario.initial_y_m, scenario.initial_psi_rad)
    probe = new_plant(requested, scenario.initial_speed_mps, terrain_file)
    probe_result = _settle(probe, calibration, speed_mps=scenario.initial_speed_mps)
    probe_pose = _raw_pose(probe)
    formal_request = inverse_settling_pose(probe_pose)
    del probe
    plant = new_plant(formal_request, scenario.initial_speed_mps, terrain_file)
    formal_result = _settle(plant, calibration, speed_mps=scenario.initial_speed_mps)
    return plant, calibration, {
        "protocol": "same staged settling/probe/inverse-pose as frozen C0",
        "requested": asdict(requested), "probe_result": asdict(probe_result),
        "probe_post_settling_pose": asdict(probe_pose),
        "formal_requested_pose": asdict(formal_request),
        "formal_result": asdict(formal_result), "formal_post_settling_pose": asdict(_raw_pose(plant)),
    }


def run_one(group: str, *, repeat: bool = False) -> dict:
    if group not in GROUPS:
        raise ValueError(group)
    folder = HERE / "logs" / (group + ("-repeat" if repeat else ""))
    if folder.exists():
        raise FileExistsError(f"Never overwrite a prior run: {folder}")
    terrain_file = HERE / "configs" / f"{group}_terrain.json"
    mu = GROUPS[group]
    actual_config_mu = json.loads(terrain_file.read_text(encoding="utf8"))["Patches"][0]["Contact Material"]["Coefficient of Friction"]
    if actual_config_mu != mu:
        raise RuntimeError("Independent terrain config does not match preregistration")
    built = build_path(77.5, 82.45)
    profile = stanford_profile(built.frame)
    scenario = ChronoMPCScenario(0., 0., 0., float(profile.sample(0)), MAX_TIME_S, built.path)
    plant, calibration, initialization = initialized_plant(scenario, terrain_file)
    if plant.config.rigid_terrain_json.resolve() != terrain_file.resolve() or plant._terrain is None:
        raise RuntimeError("Configured terrain did not reach initialized Chrono plant")
    bridge = VariableSpeedNominalBridge(built.path, profile)
    bridge.reset(0.)
    pi = LongitudinalSpeedPI()
    pi.reset()
    rows = []
    ratio = int(round(TS / DT))
    normalized = 0.
    previous_delta = 0.
    delta_inc = 0.
    mpc = None
    start_time = plant.get_raw_state().time
    failure = None
    for step in range(int(MAX_TIME_S / DT) + 1):
        raw = plant.get_raw_state()
        projected = built.path.project(raw.X, raw.Y)
        if step % ratio == 0:
            previous_delta = bridge.previous_delta_rad
            mpc = bridge.compute_control(raw)
            delta_inc = mpc.delta_rad - previous_delta
            normalized = calibration.delta_rad_to_normalized(mpc.delta_rad)
        else:
            delta_inc = 0.
        speed_ref = float(profile.sample(projected.s_m))
        lon = pi.update(vx_reference_mps=speed_ref, vx_measured_mps=raw.vx_body, dt_s=DT)
        err = global_state_to_error_state(chrono_raw_to_vehicle_state(raw), built.path)
        wheels = plant.get_wheel_diagnostics()
        suspension = plant.get_suspension_diagnostics()
        wmap = {w.wheel_id: w for w in wheels}
        front = [w.road_wheel_angle_rad for w in wheels if w.axle_index == 0]
        row = {
            "time_s": raw.time - start_time, "X_m": raw.X, "Y_m": raw.Y, "psi_rad": raw.psi,
            "vx_mps": raw.vx_body, "vy_mps": raw.vy_body, "r_radps": raw.yaw_rate_body,
            "ey_m": err.ey_m, "epsi_rad": err.epsi_rad, "projection_station_m": projected.s_m,
            "remaining_path_m": built.path.length_m-projected.s_m,
            "reference_X_m": projected.X_m, "reference_Y_m": projected.Y_m,
            "reference_psi_rad": projected.psi_rad, "reference_curvature_per_m": projected.curvature_per_m,
            "speed_reference_mps": speed_ref, "reference_r_radps": speed_ref * projected.curvature_per_m,
            "actual_path_r_demand_radps": raw.vx_body * projected.curvature_per_m,
            "ay_reference_demand_mps2": speed_ref**2 * projected.curvature_per_m,
            "delta_command_rad": mpc.delta_rad, "delta_actual_rad": float(math.atan(np.mean(np.tan(front)))),
            "delta_increment_rad": delta_inc, "steering_normalized": normalized,
            "throttle": lon.throttle, "brake": lon.braking,
            "solver_status": mpc.solution.status, "solver_time_s": mpc.solution.solve_time_s,
            "mpc_updated": step % ratio == 0, "roll_rad": plant.get_chassis_roll_pitch_rad()[0],
            "beta_rad": math.atan2(raw.vy_body, raw.vx_body),
            "ay_com_mps2": plant.get_aggregate_com_acceleration_body_mps2()[1],
            "wheel_contact_count": sum(x.contact for x in suspension),
            "formal_interval": built.formal_start_s <= projected.s_m <= built.formal_end_s,
            "requested_mu": mu, "actual_runtime_mu": actual_config_mu,
        }
        for wid in ("FL", "FR", "RL", "RR"):
            w = wmap[wid]
            row.update({f"{wid}_Fz_N": w.force_z_tire_N,
                        f"{wid}_slip_angle_rad": w.slip_angle_rad,
                        f"{wid}_lateral_slip_sy": w.lateral_slip_sy,
                        f"{wid}_longitudinal_slip_sx": w.longitudinal_slip_sx,
                        f"{wid}_contact": bool(next(x.contact for x in suspension if x.wheel_id == wid))})
        row.update(force_snapshot(plant))
        if step % ratio == 0:
            controls = np.asarray(mpc.solution.delta_sequence, float)
            speeds = np.asarray(mpc.solution.diagnostics["speed_preview_mps"], float)
            for j in range(20):
                row[f"candidate_delta_{j+1:02d}_rad"] = controls[j]
                row[f"preview_speed_{j+1:02d}_mps"] = speeds[j]
        rows.append(row)
        if projected.s_m >= built.formal_end_s:
            break
        if step == int(MAX_TIME_S / DT):
            break
        try:
            plant.advance(DT, normalized_steering=normalized, throttle=lon.throttle, braking=lon.braking)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
    frame = pd.DataFrame(rows)
    frame["delta_actual_dot_radps"] = np.gradient(frame.delta_actual_rad.to_numpy(float), DT, edge_order=2)
    frame["ay_kin_mps2"] = np.gradient(frame.vy_mps.to_numpy(float), DT, edge_order=2) + frame.vx_mps * frame.r_radps
    folder.mkdir(parents=True)
    frame.to_csv(folder / "physics_log.csv", index=False)
    frame.loc[frame.mpc_updated].to_csv(folder / "mpc_log.csv", index=False)
    result = {
        "group": group, "requested_mu": mu, "actual_runtime_mu": actual_config_mu,
        "runtime_mu_evidence": "loaded independent terrain JSON path verified on initialized plant; direct material readback unavailable",
        "terrain_json_sha256": digest(terrain_file), "status": "FINISHED" if failure is None else "FAILED",
        "failure": failure, "sample_count": len(frame), "simulated_time_s": float(frame.time_s.iloc[-1]),
        "projection_reached": bool(frame.projection_station_m.iloc[-1] >= built.formal_end_s),
        "initialization": initialization, "solver_failures": int((~frame.loc[frame.mpc_updated, "solver_status"].str.lower().str.startswith("solved")).sum()),
        "min_contact_count": int(frame.wheel_contact_count.min()),
        "all_core_finite": bool(np.isfinite(frame[["X_m","Y_m","psi_rad","vx_mps","vy_mps","r_radps","ey_m","epsi_rad"]]).all().all()),
        "no_GMR": True,
    }
    (folder / "run_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return result


def baseline_gate() -> dict:
    old = pd.read_csv(SOURCE_C0)
    new = pd.read_csv(HERE / "logs" / "C-mu090" / "physics_log.csv")
    channels = ["time_s","X_m","Y_m","psi_rad","vx_mps","vy_mps","r_radps","ey_m","epsi_rad",
                "projection_station_m","delta_command_rad","delta_actual_rad","throttle","brake"]
    if len(old) != len(new):
        return {"passed": False, "reason": "row count mismatch", "old_rows": len(old), "new_rows": len(new)}
    differences = {c: float(np.max(np.abs(old[c].to_numpy(float) - new[c].to_numpy(float)))) for c in channels}
    return {"passed": all(v <= BASELINE_ABS_TOL for v in differences.values()),
            "max_abs_by_channel": differences, "absolute_tolerance": BASELINE_ABS_TOL,
            "old_rows": len(old), "new_rows": len(new)}


def main() -> None:
    preregister()
    # Re-check before hashes before *any* run, including resumed invocations.
    before = json.loads((HERE / "manifests" / "artifact_hashes_before.json").read_text(encoding="utf8"))
    if before != protected_hashes():
        raise RuntimeError("Protected files differ from preregistered before hashes")
    for group in GROUPS:
        folder = HERE / "logs" / group
        if not folder.exists():
            print(f"START {group}", flush=True)
            result = run_one(group)
            print(json.dumps({k: v for k, v in result.items() if k != "initialization"}), flush=True)
        else:
            print(f"EXISTING {group}; never overwrite", flush=True)
        if group == "C-mu090":
            gate = baseline_gate()
            path = HERE / "manifests" / "baseline_reproduction_gate.json"
            path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf8")
            print(f"C0 baseline reproduction: {gate['passed']}", flush=True)
            if not gate["passed"]:
                raise RuntimeError("C-mu090 does not reproduce frozen C0; stop friction comparison")
    after = protected_hashes()
    record = {"before": before, "after": after, "unchanged": before == after}
    (HERE / "artifact_hashes_before_after.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    if not record["unchanged"]:
        raise RuntimeError("Frozen asset hash changed")


if __name__ == "__main__":
    main()
