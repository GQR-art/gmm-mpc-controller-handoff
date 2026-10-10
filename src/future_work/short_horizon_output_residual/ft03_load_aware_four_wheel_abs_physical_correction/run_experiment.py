"""Run isolated FT03 brake-physics correction with frozen Nominal steering MPC."""
from __future__ import annotations

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
from future_work.short_horizon_output_residual.ft03_high_speed_yaw_dynamics_audit.run_diagnostic_replay import force_snapshot
from future_work.short_horizon_output_residual.ft03_original_speed_friction_sensitivity.run_experiment import initialized_plant
from future_work.short_horizon_output_residual.ft03_reference_nominal_failure_localization.reference_path import build_path
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.profiles import stanford_profile
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import VariableSpeedNominalBridge
from future_work.short_horizon_output_residual.ft03_fullterrain_frozen_gmr_exploratory_closed_loop.run_experiment import DT, TS, MAX_TIME_S, TERRAIN, WHEELS, _terrain_fields

from .brake_supervisor import FourWheelBrakeSupervisor, WheelBrakeCommand, WheelBrakeState


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GROUPS = ("Original", "LoadAware", "LoadAwareABS")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf8")


def frozen_assets() -> dict[str, Path]:
    return {
        "terrain": TERRAIN,
        "front_tire": ROOT / "future_work/chrono_mpc_integration/vehicle_models/GolfLike/GolfLike_TMeasy_Front.json",
        "rear_tire": ROOT / "future_work/chrono_mpc_integration/vehicle_models/GolfLike/GolfLike_TMeasy_Rear.json",
        "vehicle": ROOT / "future_work/chrono_mpc_integration/vehicle_models/GolfLike/GolfLike_Vehicle.json",
        "speed_pi": ROOT / "future_work/chrono_mpc_integration/bridge/speed_controller.py",
        "nominal_mpc": ROOT / "future_work/short_horizon_output_residual/ft03_variable_speed_nominal_validation/variable_speed_mpc.py",
        "supervisor": HERE / "brake_supervisor.py",
    }


def hashes() -> dict[str, str]:
    return {str(path): digest(path) for path in frozen_assets().values()}


def prepare() -> dict:
    for folder in ("audits", "manifests", "logs", "metrics", "figures", "reports", "tests"):
        (HERE / folder).mkdir(parents=True, exist_ok=True)
    before = hashes()
    path = HERE / "manifests/artifact_hashes_before.json"
    if path.exists() and json.loads(path.read_text(encoding="utf8")) != before:
        raise RuntimeError("Input drift after preregistration")
    save_json(path, before)
    manifest = {
        "status": "PREREGISTERED_BEFORE_PHYSICAL_CORRECTION_OUTCOMES",
        "scope": "general symmetric four-wheel brake correction; FT03 is discovery and validation scene",
        "groups": {
            "Original": "unchanged BrakeSimple; same normalized command at four wheels",
            "LoadAware": "wheel-load and combined-force-aware torque allocation; no ABS feedback",
            "LoadAwareABS": "same allocation plus symmetric independent four-wheel ABS",
        },
        "only_design_difference": "longitudinal brake actuation boundary",
        "shared": {
            "controller": "frozen variable-speed Nominal steering MPC; no GMR",
            "path": "FT03 Stanford original GNSS",
            "speed_reference": "Stanford original station-speed profile",
            "terrain_mu": 1.0,
            "physics_dt_s": DT,
            "control_dt_s": TS,
            "vehicle_tires_initialization_stopping_rule": "unchanged",
        },
        "brake_parameters": {
            "maximum_torque_per_wheel_Nm": 2000.0,
            "capacity_margin": 0.85,
            "allocation": "requested total 4*2000*b, proportional to instantaneous Fz, iteratively redistributed subject to per-wheel combined-force capacity",
            "abs": {"release_slip": 0.20, "reapply_slip": 0.10, "release_rate_per_s": 8.0, "reapply_rate_per_s": 2.0, "minimum_speed_mps": 3.0},
        },
        "wheel_lock_definition": "abs(omega)*R/abs(vx_body)<0.10 while abs(vx_body)>10 m/s",
        "sustained_duration_s": 0.05,
        "success_requirements": [
            "no sustained wheel lock", "extreme internal sy samples eliminated or nearly eliminated",
            "full path completion and four contacts", "speed tracking and lateral stability not materially degraded",
        ],
        "forbidden": ["GMR changes", "MPC tuning", "tire or terrain friction changes", "left-rear special case", "state or wheel-speed overwrite"],
        "source_hashes": before,
    }
    mpath = HERE / "manifests/experiment_manifest.json"
    if mpath.exists() and json.loads(mpath.read_text(encoding="utf8")) != manifest:
        raise RuntimeError("Manifest drift")
    save_json(mpath, manifest)
    return manifest


def direct_kinematic_slips(plant) -> dict[str, float]:
    values = {}
    ids = (("FL", "FR"), ("RL", "RR"))
    for ai, axle in enumerate(plant._vehicle.GetAxles()):
        for wi, wheel in enumerate(axle.GetWheels()):
            values[ids[ai][wi]] = float(wheel.GetTire().GetLongitudinalSlip())
    return values


def supervisor_states(plant, diagnostics, contacts) -> tuple[WheelBrakeState, ...]:
    slip = direct_kinematic_slips(plant)
    by_wheel = {item.wheel_id: item for item in diagnostics}
    return tuple(WheelBrakeState(
        wheel_id=wheel,
        normal_load_N=abs(by_wheel[wheel].force_z_tire_N),
        lateral_force_N=by_wheel[wheel].force_y_tire_N,
        rolling_radius_m=by_wheel[wheel].effective_radius_m,
        kinematic_longitudinal_slip=slip[wheel],
        contact=bool(contacts[wheel]),
    ) for wheel in WHEELS)


def original_commands(request: float, states: tuple[WheelBrakeState, ...]) -> tuple[WheelBrakeCommand, ...]:
    return tuple(WheelBrakeCommand(
        wheel_id=state.wheel_id,
        normalized=float(request),
        torque_Nm=float(request) * 2000.0,
        capacity_Nm=float("nan"),
        base_torque_Nm=float(request) * 2000.0,
        abs_multiplier=1.0,
        braking_slip=max(0.0, -state.kinematic_longitudinal_slip),
    ) for state in states)


def advance_individual_brakes(plant, *, normalized_steering: float, throttle: float, requested_braking: float, commands: tuple[WheelBrakeCommand, ...]) -> None:
    vehicle = plant._vehicle
    terrain = plant._terrain
    time = vehicle.GetSystem().GetChTime()
    driver = plant.make_driver_inputs(normalized_steering=normalized_steering, throttle=throttle, braking=requested_braking)
    vehicle.Synchronize(time, driver, terrain)
    terrain.Synchronize(time)
    sides = (plant._vehicle_module.LEFT, plant._vehicle_module.RIGHT)
    by_wheel = {command.wheel_id: command for command in commands}
    ids = (("FL", "FR"), ("RL", "RR"))
    for ai in range(2):
        for wi, side in enumerate(sides):
            vehicle.GetBrake(ai, side).Synchronize(time, by_wheel[ids[ai][wi]].normalized)
    vehicle.Advance(DT)
    terrain.Advance(DT)


def run_group(group: str) -> dict:
    folder = HERE / "logs" / group
    if folder.exists():
        raise FileExistsError(f"Never overwrite prior group: {folder}")
    if json.loads((HERE / "manifests/artifact_hashes_before.json").read_text(encoding="utf8")) != hashes():
        raise RuntimeError("Frozen assets changed before run")
    built = build_path(77.5, 82.45)
    profile = stanford_profile(built.frame)
    scenario = ChronoMPCScenario(0.0, 0.0, 0.0, float(profile.sample(0)), MAX_TIME_S, built.path)
    plant, calibration, initialization = initialized_plant(scenario, TERRAIN)
    bridge = VariableSpeedNominalBridge(built.path, profile); bridge.reset(0.0)
    speed_pi = LongitudinalSpeedPI(); speed_pi.reset()
    supervisor = None if group == "Original" else FourWheelBrakeSupervisor(mode="load_aware" if group == "LoadAware" else "load_aware_abs", dt_s=DT)
    control_ratio = int(round(TS / DT)); normalized = 0.0; delta_increment = 0.0; output = None
    rows = []; failure = None; start = plant.get_raw_state().time
    for step in range(int(MAX_TIME_S / DT) + 1):
        raw = plant.get_raw_state(); projected = built.path.project(raw.X, raw.Y)
        if step % control_ratio == 0:
            previous = bridge.previous_delta_rad
            try:
                output = bridge.compute_control(raw)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"; break
            delta_increment = output.delta_rad - previous
            normalized = calibration.delta_rad_to_normalized(output.delta_rad)
        else:
            delta_increment = 0.0
        speed_reference = float(profile.sample(projected.s_m))
        longitudinal = speed_pi.update(vx_reference_mps=speed_reference, vx_measured_mps=raw.vx_body, dt_s=DT)
        error = global_state_to_error_state(chrono_raw_to_vehicle_state(raw), built.path)
        diagnostics = plant.get_wheel_diagnostics(); suspension = plant.get_suspension_diagnostics()
        contacts = {item.wheel_id: item.contact for item in suspension}
        states = supervisor_states(plant, diagnostics, contacts)
        commands = original_commands(longitudinal.braking, states) if supervisor is None else supervisor.update(longitudinal.braking, raw.vx_body, states)
        wmap = {item.wheel_id: item for item in diagnostics}; cmap = {item.wheel_id: item for item in commands}
        front = [item.road_wheel_angle_rad for item in diagnostics if item.axle_index == 0]
        row = {
            "time_s": raw.time-start, "X_m": raw.X, "Y_m": raw.Y, "psi_rad": raw.psi,
            "vx_mps": raw.vx_body, "vy_mps": raw.vy_body, "r_radps": raw.yaw_rate_body,
            "ey_m": error.ey_m, "epsi_rad": error.epsi_rad, "beta_rad": math.atan2(raw.vy_body, raw.vx_body),
            "projection_station_m": projected.s_m, "remaining_path_m": built.path.length_m-projected.s_m,
            "reference_X_m": projected.X_m, "reference_Y_m": projected.Y_m, "reference_psi_rad": projected.psi_rad,
            "reference_curvature_per_m": projected.curvature_per_m, "speed_reference_mps": speed_reference,
            "reference_r_radps": speed_reference*projected.curvature_per_m,
            "delta_command_rad": output.delta_rad, "delta_actual_rad": float(math.atan(np.mean(np.tan(front)))),
            "delta_increment_rad": delta_increment, "steering_normalized": normalized,
            "throttle": longitudinal.throttle, "requested_brake": longitudinal.braking,
            "applied_total_brake_torque_Nm": sum(item.torque_Nm for item in commands),
            "requested_total_brake_torque_Nm": longitudinal.braking*8000.0,
            "solver_status": output.solution.status, "solver_time_s": output.solution.solve_time_s,
            "mpc_updated": step % control_ratio == 0, "wheel_contact_count": sum(contacts.values()),
            "formal_interval": built.formal_start_s <= projected.s_m <= built.formal_end_s, "group": group,
        }
        for wheel in WHEELS:
            state = next(item for item in states if item.wheel_id == wheel); diag = wmap[wheel]; command = cmap[wheel]
            rolling_ratio = abs(diag.wheel_omega_radps)*diag.effective_radius_m/max(abs(raw.vx_body), 0.1)
            row.update({
                f"{wheel}_Fz_N": diag.force_z_tire_N, f"{wheel}_Fy_N": diag.force_y_tire_N,
                f"{wheel}_slip_angle_rad": diag.slip_angle_rad, f"{wheel}_internal_sy": diag.lateral_slip_sy,
                f"{wheel}_internal_sx": diag.longitudinal_slip_sx,
                f"{wheel}_kinematic_longitudinal_slip": state.kinematic_longitudinal_slip,
                f"{wheel}_wheel_omega_radps": diag.wheel_omega_radps, f"{wheel}_rolling_speed_ratio": rolling_ratio,
                f"{wheel}_contact": contacts[wheel], f"{wheel}_brake_normalized": command.normalized,
                f"{wheel}_brake_torque_Nm": command.torque_Nm, f"{wheel}_brake_capacity_Nm": command.capacity_Nm,
                f"{wheel}_base_brake_torque_Nm": command.base_torque_Nm, f"{wheel}_abs_multiplier": command.abs_multiplier,
                f"{wheel}_braking_slip": command.braking_slip,
            })
        row.update(_terrain_fields(plant, force_snapshot(plant))); rows.append(row)
        if projected.s_m >= built.formal_end_s or step == int(MAX_TIME_S / DT):
            break
        try:
            if group == "Original":
                plant.advance(DT, normalized_steering=normalized, throttle=longitudinal.throttle, braking=longitudinal.braking)
            else:
                advance_individual_brakes(plant, normalized_steering=normalized, throttle=longitudinal.throttle, requested_braking=longitudinal.braking, commands=commands)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"; break
    frame = pd.DataFrame(rows)
    frame["delta_actual_dot_radps"] = np.gradient(frame.delta_actual_rad.to_numpy(float), DT, edge_order=2)
    folder.mkdir(parents=True); frame.to_csv(folder / "physics_log.csv", index=False)
    frame.loc[frame.mpc_updated].to_csv(folder / "mpc_log.csv", index=False)
    summary = {
        "group": group, "status": "FINISHED" if failure is None else "FAILED", "failure": failure,
        "sample_count": len(frame), "simulated_time_s": float(frame.time_s.iloc[-1]),
        "terminal_projection_reached": bool(frame.projection_station_m.iloc[-1] >= built.formal_end_s),
        "terminal_ey_m": float(frame.ey_m.iloc[-1]), "terminal_epsi_rad": float(frame.epsi_rad.iloc[-1]),
        "solver_failures": int((~frame.loc[frame.mpc_updated, "solver_status"].str.lower().str.startswith("solved")).sum()),
        "min_contact_count": int(frame.wheel_contact_count.min()), "initialization": initialization,
    }
    save_json(folder / "run_summary.json", summary)
    return summary


def finalize() -> dict:
    before = json.loads((HERE / "manifests/artifact_hashes_before.json").read_text(encoding="utf8"))
    after = hashes(); result = {"before": before, "after": after, "unchanged": before == after}
    save_json(HERE / "artifact_hashes_before_after.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("prepare", "run", "all")); parser.add_argument("--group", choices=GROUPS)
    args = parser.parse_args()
    if args.action == "prepare": print(json.dumps(prepare(), indent=2)); return
    if args.action == "run": print(json.dumps(run_group(args.group), indent=2)); return
    prepare(); results = {}
    for group in GROUPS:
        results[group] = run_group(group)
        if results[group]["status"] != "FINISHED": break
    integrity = finalize(); print(json.dumps({"results": results, "source_hashes_unchanged": integrity["unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
