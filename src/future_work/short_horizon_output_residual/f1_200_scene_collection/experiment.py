"""Run the frozen FoundationBridge on the preregistered 200-scene catalog.

This stage only collects full closed-loop data.  It does not launch frozen-plan
branches, F1D, GMM2 training, model selection, or outcome-driven scene changes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from future_work.chrono_mpc_integration.bridge.speed_controller import LongitudinalSpeedPI
from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state
from future_work.chrono_mpc_integration.scenarios import ChronoMPCScenario
from future_work.later_mpc.error_state import global_state_to_error_state
from future_work.short_horizon_output_residual.brake_supervisor_v2_validation.brake_supervisor_v2 import FourWheelBrakeSupervisorV2
from future_work.short_horizon_output_residual.ft03_load_aware_four_wheel_abs_physical_correction.run_experiment import advance_individual_brakes, supervisor_states
from future_work.short_horizon_output_residual.ft03_original_speed_friction_sensitivity.run_experiment import initialized_plant
from future_work.short_horizon_output_residual.gmm1_foundation_stage1.model import FoundationBridge
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import MODEL

from .catalog import SEED, build_catalog
from .scenarios import build_scene

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_TERRAIN = HERE.parent / "ft03_terrain_coverage_rebuild/configs/C-MU100-FULL_terrain.json"
FOUNDATION = HERE.parent / "gmm1_foundation_stage1/model.py"
BRAKES = HERE.parent / "brake_supervisor_v2_validation/brake_supervisor_v2.py"
DT, TS, HORIZON = 0.002, 0.05, 20
MAX_TIME_S = 40.0
TERRAIN_MARGIN_M = 120.0
WHEELS = ("FL", "FR", "RL", "RR")
GROUP = "F1"
REQUIRED = ("physics_log.csv", "control_cycles.csv", "horizon_io.csv", "run_status.json")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+"\n",
                   encoding="utf-8")
    tmp.replace(path)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def create_terrain(spec: dict, path) -> dict:
    samples = [path.sample(s) for s in np.linspace(0.0, path.length_m, 3001)]
    x = np.asarray([p.X_m for p in samples]); y = np.asarray([p.Y_m for p in samples])
    cx, cy = float((x.min()+x.max())/2), float((y.min()+y.max())/2)
    sx = float(max(600.0, math.ceil(float(np.ptp(x)+2*TERRAIN_MARGIN_M))))
    sy = float(max(500.0, math.ceil(float(np.ptp(y)+2*TERRAIN_MARGIN_M))))
    config = json.loads(BASE_TERRAIN.read_text(encoding="utf-8"))
    patch = config["Patches"][0]
    patch["Location"] = [cx, cy, 0]
    patch["Geometry"]["Dimensions"] = [sx, sy, 2]
    patch["Contact Material"]["Coefficient of Friction"] = 1.0
    target = HERE / "configs" / f"{spec['scenario_id']}_terrain.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != config:
        if list((HERE/"logs").glob("*/F1/run_status.json")):
            raise RuntimeError(f"Terrain drift after outcomes exist: {target}")
    save_json(target, config)
    return {"path": str(target), "sha256": digest(target), "center_m": [cx, cy],
            "dimensions_m": [sx, sy, 2.0],
            "reference_bbox_m": [float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            "margin_m": TERRAIN_MARGIN_M}


def frozen_hashes() -> dict:
    assets = (BASE_TERRAIN, FOUNDATION, BRAKES, Path(MODEL), Path(__file__), HERE/"catalog.py", HERE/"scenarios.py")
    return {str(p): digest(p) for p in assets}


def io_schema() -> dict:
    return {
        "physics_log.csv": "0.002 s pre-advance plant state and command subsequently applied",
        "control_cycles.csv": "0.05 s current state, F1 controller result, memory and timing",
        "horizon_io.csv": "20 rows per control cycle: U0, U1, GMM1 output, F0(U1), frozen correction and F1 prediction",
        "run_status.json": "completion or retained failure record",
        "time_alignment": "control row k predicts endpoints k+25*i physics steps; one control step is 0.05 s",
        "F1_semantics": "F0(U1)+C(U0); vy/r remain nominal and are explicitly marked as such",
        "multistep_note": "closed-loop future replans are not labeled same-plan residuals; raw components are retained for later replay",
    }


def prepare() -> dict:
    for name in ("configs", "logs", "manifests", "organized"):
        (HERE/name).mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    paths, profiles, precheck = {}, {}, {}
    for spec in catalog:
        path, profile, start, end, meta = build_scene(spec)
        problems = []
        if meta["minimum_reference_speed_mps"] <= 0: problems.append("nonpositive reference speed")
        if meta["peak_abs_curvature_per_m"] > 0.035: problems.append("curvature outside prior operating envelope")
        terrain = create_terrain(spec, path)
        paths[spec["scenario_id"]] = {**meta, "formal_start_station_m": start,
                                      "formal_end_station_m": end}
        profiles[spec["scenario_id"]] = terrain
        precheck[spec["scenario_id"]] = {"status": "REJECTED" if problems else "PASS",
                                          "problems": problems}
    catalog_doc = {"status": "FROZEN_BEFORE_200_SCENE_OUTCOMES", "seed": SEED,
                   "scenario_count": 200, "scenarios": catalog}
    catalog_path = HERE/"manifests/scenario_catalog.json"
    old = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else None
    if old is not None and old != catalog_doc and list((HERE/"logs").glob("*/F1/run_status.json")):
        raise RuntimeError("Catalog drift after outcomes exist")
    save_json(catalog_path, catalog_doc)
    pd.DataFrame(catalog).to_csv(HERE/"manifests/scenario_catalog.csv", index=False)
    conflicts = [x for x in catalog if x["prior_identity"] != "none"]
    pd.DataFrame(conflicts).to_csv(HERE/"manifests/prior_identity_conflicts.csv", index=False)
    save_json(HERE/"manifests/io_schema.json", io_schema())
    manifest = {
        "status": "FROZEN_BEFORE_200_SCENE_OUTCOMES",
        "scope": "200 complete F1 closed loops only; no fixed-plan branches, F1D, GMM2 training or model selection",
        "controller": "unchanged FoundationBridge single-query corrected-QP F1",
        "scenario_count": 200, "run_count": 200, "seed": SEED,
        "split_counts": {"train": 120, "validation": 40, "test": 40},
        "sampling": {"physics_dt_s": DT, "control_dt_s": TS, "horizon_steps": HORIZON},
        "resume_rule": "FINISHED and FAILED runs are retained and never overwritten; incomplete directories require audit",
        "test_rule": "test is collected as authorized but is not used for model selection or scene changes",
        "paths": paths, "terrains": profiles, "precheck": precheck,
        "frozen_hashes": frozen_hashes(),
    }
    manifest_path = HERE/"manifests/experiment_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        if list((HERE/"logs").glob("*/F1/run_status.json")):
            raise RuntimeError("Manifest drift after outcomes exist")
    save_json(manifest_path, manifest)
    return manifest


def _complete(folder: Path) -> bool:
    if not all((folder/x).exists() for x in REQUIRED): return False
    return json.loads((folder/"run_status.json").read_text(encoding="utf-8")).get("status") in {"FINISHED", "FAILED"}


def _horizon_rows(spec, bridge, output, raw, control_index, step, time_s, previous):
    states = np.asarray(output.solution.predicted_error_states, float)
    if states.shape == (21, 4): states = states[1:]
    if states.shape != (20, 4): raise RuntimeError(f"Unexpected F1 state shape {states.shape}")
    u1 = np.asarray(output.solution.delta_sequence, float)
    u0 = np.asarray(bridge.last_diagnostics.nominal_controls, float)
    adapter = bridge.full_adapter
    prediction = bridge.last_diagnostics.iteration.prediction
    preview = output.reference_preview
    speeds = np.asarray(output.solution.diagnostics["speed_preview_mps"], float)
    rows=[]
    for i in range(HORIZON):
        pidx=min(i+1,len(preview.X_m)-1)
        row = {
            "scenario_id": spec["scenario_id"], "split": spec["split"],
            "control_index": control_index, "physics_step": step, "control_time_s": time_s,
            "prediction_step": i+1, "prediction_time_s": (i+1)*TS,
            "target_physics_step": step+(i+1)*25,
            "current_vx_mps": raw.vx_body, "current_vy_mps": raw.vy_body,
            "current_r_radps": raw.yaw_rate_body, "current_ey_m": output.error_state.ey_m,
            "current_epsi_rad": output.error_state.epsi_rad, "previous_request_rad": previous,
            "U0_query_request_rad": u0[i], "U1_selected_request_rad": u1[i],
            "preview_station_m": preview.arc_length_m[pidx], "preview_X_m": preview.X_m[pidx],
            "preview_Y_m": preview.Y_m[pidx], "preview_psi_rad": preview.psi_rad[pidx],
            "preview_curvature_per_m": preview.curvature_per_m[i], "preview_speed_mps": speeds[i],
            "gmm1_delta_ay_mps2": prediction.mu_deltaY[i,0],
            "gmm1_delta_r_endpoint_radps": prediction.mu_deltaY[i,1],
            "gmm1_bicycle_ay_mps2": prediction.Y_bicycle[i,0],
            "gmm1_bicycle_r_radps": prediction.Y_bicycle[i,1],
            "F1_prediction_semantics": "F0(U1)+C(U0)",
            "F1_vy_r_channel": "nominal_unmodified",
        }
        for j,name in enumerate(("ey_m","epsi_rad","vy_mps","r_radps")):
            row[f"F0_U1_predicted_{name}"] = adapter.nominal1[i+1,j]
            row[f"frozen_GMM1_correction_{name}"] = adapter.correction[i,j]
            row[f"F1_predicted_{name}"] = states[i,j]
        rows.append(row)
    return rows


def run_one(scenario_id: str, manifest: dict | None = None) -> dict:
    manifest = prepare() if manifest is None else manifest
    spec = next(x for x in build_catalog() if x["scenario_id"] == scenario_id)
    folder = HERE/"logs"/scenario_id/GROUP
    if _complete(folder): return json.loads((folder/"run_status.json").read_text(encoding="utf-8"))
    if folder.exists(): raise RuntimeError(f"Incomplete run directory requires audit: {folder}")
    if manifest["precheck"][scenario_id]["status"] != "PASS":
        raise RuntimeError(f"Scenario failed precheck: {manifest['precheck'][scenario_id]}")
    if frozen_hashes() != manifest["frozen_hashes"]: raise RuntimeError("Frozen asset drift")
    path, profile, formal_start, formal_end, reference_meta = build_scene(spec)
    terrain = Path(manifest["terrains"][scenario_id]["path"])
    scenario = ChronoMPCScenario(0.,0.,0.,float(spec["speed_mps"]),MAX_TIME_S,path)
    physics_rows=[]; control_rows=[]; horizon_rows=[]; failure=None
    folder.mkdir(parents=True)
    try:
        plant, calibration, initialization = initialized_plant(scenario, terrain)
        bridge=FoundationBridge(path,profile); bridge.reset(0.)
        pi=LongitudinalSpeedPI(); pi.reset(); supervisor=FourWheelBrakeSupervisorV2(dt_s=DT)
        t0=float(plant.get_raw_state().time); request=0.; wall=np.nan; control_index=-1; saturated=0
        for step in range(int(MAX_TIME_S/DT)+1):
            raw=plant.get_raw_state(); time_s=float(raw.time-t0); projected=path.project(raw.X,raw.Y)
            error=global_state_to_error_state(chrono_raw_to_vehicle_state(raw),path)
            diagnostics=plant.get_wheel_diagnostics(); suspension=plant.get_suspension_diagnostics()
            contacts={x.wheel_id:x.contact for x in suspension}; dmap={x.wheel_id:x for x in diagnostics}
            front=[x.road_wheel_angle_rad for x in diagnostics if x.axle_index==0]
            delta_actual=float(math.atan(np.mean(np.tan(front))))
            updated=step%25==0
            if updated:
                control_index+=1; previous=float(bridge.previous_delta_rad); began=perf_counter()
                output=bridge.compute_control(raw); wall=perf_counter()-began
                if not output.solution.success: raise RuntimeError(output.solution.status)
                request=float(output.delta_rad)
                horizon_rows.extend(_horizon_rows(spec,bridge,output,raw,control_index,step,time_s,previous))
                cycle={
                    "scenario_id":scenario_id,"source_group":spec["source_group"],"split":spec["split"],
                    "control_index":control_index,"physics_step":step,"time_s":time_s,
                    "vx_mps":raw.vx_body,"vy_mps":raw.vy_body,"r_radps":raw.yaw_rate_body,
                    "ey_m":error.ey_m,"epsi_rad":error.epsi_rad,"previous_request_rad":previous,
                    "U0_first_rad":float(bridge.last_diagnostics.nominal_controls[0]),
                    "U1_first_rad":float(output.solution.delta_sequence[0]),"actual_front_delta_rad":delta_actual,
                    "pi_integral_before":pi.integral_error_mps_s,
                    "solver_status":output.solution.status,"solver_time_s":output.solution.solve_time_s,
                    "controller_wall_s":wall,"prediction_semantics":"F0(U1)+C(U0)",
                    "F1_vy_r_channel":"nominal_unmodified",
                    "formal_interval":formal_start<=projected.s_m<=formal_end,
                }
                for w in WHEELS: cycle[f"{w}_abs_before"]=supervisor._multipliers[w]
                control_rows.append(cycle)
            speed_ref=float(profile.sample(projected.s_m)); pi_before=pi.integral_error_mps_s
            lon=pi.update(vx_reference_mps=speed_ref,vx_measured_mps=raw.vx_body,dt_s=DT)
            commands=supervisor.update(lon.braking,raw.vx_body,supervisor_states(plant,diagnostics,contacts))
            cmap={x.wheel_id:x for x in commands}; ax,ay,*_=plant.get_aggregate_com_acceleration_body_mps2()
            row={
                "scenario_id":scenario_id,"source_group":spec["source_group"],"split":spec["split"],
                "scene_type":spec["scene_type"],"physics_step":step,"time_s":time_s,
                "control_index":control_index,"mpc_updated":updated,
                "formal_interval":formal_start<=projected.s_m<=formal_end,
                "X_m":raw.X,"Y_m":raw.Y,"psi_rad":raw.psi,"vx_mps":raw.vx_body,
                "vy_mps":raw.vy_body,"r_radps":raw.yaw_rate_body,"ax_com_mps2":ax,"ay_com_mps2":ay,
                "beta_rad":math.atan2(raw.vy_body,max(abs(raw.vx_body),.1)),
                "ey_m":error.ey_m,"epsi_rad":error.epsi_rad,"projection_station_m":projected.s_m,
                "reference_X_m":projected.X_m,"reference_Y_m":projected.Y_m,
                "reference_psi_rad":projected.psi_rad,"reference_curvature_per_m":projected.curvature_per_m,
                "speed_reference_mps":speed_ref,"delta_request_rad":request,"delta_actual_rad":delta_actual,
                "steering_saturated":abs(request)>=.349,"throttle":lon.throttle,
                "total_brake_request":lon.braking,"pi_integral_before":pi_before,
                "pi_integral_after":pi.integral_error_mps_s,"solver_status":output.solution.status,
                "controller_wall_s":wall if updated else np.nan,"wheel_contact_count":sum(contacts.values()),
            }
            for w in WHEELS:
                d,c=dmap[w],cmap[w]
                row.update({f"{w}_contact":contacts[w],f"{w}_wheel_speed_radps":d.wheel_omega_radps,
                    f"{w}_longitudinal_slip":d.longitudinal_slip_sx,f"{w}_lateral_slip":d.lateral_slip_sy,
                    f"{w}_slip_angle_rad":d.slip_angle_rad,f"{w}_braking_slip":c.braking_slip,
                    f"{w}_abs_multiplier":c.abs_multiplier,f"{w}_brake_torque_Nm":c.torque_Nm,
                    f"{w}_road_wheel_angle_rad":d.road_wheel_angle_rad})
            physics_rows.append(row)
            saturated=saturated+1 if row["steering_saturated"] else 0
            if abs(error.ey_m)>5 or abs(error.epsi_rad)>1 or saturated*DT>=1.0:
                raise RuntimeError("severe deviation or sustained saturation stop")
            if projected.s_m>=formal_end: break
            advance_individual_brakes(plant,normalized_steering=float(calibration.delta_rad_to_normalized(request)),
                                      throttle=lon.throttle,requested_braking=lon.braking,commands=commands)
        reached=bool(physics_rows and physics_rows[-1]["projection_station_m"]>=formal_end)
        if not reached: failure="formal event not reached"
    except Exception as exc:
        failure=f"{type(exc).__name__}: {exc}"
        (folder/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        initialization=locals().get("initialization",{})
        reached=False
    pd.DataFrame(physics_rows).to_csv(folder/"physics_log.csv",index=False)
    pd.DataFrame(control_rows).to_csv(folder/"control_cycles.csv",index=False)
    pd.DataFrame(horizon_rows).to_csv(folder/"horizon_io.csv",index=False)
    status={"scenario_id":scenario_id,"group":GROUP,"source_group":spec["source_group"],"split":spec["split"],
            "status":"FINISHED" if failure is None and reached else "FAILED","failure":failure,
            "physics_rows":len(physics_rows),"control_cycle_rows":len(control_rows),
            "horizon_rows":len(horizon_rows),"expected_horizon_rows":20*len(control_rows),
            "full_formal_event_reached":reached,"initialization":initialization,
            "reference":reference_meta,"analysis_performed":False}
    save_json(folder/"run_status.json",status); print(json.dumps(status,ensure_ascii=False),flush=True)
    return status


def progress() -> dict:
    rows=[]
    for spec in build_catalog():
        folder=HERE/"logs"/spec["scenario_id"]/GROUP
        if _complete(folder):
            s=json.loads((folder/"run_status.json").read_text(encoding="utf-8")); rows.append(s)
    result={"expected":200,"recorded":len(rows),"finished":sum(x["status"]=="FINISHED" for x in rows),
            "failed":sum(x["status"]=="FAILED" for x in rows),"remaining":200-len(rows),
            "by_split":pd.Series([x["split"] for x in rows]).value_counts().to_dict() if rows else {}}
    save_json(HERE/"manifests/batch_progress.json",result); return result


def run_all(splits: tuple[str,...]) -> dict:
    manifest=prepare(); errors=HERE/"manifests/run_dispatch_errors.jsonl"
    for spec in build_catalog():
        if spec["split"] not in splits: continue
        try: run_one(spec["scenario_id"], manifest)
        except Exception as exc:
            with errors.open("a",encoding="utf-8") as f:
                f.write(json.dumps({"scenario_id":spec["scenario_id"],"error":f"{type(exc).__name__}: {exc}"})+"\n")
        print(json.dumps(progress()),flush=True)
    return progress()


def main():
    p=argparse.ArgumentParser(); p.add_argument("action",choices=("prepare","run","all","progress"))
    p.add_argument("--scenario"); p.add_argument("--splits",nargs="+",choices=("train","validation","test"),
                                                default=("train","validation","test"))
    a=p.parse_args()
    if a.action=="prepare": print(json.dumps(prepare(),indent=2)); return
    if a.action=="progress": print(json.dumps(progress(),indent=2)); return
    if a.action=="run":
        if not a.scenario: raise SystemExit("--scenario is required")
        print(json.dumps(run_one(a.scenario),indent=2)); progress(); return
    print(json.dumps(run_all(tuple(a.splits)),indent=2))


if __name__=="__main__": main()
