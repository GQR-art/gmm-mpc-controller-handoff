"""Run pre-registered FT03 variable-speed Nominal-MPC groups without touching frozen code."""

from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import argparse, json, math

import numpy as np
import pandas as pd

from future_work.chrono_mpc_integration.bridge.speed_controller import LongitudinalSpeedPI
from future_work.chrono_mpc_integration.scenarios import ChronoMPCScenario
from future_work.later_mpc.error_state import global_state_to_error_state
from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state
from future_work.short_horizon_output_residual.chrono_full_mean_gmr_mpc_smoke.run_smoke import _prepare_formal_plant
from future_work.short_horizon_output_residual.ft03_reference_nominal_failure_localization.reference_path import build_path

from .profiles import constant_profile, stanford_profile, curvature_limited_profile, source_stations, AY_ALLOW_MPS2
from .variable_speed_mpc import VariableSpeedNominalBridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOGS = HERE / "logs"
MANIFESTS = HERE / "manifests"
PHYSICS_DT = .002
MPC_DT = .05
MAX_RUN_TIME = 35.0

PROTECTED = [
    ROOT / "bicycle_model.py", ROOT / "FITTED_MODEL_PARAMETERS.json",
    ROOT / "future_work/later_mpc/controller.py", ROOT / "future_work/later_mpc/mpc_optimizer.py",
    ROOT / "future_work/later_mpc/nominal_bicycle_predictor.py", ROOT / "future_work/later_mpc/reference_path.py",
    ROOT / "future_work/chrono_mpc_integration/bridge/chrono_vehicle.py",
    ROOT / "future_work/chrono_mpc_integration/bridge/runner.py",
    ROOT / "future_work/chrono_mpc_integration/bridge/speed_controller.py",
    ROOT / "future_work/chrono_mpc_integration/bridge/steering_adapter.py",
    ROOT / "future_work/chrono_mpc_integration/configs/steering/golflike3d_v1_steering_calibration.json",
]

def hashes():
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest().upper() for p in PROTECTED}

def _equivalent_delta(wheels):
    front = [w.road_wheel_angle_rad for w in wheels if w.axle_index == 0]
    return float(math.atan(np.mean(np.tan(front))))

def profiles():
    b = build_path(77.5, 82.45)
    return b, {"B_fixed15": constant_profile(b.frame), "C_stanford": stanford_profile(b.frame),
               "D_curvature_cap": curvature_limited_profile(b.frame)}

def write_preregistered_assets():
    b, ps = profiles(); MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_before_outcome_runs": True, "groups": list(ps), "source": "2023-11_11.csv",
        "formal_source_interval_s": [77.5,82.45], "physics_dt_s": PHYSICS_DT, "mpc_dt_s": MPC_DT,
        "horizon_steps": 20, "max_run_time_s": MAX_RUN_TIME, "terrain_friction": .9,
        "curvature_cap": {"ay_allow_mps2": AY_ALLOW_MPS2, "basis": "pre-registered 0.8g diagnostic",
          "acceleration_bound_mps2": 1.5, "deceleration_bound_mps2": 2.5,
          "bounds_status": "conservative diagnostic assumptions; not identified GolfLike3D capability"},
        "profile_hashes": {k: sha256(np.c_[v.station_m,v.speed_mps].tobytes()).hexdigest().upper() for k,v in ps.items()},
    }
    (MANIFESTS/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    rows=[]
    for name,p in ps.items():
        for s,v in zip(p.station_m,p.speed_mps): rows.append({"group":name,"station_m":s,"speed_reference_mps":v})
    pd.DataFrame(rows).to_csv(HERE/"speed_profile_comparison.csv",index=False)
    return b,ps,manifest

def run_group(name: str, overwrite=False):
    b, ps, manifest = write_preregistered_assets(); p=ps[name]
    folder=LOGS/name
    if folder.exists() and not overwrite: raise FileExistsError(folder)
    folder.mkdir(parents=True,exist_ok=True)
    initial_speed=float(p.sample(0.0))
    scenario=ChronoMPCScenario(0.,0.,0.,initial_speed,MAX_RUN_TIME,b.path)
    plant,calibration,initialization=_prepare_formal_plant(scenario,"staged")
    bridge=VariableSpeedNominalBridge(b.path,p); bridge.reset(0.)
    speed_pi=LongitudinalSpeedPI(); speed_pi.reset()
    ratio=int(round(MPC_DT/PHYSICS_DT)); rows=[]; failure=None; reached=False
    raw=plant.get_raw_state(); start_time=raw.time; mpc_out=None; normalized=0.; prev_delta=0.; delta_inc=0.
    for step in range(int(MAX_RUN_TIME/PHYSICS_DT)+1):
        raw=plant.get_raw_state(); projected=b.path.project(raw.X,raw.Y)
        if step%ratio==0:
            prev_delta=bridge.previous_delta_rad; mpc_out=bridge.compute_control(raw)
            delta_inc=mpc_out.delta_rad-prev_delta
            normalized=calibration.delta_rad_to_normalized(mpc_out.delta_rad)
        else: delta_inc=0.
        speed_ref=float(p.sample(projected.s_m))
        lon=speed_pi.update(vx_reference_mps=speed_ref,vx_measured_mps=raw.vx_body,dt_s=PHYSICS_DT)
        err=global_state_to_error_state(chrono_raw_to_vehicle_state(raw),b.path)
        wheels=plant.get_wheel_diagnostics(); susp=plant.get_suspension_diagnostics(); wmap={w.wheel_id:w for w in wheels}
        row={"time_s":raw.time-start_time,"X_m":raw.X,"Y_m":raw.Y,"psi_rad":raw.psi,
             "vx_mps":raw.vx_body,"vy_mps":raw.vy_body,"r_radps":raw.yaw_rate_body,
             "ey_m":err.ey_m,"epsi_rad":err.epsi_rad,"projection_station_m":projected.s_m,
             "remaining_path_m":b.path.length_m-projected.s_m,"reference_X_m":projected.X_m,"reference_Y_m":projected.Y_m,
             "reference_psi_rad":projected.psi_rad,"reference_curvature_per_m":projected.curvature_per_m,
             "speed_reference_mps":speed_ref,"reference_r_radps":speed_ref*projected.curvature_per_m,
             "actual_path_r_demand_radps":raw.vx_body*projected.curvature_per_m,
             "ay_reference_demand_mps2":speed_ref**2*projected.curvature_per_m,
             "delta_command_rad":mpc_out.delta_rad,"delta_actual_rad":_equivalent_delta(wheels),
             "delta_increment_rad":delta_inc,"steering_normalized":normalized,"throttle":lon.throttle,"brake":lon.braking,
             "solver_status":mpc_out.solution.status,"solver_time_s":mpc_out.solution.solve_time_s,"mpc_updated":step%ratio==0,
             "roll_rad":plant.get_chassis_roll_pitch_rad()[0],"beta_rad":math.atan2(raw.vy_body,raw.vx_body),
             "ay_com_mps2":plant.get_aggregate_com_acceleration_body_mps2()[1],
             "wheel_contact_count":sum(x.contact for x in susp),"formal_interval":b.formal_start_s<=projected.s_m<=b.formal_end_s}
        for wid in ("FL","FR","RL","RR"):
            row[f"{wid}_Fz_N"]=wmap[wid].force_z_tire_N; row[f"{wid}_slip_angle_rad"]=wmap[wid].slip_angle_rad
            row[f"{wid}_lateral_slip_sy"]=wmap[wid].lateral_slip_sy; row[f"{wid}_longitudinal_slip_sx"]=wmap[wid].longitudinal_slip_sx
            row[f"{wid}_contact"]=bool(next(x.contact for x in susp if x.wheel_id==wid))
        rows.append(row)
        if projected.s_m>=b.formal_end_s:
            reached=True; break
        if step==int(MAX_RUN_TIME/PHYSICS_DT): break
        try: plant.advance(PHYSICS_DT,normalized_steering=normalized,throttle=lon.throttle,braking=lon.braking)
        except Exception as exc: failure=f"{type(exc).__name__}: {exc}"; break
    frame=pd.DataFrame(rows)
    if len(frame)>2:
        frame["delta_actual_dot_radps"]=np.gradient(frame.delta_actual_rad.to_numpy(float),PHYSICS_DT,edge_order=2)
        frame["ay_kin_mps2"]=np.gradient(frame.vy_mps.to_numpy(float),PHYSICS_DT,edge_order=2)+frame.vx_mps*frame.r_radps
    frame.to_csv(folder/"physics_log.csv",index=False); frame.loc[frame.mpc_updated].to_csv(folder/"mpc_log.csv",index=False)
    summary={"group":name,"successfully_executed":failure is None,"failure":failure,"path_completed":reached,
             "completion_ratio":float(min(1,frame.projection_station_m.max()/b.formal_end_s)),
             "completion_time_s":float(frame.time_s.iloc[-1]),"initial_speed_reference_mps":initial_speed,
             "initialization":asdict(initialization),"samples":len(frame),"all_finite":bool(np.isfinite(frame.select_dtypes('number')).all().all()),
             "solver_failures":int((~frame.loc[frame.mpc_updated,'solver_status'].str.lower().str.startswith('solved')).sum()),
             "min_contacts":int(frame.wheel_contact_count.min()),"formal_start_station_m":b.formal_start_s,"formal_end_station_m":b.formal_end_s}
    (folder/"run_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("group",choices=["B_fixed15","C_stanford","D_curvature_cap","all"]);ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
    before=hashes();MANIFESTS.mkdir(parents=True,exist_ok=True);(MANIFESTS/'artifact_hashes_before.json').write_text(json.dumps(before,indent=2)+'\n')
    groups=["B_fixed15","C_stanford","D_curvature_cap"] if args.group=='all' else [args.group]
    out=[run_group(g,args.overwrite) for g in groups]
    after=hashes();(MANIFESTS/'artifact_hashes_after.json').write_text(json.dumps(after,indent=2)+'\n')
    (MANIFESTS/'artifact_hashes_before_after.json').write_text(json.dumps({'before':before,'after':after,'unchanged':before==after,'changed':[k for k in before if before[k]!=after.get(k)]},indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()

