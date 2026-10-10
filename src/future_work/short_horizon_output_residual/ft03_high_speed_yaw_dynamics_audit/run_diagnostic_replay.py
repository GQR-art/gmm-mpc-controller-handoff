"""Frozen FT03 C0 replay with read-only tire-force and MPC-preview logging."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

from future_work.chrono_mpc_integration.bridge.speed_controller import LongitudinalSpeedPI
from future_work.chrono_mpc_integration.scenarios import ChronoMPCScenario
from future_work.later_mpc.error_state import global_state_to_error_state
from future_work.chrono_mpc_integration.bridge.state_mapper import chrono_raw_to_vehicle_state
from future_work.short_horizon_output_residual.chrono_full_mean_gmr_mpc_smoke.run_smoke import _prepare_formal_plant
from future_work.short_horizon_output_residual.ft03_reference_nominal_failure_localization.reference_path import build_path
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.profiles import stanford_profile
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import VariableSpeedNominalBridge
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.run_experiment import hashes

HERE=Path(__file__).resolve().parent;OUT=HERE/'logs'/'diagnostic_c0'
DT=.002;MPC_DT=.05;N=20

def vec(v):return np.array([float(v.x),float(v.y),float(v.z)])

def force_snapshot(plant):
    body=plant._vehicle.GetChassisBody();com=vec(body.GetPos());rows={};front=rear=0.;total=0.
    ids=(("FL","FR"),("RL","RR"))
    for ai,axle in enumerate(plant._vehicle.GetAxles()):
        for wi,wheel in enumerate(axle.GetWheels()):
            wid=ids[ai][wi];tire=wheel.GetTire()
            owner=tire.ReportTireForce(plant._terrain);force=vec(owner.force);point=vec(owner.point);moment=vec(owner.moment)
            local_owner=tire.ReportTireForceLocal(plant._terrain,plant._chrono.ChCoordsysd());local=vec(local_owner.force)
            body_force=vec(body.TransformDirectionParentToLocal(owner.force));lever=point-com
            cross_z=float(np.cross(lever,force)[2]);mz=float(cross_z+moment[2]);state=wheel.GetState()
            rows.update({f'{wid}_Fx_world_N':force[0],f'{wid}_Fy_world_N':force[1],f'{wid}_Fz_world_N':force[2],
              f'{wid}_Fx_body_N':body_force[0],f'{wid}_Fy_body_N':body_force[1],f'{wid}_Fz_body_N':body_force[2],
              f'{wid}_Fx_tire_N':local[0],f'{wid}_Fy_tire_N':local[1],f'{wid}_Fz_tire_N':local[2],
              f'{wid}_point_X_m':point[0],f'{wid}_point_Y_m':point[1],f'{wid}_point_Z_m':point[2],
              f'{wid}_moment_Z_Nm':moment[2],f'{wid}_r_cross_F_Z_Nm':cross_z,f'{wid}_yaw_moment_about_chassis_Nm':mz,
              f'{wid}_wheel_omega_radps':float(state.omega)})
            if ai==0:front+=mz
            else:rear+=mz
            total+=mz
    rows.update({'front_axle_yaw_moment_Nm':front,'rear_axle_yaw_moment_Nm':rear,'total_tire_yaw_moment_Nm':total})
    return rows

def run():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True);before=hashes()
    built=build_path(77.5,82.45);profile=stanford_profile(built.frame)
    scenario=ChronoMPCScenario(0.,0.,0.,float(profile.sample(0)),35.,built.path)
    plant,calibration,initialization=_prepare_formal_plant(scenario,'staged')
    bridge=VariableSpeedNominalBridge(built.path,profile);bridge.reset(0.)
    pi=LongitudinalSpeedPI();pi.reset();ratio=int(round(MPC_DT/DT));rows=[];mpc=None;normalized=0.;prev_delta=0.;delta_inc=0.;start=plant.get_raw_state().time
    for step in range(int(35/DT)+1):
        raw=plant.get_raw_state();projected=built.path.project(raw.X,raw.Y)
        if step%ratio==0:
            prev_delta=bridge.previous_delta_rad;mpc=bridge.compute_control(raw);delta_inc=mpc.delta_rad-prev_delta
            normalized=calibration.delta_rad_to_normalized(mpc.delta_rad)
        else:delta_inc=0.
        speed_ref=float(profile.sample(projected.s_m));lon=pi.update(vx_reference_mps=speed_ref,vx_measured_mps=raw.vx_body,dt_s=DT)
        err=global_state_to_error_state(chrono_raw_to_vehicle_state(raw),built.path);wheel_diag=plant.get_wheel_diagnostics();wmap={w.wheel_id:w for w in wheel_diag};susp=plant.get_suspension_diagnostics()
        row={'time_s':raw.time-start,'X_m':raw.X,'Y_m':raw.Y,'psi_rad':raw.psi,'vx_mps':raw.vx_body,'vy_mps':raw.vy_body,'r_radps':raw.yaw_rate_body,
          'ey_m':err.ey_m,'epsi_rad':err.epsi_rad,'projection_station_m':projected.s_m,'remaining_path_m':built.path.length_m-projected.s_m,
          'reference_X_m':projected.X_m,'reference_Y_m':projected.Y_m,'reference_psi_rad':projected.psi_rad,'reference_curvature_per_m':projected.curvature_per_m,
          'speed_reference_mps':speed_ref,'reference_r_radps':speed_ref*projected.curvature_per_m,'actual_path_r_demand_radps':raw.vx_body*projected.curvature_per_m,
          'ay_reference_demand_mps2':speed_ref**2*projected.curvature_per_m,'delta_command_rad':mpc.delta_rad,'delta_actual_rad':float(math.atan(np.mean(np.tan([w.road_wheel_angle_rad for w in wheel_diag if w.axle_index==0])))),
          'delta_increment_rad':delta_inc,'steering_normalized':normalized,'throttle':lon.throttle,'brake':lon.braking,'solver_status':mpc.solution.status,
          'solver_time_s':mpc.solution.solve_time_s,'mpc_updated':step%ratio==0,'roll_rad':plant.get_chassis_roll_pitch_rad()[0],
          'beta_rad':math.atan2(raw.vy_body,raw.vx_body),'ay_com_mps2':plant.get_aggregate_com_acceleration_body_mps2()[1],
          'wheel_contact_count':sum(x.contact for x in susp),'formal_interval':built.formal_start_s<=projected.s_m<=built.formal_end_s}
        for wid in ('FL','FR','RL','RR'):
            w=wmap[wid];row.update({f'{wid}_Fz_N':w.force_z_tire_N,f'{wid}_slip_angle_rad':w.slip_angle_rad,f'{wid}_tmeasy_slip_angle_rad':w.tmeasy_slip_angle_rad,
             f'{wid}_lateral_slip_sy':w.lateral_slip_sy,f'{wid}_longitudinal_slip_sx':w.longitudinal_slip_sx,f'{wid}_contact':bool(next(x.contact for x in susp if x.wheel_id==wid))})
        row.update(force_snapshot(plant))
        if step%ratio==0:
            controls=np.asarray(mpc.solution.delta_sequence,float);states=np.asarray(mpc.solution.predicted_error_states,float);speeds=np.asarray(mpc.solution.diagnostics['speed_preview_mps'],float)
            for i in range(N):
                row[f'candidate_delta_{i+1:02d}_rad']=controls[i];row[f'preview_speed_{i+1:02d}_mps']=speeds[i];row[f'preview_kappa_{i+1:02d}_per_m']=mpc.reference_preview.curvature_per_m[i]
                for j,name in enumerate(('ey','epsi','vy','r')):row[f'candidate_{name}_{i+1:02d}']=states[i+1,j]
        rows.append(row)
        if projected.s_m>=built.formal_end_s:break
        plant.advance(DT,normalized_steering=normalized,throttle=lon.throttle,braking=lon.braking)
    frame=pd.DataFrame(rows);frame['delta_actual_dot_radps']=np.gradient(frame.delta_actual_rad,DT,edge_order=2);frame['ay_kin_mps2']=np.gradient(frame.vy_mps,DT,edge_order=2)+frame.vx_mps*frame.r_radps
    frame.to_csv(OUT/'physics_log.csv',index=False);frame.loc[frame.mpc_updated].to_csv(OUT/'mpc_log.csv',index=False)
    after=hashes();summary={'success':True,'samples':len(frame),'simulation_time_s':float(frame.time_s.iloc[-1]),'initialization':asdict(initialization),
      'solver_failures':int((~frame.loc[frame.mpc_updated,'solver_status'].str.lower().str.startswith('solved')).sum()),'min_contacts':int(frame.wheel_contact_count.min()),
      'all_finite_base':bool(np.isfinite(frame[['X_m','Y_m','psi_rad','vx_mps','vy_mps','r_radps']]).all().all()),'hashes_unchanged':before==after}
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(HERE/'artifact_hashes_before_after.json').write_text(json.dumps({'before':before,'after':after,'unchanged':before==after},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
