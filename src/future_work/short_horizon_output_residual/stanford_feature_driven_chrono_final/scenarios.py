"""Independent analytic reconstructions fitted to frozen Stanford feature anchors."""
from __future__ import annotations
from dataclasses import dataclass
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from future_work.later_mpc.reference_path import ReferencePath
from future_work.later_mpc.types import ReferencePoint
from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import SpeedProfile

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; TS=.05
SOURCE_TERRAIN=HERE.parent/"ft03_terrain_coverage_rebuild/configs/C-MU100-FULL_terrain.json"

class CurvatureTemplatePath(ReferencePath):
    name="stanford_feature_analytic_curvature_template"
    def __init__(self,station_knots,curvature_knots):
        self.station_knots=np.asarray(station_knots,float);self.curvature_knots=np.asarray(curvature_knots,float)
        if len(self.station_knots)<3 or np.any(np.diff(self.station_knots)<=0):raise ValueError("invalid template stations")
        self._s=np.linspace(0,float(self.station_knots[-1]),max(2001,int(self.station_knots[-1]/.05)+1))
        self._k=np.interp(self._s,self.station_knots,self.curvature_knots)
        ds=np.diff(self._s);self._psi=np.r_[0,np.cumsum(.5*ds*(self._k[:-1]+self._k[1:]))]
        self._x=np.r_[0,np.cumsum(.5*ds*(np.cos(self._psi[:-1])+np.cos(self._psi[1:])))]
        self._y=np.r_[0,np.cumsum(.5*ds*(np.sin(self._psi[:-1])+np.sin(self._psi[1:])))]
    @property
    def length_m(self):return float(self._s[-1])
    def sample(self,s_m):
        s=float(np.clip(s_m,0,self.length_m));return ReferencePoint(s,float(np.interp(s,self._s,self._x)),float(np.interp(s,self._s,self._y)),float(np.interp(s,self._s,self._psi)),float(np.interp(s,self._s,self._k)))
    def project(self,X_m,Y_m):
        d=(self._x-X_m)**2+(self._y-Y_m)**2;return self.sample(float(self._s[int(np.argmin(d))]))

@dataclass(frozen=True)
class BuiltScene:
    spec:dict;path:CurvatureTemplatePath;profile:SpeedProfile;formal_start_m:float;formal_end_m:float;terrain_path:Path;terrain_mu:float;supervisor_mu:float;source_fit:dict

def source_path(name):return ROOT/"VehicleDynamicsDataset"/("Oct2023" if name.startswith("2023-10") else "Nov2023")/name
def load_anchor_series(spec):
    cols=["t_s","Ve_mps","Vn_mps","yawRate_radps","Vx_mps","ayCG_mps2"]
    q=pd.read_csv(source_path(spec["source_file"]),skiprows=2,usecols=cols).sort_values("t_s").drop_duplicates("t_s")
    t=np.arange(float(spec["start_s"]),float(spec["end_s"])+.25*TS,TS);src=q.t_s.to_numpy(float)
    z={c:np.interp(t,src,q[c].to_numpy(float)) for c in cols if c!="t_s"};z["t_s"]=t
    ground=np.maximum(np.hypot(z["Ve_mps"],z["Vn_mps"]),1.0);raw=z["yawRate_radps"]/ground
    z["curvature"]=savgol_filter(raw,min(31,len(raw)//2*2-1),3,mode="interp")
    z["station"]=np.r_[0,np.cumsum(.5*TS*(z["Vx_mps"][:-1]+z["Vx_mps"][1:]))]
    return z

def build_scene(spec,write_config=True):
    z=load_anchor_series(spec);idx=np.linspace(0,len(z["t_s"])-1,9).round().astype(int)
    es=z["station"][idx];ek=z["curvature"][idx];ev=z["Vx_mps"][idx]
    if spec["category"] in {"straight_emergency_brake","split_mu_brake"}:ek=np.zeros_like(ek)
    lead=max(45.,2.*float(ev[0]));transition=max(20.,float(ev[0]));trail=60.
    ks=[0.,0.,float(ek[0])];ss=[0.,lead-transition,lead]
    for s,k in zip(es[1:],ek[1:]):ss.append(lead+float(s));ks.append(float(k))
    ss.append(lead+float(es[-1])+trail);ks.append(0.)
    # Remove any accidental duplicate stations without changing order.
    keep=np.r_[True,np.diff(ss)>1e-8];ss=np.asarray(ss)[keep];ks=np.asarray(ks)[keep]
    path=CurvatureTemplatePath(ss,ks)
    mu=.5 if spec["category"]=="low_mu_turn" else 1.
    scale=1.
    if mu<1:
        peak=max(np.max(np.abs(ek)),1e-6);scale=min(1.,math.sqrt(.80*mu*9.81/peak)/max(np.max(ev),1.))
    ev=np.maximum(3.,ev*scale)
    ps=np.r_[0.,lead,lead+es[1:],path.length_m];pv=np.r_[ev[0],ev[0],ev[1:],ev[-1]]
    profile=SpeedProfile(ps,pv,f"source_feature_{spec['scenario_id']}")
    predicted_k=np.interp(z["station"],es,ek);predicted_v=np.interp(z["station"],es,ev)
    fit={"knot_count":9,"independent_geometry":"curvature integrated from nine source-feature knots; no GNSS positions used","speed_scale":scale,
      "curvature_rmse_per_m":float(np.sqrt(np.mean((predicted_k-z["curvature"])**2))),"yaw_rate_rmse_radps":float(np.sqrt(np.mean((predicted_k*z["Vx_mps"]-z["yawRate_radps"])**2))),"speed_rmse_mps":float(np.sqrt(np.mean((predicted_v-z["Vx_mps"]*scale)**2))),
      "source_event_length_m":float(es[-1]),"source_peak_abs_curvature_per_m":float(np.max(np.abs(z["curvature"]))),"template_peak_abs_curvature_per_m":float(np.max(np.abs(ek))),"source_mean_speed_mps":float(np.mean(z["Vx_mps"])),"template_mean_speed_mps":float(np.mean(ev))}
    samples=[path.sample(s) for s in np.linspace(0,path.length_m,2001)];x=np.array([p.X_m for p in samples]);y=np.array([p.Y_m for p in samples])
    terrain=json.loads(SOURCE_TERRAIN.read_text(encoding="utf8"));patch=terrain["Patches"][0];cx=float((x.min()+x.max())/2);cy=float((y.min()+y.max())/2);sx=float(max(500,math.ceil(x.max()-x.min()+200)));sy=float(max(400,math.ceil(y.max()-y.min()+200)))
    patch["Location"]=[cx,cy,0];patch["Geometry"]["Dimensions"]=[sx,sy,2];patch["Contact Material"]["Coefficient of Friction"]=mu
    tp=HERE/"configs"/f"{spec['scenario_id']}_terrain.json"
    if write_config:
        tp.parent.mkdir(parents=True,exist_ok=True);tp.write_text(json.dumps(terrain,indent=2)+"\n",encoding="utf8")
    supervisor_mu=1. if spec["category"]=="split_mu_brake" else mu
    return BuiltScene(spec,path,profile,lead,lead+float(es[-1]),tp,mu,supervisor_mu,fit)

def all_scenes():
    selected=json.loads((HERE/"source_screening/selected_source_anchors.json").read_text(encoding="utf8"))["selected"]
    return [build_scene(s) for s in selected]
