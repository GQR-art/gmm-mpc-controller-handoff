"""Open-loop RK4 rollout; only the initial measured [vy, r] is accepted."""
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
from bicycle_model import DynamicBicycleModel
from ..config import HORIZON_STEPS, PARAMETERS_PATH, RK4_INTERNAL_STEPS, SAMPLE_TIME_S
@dataclass(frozen=True)
class Rollout: ay: np.ndarray; yaw_rate: np.ndarray
def model(): return DynamicBicycleModel(json.loads(PARAMETERS_PATH.read_text(encoding="utf8"))["fitted_model_parameters"])
def rollout(initial_vy, initial_r, vx_preview, steering_preview, bicycle=None):
    vx=np.asarray(vx_preview,float); steering=np.asarray(steering_preview,float)
    if len(vx)!=HORIZON_STEPS or len(steering)!=HORIZON_STEPS: raise ValueError("Need exactly 20 preview points")
    state=np.array([initial_vy,initial_r],float); vy=np.empty(HORIZON_STEPS); yaw=np.empty(HORIZON_STEPS); ay=np.empty(HORIZON_STEPS); bicycle=bicycle or model(); h=SAMPLE_TIME_S/RK4_INTERNAL_STEPS
    for i,(speed,delta) in enumerate(zip(vx,steering)):
        if not np.isfinite([speed,delta]).all() or speed<.1: raise ValueError("Invalid preview")
        for _ in range(RK4_INTERNAL_STEPS):
            f=lambda x:np.asarray(bicycle.state_derivative(0.,x.tolist(),float(delta),float(speed)),float)
            k1=f(state); k2=f(state+h*k1/2); k3=f(state+h*k2/2); k4=f(state+h*k3); state=state+h*(k1+2*k2+2*k3+k4)/6
        vy[i],yaw[i]=state; ay[i]=f(state)[0]+speed*state[1]
    if not np.isfinite(np.r_[ay,yaw]).all(): raise ValueError("Non-finite rollout")
    return Rollout(ay,yaw)
