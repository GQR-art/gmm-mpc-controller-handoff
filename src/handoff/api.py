from pathlib import Path
from collections.abc import Mapping
import joblib
import numpy as np
from future_work.short_horizon_output_residual.gmm_gmr.gmm_model import JointGMM
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import RolloutAlignedSequentialLayer

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ('current_vx_mps','current_vy_mps','current_r_radps','current_ey_m','current_epsi_rad',
           'request_road_wheel_rad','actual_road_wheel_rad','previous_request_road_wheel_rad',
           'preview_curvature_per_m','preview_speed_mps','pi_integral_before','abs_multiplier_min',
           'F1_next_vy_mps','F1_next_r_radps')
HISTORY = tuple(f'history_{window}_{name}_rate' for window in ('020s','005s')
                for name in ('vx_mps','vy_mps','r_radps','delta_actual_rad'))
TARGETS = ('q_vy_interval_mps2','q_r_interval_radps2')

class ControlGMM1(RolloutAlignedSequentialLayer):
    """Same K3 predictive math; excludes training-distribution diagnostics.

    predict(state, condition, controls, vx_preview) is inherited unchanged.
    Diagnostic methods needing the training table are deliberately unsupported.
    """
    def __init__(self, model_path=None):
        self.model = JointGMM.load(model_path or ROOT/'models/gmm1_control_k3.joblib')
        if self.model.input_dim != 6 or self.model.gmm.n_components != 3:
            raise ValueError('Expected the frozen six-input K3 control model')

    def audit_candidate_controls(self, *args, **kwargs):
        raise NotImplementedError('Use the original reference layer for distribution diagnostics')

    def audit_prediction_features(self, *args, **kwargs):
        raise NotImplementedError('Use the original reference layer for distribution diagnostics')

class GMM2:
    """Feature dictionaries in physical units. Scaling stays inside the model."""
    def __init__(self, variant='history', model_path=None):
        if variant not in ('current','history'):
            raise ValueError('variant must be current or history')
        self.artifact = joblib.load(model_path or ROOT/f'models/gmm2_{variant}.joblib')
        self.features = CURRENT + (HISTORY if variant=='history' else ())
        if tuple(self.artifact['features']) != self.features or tuple(self.artifact['targets']) != TARGETS:
            raise ValueError('Artifact feature/target schema does not match this interface')

    def predict(self, rows):
        if isinstance(rows, Mapping): rows=[rows]
        x=np.asarray([[row[n] for n in self.features] for row in rows],dtype=float)
        if x.ndim!=2 or not np.isfinite(x).all():
            raise ValueError('Expected finite named feature rows')
        mean,covariance,responsibility=self.artifact['model'].predict(x)
        base=np.asarray([[row['F1_next_vy_mps'],row['F1_next_r_radps']] for row in rows],float)
        return {'mean':mean,'covariance':covariance,'responsibility':responsibility,
                'next_vr':base+0.05*mean}

def history_features(frame):
    """101 regular physical samples ending at query time, including both ends.

    Caller must supply only observed history; timestamps are validated for spacing,
    not for wall-clock availability. Returns the exact original least-squares slopes.
    """
    if len(frame)!=101: raise ValueError('Expected 101 samples spanning 0.20 s')
    names=('vx_mps','vy_mps','r_radps','delta_actual_rad')
    t=np.asarray(frame['time_s'],float)
    if not np.isfinite(t).all() or not np.allclose(np.diff(t),.002,atol=1e-9,rtol=0):
        raise ValueError('History must be ordered, continuous at 0.002 s')
    result={}
    for tag,count in [('020s',101),('005s',26)]:
        tx=t[-count:];xc=tx-tx.mean()
        for name in names:
            y=np.asarray(frame[name],float)[-count:]
            if not np.isfinite(y).all(): raise ValueError('Nonfinite history')
            result[f'history_{tag}_{name}_rate']=float(np.dot(xc,y-y.mean())/np.dot(xc,xc))
    return result

class PaperGMM1:
    """Paper K15 scalar lateral-acceleration residual, never an F1 drop-in.

    Uses predict_residual only: no gating or smoothing is invoked.
    """
    features=('vx','steering_angle','steering_rate','yaw_rate','longitudinal_acceleration','ay_bike')
    def __init__(self, model_path=None):
        from residual_model.gmm_residual import GMMResidualModel
        self.model=GMMResidualModel.load(str(model_path or ROOT/'models/gmm1_paper_k15.joblib'))
    def predict(self,x):
        a=np.asarray(x,float)
        if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all():
            raise ValueError('Expected finite [n,6] paper features in documented order')
        return self.model.predict_residual(a)
