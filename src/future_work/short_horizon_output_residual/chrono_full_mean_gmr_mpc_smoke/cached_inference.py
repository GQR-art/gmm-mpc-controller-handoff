"""Frozen GMR math with component constants calculated once per model instance.

The row/component loops and floating-point operation order are retained. No
training, covariance regularization, scaler settings or solver changes occur.
"""
import numpy as np
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import RolloutAlignedSequentialLayer
from future_work.short_horizon_output_residual.mpc_integration.full_residual_propagated_gmr_mpc.adapter import FullResidualPropagatedMeanGMRMPC
from future_work.later_mpc.controller import NominalMPCController
from .bridge.chrono_full_mean_gmr_bridge import FullMeanGMRChronoBridge


class CachedRolloutAlignedLayer(RolloutAlignedSequentialLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d=self.model.input_dim
        self._components=[]
        for k in range(self.model.gmm.n_components):
            cov=self.model.gmm.covariances_[k]
            inv=np.linalg.pinv(cov[:d,:d])
            _,ld=np.linalg.slogdet(cov[:d,:d])
            cyx=cov[d:,:d]
            gain=cyx@inv
            self._components.append((self.model.gmm.means_[k,:d].copy(),self.model.gmm.means_[k,d:].copy(),
                                     inv,ld,np.log(self.model.gmm.weights_[k]),gain,cov[d:,d:]-gain@cyx.T))

    def _conditional_prediction(self,x):
        x=np.asarray(x,float); d=self.model.input_dim
        scaled=self.model.scaler.transform(np.c_[x,np.zeros((len(x),2))])[:,:d]
        out_mean=[]; out_cov=[]
        for row in scaled:
            log=[]; cond=[]
            for mux,muy,inv,ld,prior_log,gain,conditional_cov in self._components:
                diff=row-mux
                log.append(prior_log-.5*(d*np.log(2*np.pi)+ld+diff@inv@diff))
                cond.append((muy+gain@diff,conditional_cov))
            a=np.exp(np.array(log)-np.max(log)); a/=a.sum()
            m=sum(w*c[0] for w,c in zip(a,cond))
            c=sum(w*(cc+(mu-m)[:,None]@(mu-m)[None,:]) for w,(mu,cc) in zip(a,cond))
            out_mean.append(m); out_cov.append(c)
        full=np.c_[np.zeros((len(x),d)),np.asarray(out_mean)]
        y=self.model.scaler.inverse_transform(full)[:,d:]
        scale=self.model.scaler.scale_[d:]
        cov=np.asarray(out_cov)*scale[None,:,None]*scale[None,None,:]
        return y,cov


def make_cached_bridge(reference_path):
    controller=NominalMPCController()
    layer=CachedRolloutAlignedLayer()
    adapter=FullResidualPropagatedMeanGMRMPC(controller.optimizer.config,layer=layer,max_iterations=1)
    return FullMeanGMRChronoBridge(reference_path=reference_path,controller=controller,full_adapter=adapter)
