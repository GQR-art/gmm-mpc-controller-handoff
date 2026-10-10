from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class CEGMRModel:
    """Joint-GMM conditional regressor with predictive covariance."""

    scaler: StandardScaler
    gmm: GaussianMixture
    input_dim: int
    target_dim: int
    input_columns: list[str]
    target_columns: list[str]

    def predict(self, x: np.ndarray, chunk_size: int = 20000):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x with shape (n, {self.input_dim}), got {x.shape}")
        if len(x) == 0:
            return (
                np.empty((0, self.target_dim), dtype=float),
                np.empty((0, self.target_dim, self.target_dim), dtype=float),
                np.empty((0, self.gmm.n_components), dtype=float),
            )
        means = []
        covariances = []
        responsibilities = []
        for start in range(0, len(x), chunk_size):
            stop = min(start + chunk_size, len(x))
            mean, covariance, responsibility = self._predict_chunk(x[start:stop])
            means.append(mean)
            covariances.append(covariance)
            responsibilities.append(responsibility)
        return (
            np.concatenate(means, axis=0),
            np.concatenate(covariances, axis=0),
            np.concatenate(responsibilities, axis=0),
        )

    def _predict_chunk(self, x: np.ndarray):
        d_x = self.input_dim
        d_y = self.target_dim
        n = len(x)
        k_count = self.gmm.n_components

        x_mean = self.scaler.mean_[:d_x]
        x_scale = self.scaler.scale_[:d_x]
        x_std = (np.asarray(x, dtype=float) - x_mean) / x_scale

        component_means = np.empty((n, k_count, d_y), dtype=float)
        component_covariances = np.empty((k_count, d_y, d_y), dtype=float)
        log_prob = np.empty((n, k_count), dtype=float)

        for k in range(k_count):
            mu = self.gmm.means_[k]
            sigma = self.gmm.covariances_[k]
            mu_x = mu[:d_x]
            mu_y = mu[d_x:]
            s_xx = sigma[:d_x, :d_x]
            s_xy = sigma[:d_x, d_x:]
            s_yx = sigma[d_x:, :d_x]
            s_yy = sigma[d_x:, d_x:]
            inv_xx = np.linalg.pinv(s_xx)
            gain = s_yx @ inv_xx
            component_means[:, k, :] = mu_y + (x_std - mu_x) @ gain.T
            conditional = s_yy - gain @ s_xy
            component_covariances[k] = 0.5 * (conditional + conditional.T)

            sign, logdet = np.linalg.slogdet(s_xx)
            if sign <= 0:
                logdet = np.log(max(np.linalg.det(s_xx + 1e-9 * np.eye(d_x)), 1e-300))
            centered = x_std - mu_x
            mahalanobis = np.einsum("ni,ij,nj->n", centered, inv_xx, centered)
            log_prob[:, k] = (
                np.log(max(self.gmm.weights_[k], 1e-300))
                - 0.5 * (d_x * np.log(2.0 * np.pi) + logdet + mahalanobis)
            )

        log_norm = logsumexp(log_prob, axis=1, keepdims=True)
        responsibility = np.exp(log_prob - log_norm)
        mean_std = np.einsum("nk,nkd->nd", responsibility, component_means)

        covariance_std = np.zeros((n, d_y, d_y), dtype=float)
        for k in range(k_count):
            diff = component_means[:, k, :] - mean_std
            second = component_covariances[k][None, :, :] + np.einsum(
                "ni,nj->nij", diff, diff
            )
            covariance_std += responsibility[:, k, None, None] * second

        y_mean = self.scaler.mean_[d_x:]
        y_scale = self.scaler.scale_[d_x:]
        mean = mean_std * y_scale + y_mean
        scale_outer = np.outer(y_scale, y_scale)
        covariance = covariance_std * scale_outer[None, :, :]
        return mean, covariance, responsibility
