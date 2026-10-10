"""Gaussian Mixture Model / Gaussian Mixture Regression residual model.

Libraries: numpy, scikit-learn (``GaussianMixture``, ``StandardScaler``), scipy
(``multivariate_normal``), joblib (persistence).

A joint full-covariance GMM is fit on standardized ``[X, y]``. Residual
prediction uses Gaussian Mixture Regression (conditional expectation
``E[y | X = x]``) with numerically stable marginal responsibilities.

Input dimension is defined by ``residual_model.config.FEATURE_COLUMNS``; extend
that tuple (and ``N_FEATURES``) for richer regressors without changing the GMR
equations in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from residual_model.config import N_FEATURES
from residual_model.data_utils import assert_feature_matrix_shape, extract_ay_bike_from_X, resample_arrays_by_weight
from residual_model.residual_gating import (
    DEFAULT_RESIDUAL_GATING_CONFIG,
    ResidualGatingConfig,
    apply_gated_correction,
)
from residual_model.residual_smoothing import ResidualSmoothingConfig, smooth_residual_temporal

DEFAULT_REG_COVAR: float = 1e-6
LOG_WEIGHT_EPS: float = 1e-300
RESP_NORM_EPS: float = 1e-300


def _solve_sigma_xx(sigma_xx: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``sigma_xx @ x = rhs`` with fallback for near-singular matrices."""
    try:
        return np.linalg.solve(sigma_xx, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(sigma_xx, rhs, rcond=None)[0]


def _as_2d_X(X: np.ndarray) -> np.ndarray:
    """Coerce ``X`` to a 2-D float array."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _as_1d_y(y: np.ndarray) -> np.ndarray:
    """Coerce ``y`` to a 1-D float array."""
    return np.asarray(y, dtype=float).reshape(-1)


class GMMResidualModel:
    """Joint GMM + GMR model for lateral acceleration residuals.

    Training fits :class:`~sklearn.preprocessing.StandardScaler` on ``X`` and
    ``y``, then fits a :class:`~sklearn.mixture.GaussianMixture` on the joint
    variable ``[X_scaled, y_scaled]``.

    Parameters
    ----------
    n_components
        Number of Gaussian mixture components.
    covariance_type
        Covariance parameterization passed to ``GaussianMixture``.
    reg_covar
        Diagonal regularization added to ``Sigma_xx`` before inversion and to
        the sklearn GMM fit.
    random_state
        Random seed for the mixture fit.
    max_iter
        Maximum EM iterations for the mixture fit.
    n_init
        Number of initializations for the mixture fit.
    """

    def __init__(
        self,
        n_components: int = 4,
        *,
        covariance_type: str = "full",
        reg_covar: float = DEFAULT_REG_COVAR,
        random_state: int | None = 42,
        max_iter: int = 200,
        n_init: int = 1,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)

        self.scaler_x: StandardScaler | None = None
        self.scaler_y: StandardScaler | None = None
        self.gmm: GaussianMixture | None = None
        self.n_features_: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fit."""
        return (
            self.scaler_x is not None
            and self.scaler_y is not None
            and self.gmm is not None
            and self.n_features_ is not None
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "GMMResidualModel is not fitted. Call fit(X, y) before prediction."
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> "GMMResidualModel":
        """Fit scalers and a joint GMM on ``[X_scaled, y_scaled]``.

        Parameters
        ----------
        X
            Feature matrix with shape ``(n_samples, n_features)``.
        y
            Residual target with shape ``(n_samples,)`` or ``(n_samples, 1)``.
        sample_weight
            Optional per-sample weights passed to ``GaussianMixture.fit``.

        Returns
        -------
        GMMResidualModel
            Fitted model (self).
        """
        X = _as_2d_X(X)
        y = _as_1d_y(y)
        assert_feature_matrix_shape(X, n_features=X.shape[1])
        if len(y) != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] < 1:
            raise ValueError("Need at least one sample to fit GMMResidualModel.")
        if X.shape[0] < self.n_components:
            raise ValueError(
                f"Cannot fit n_components={self.n_components} with only "
                f"{X.shape[0]} sample(s). Provide more data or reduce n_components."
            )
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(sample_weight) != X.shape[0]:
                raise ValueError("sample_weight must have the same length as X.")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")
            X, y = resample_arrays_by_weight(
                X, y, sample_weight, random_state=self.random_state
            )

        self.n_features_ = X.shape[1]
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        joint_data = np.column_stack([X_scaled, y_scaled.reshape(-1, 1)])
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )
        self.gmm.fit(joint_data)
        return self

    def _joint_data_scaled(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Build standardized joint matrix ``[X_scaled, y_scaled]``."""
        self._check_fitted()
        assert self.scaler_x is not None
        assert self.scaler_y is not None

        X = _as_2d_X(X)
        y = _as_1d_y(y)
        assert_feature_matrix_shape(X, n_features=self.n_features_ or N_FEATURES)
        if len(y) != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        X_scaled = self.scaler_x.transform(X)
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
        return np.column_stack([X_scaled, y_scaled.reshape(-1, 1)])

    def bic(self, X: np.ndarray, y: np.ndarray) -> float:
        """Bayesian Information Criterion of the fitted joint GMM on ``(X, y)``.

        Parameters
        ----------
        X
            Feature matrix in physical units, shape ``(n_samples, n_features)``.
        y
            Residual target in physical units, shape ``(n_samples,)``.

        Returns
        -------
        float
            BIC score (lower is better).
        """
        assert self.gmm is not None
        joint_data = self._joint_data_scaled(X, y)
        return float(self.gmm.bic(joint_data))

    def _marginal_responsibilities(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute mixture weights ``beta_k(x)`` from ``N(mu_x, Sigma_xx)``.

        Parameters
        ----------
        X_scaled
            Standardized inputs with shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Responsibility matrix with shape ``(n_samples, n_components)``.
        """
        self._check_fitted()
        assert self.gmm is not None
        assert self.n_features_ is not None

        n_samples = X_scaled.shape[0]
        n_components = self.gmm.n_components
        log_resp = np.empty((n_samples, n_components), dtype=float)

        eye = np.eye(self.n_features_, dtype=float)
        for k in range(n_components):
            mu_x = self.gmm.means_[k, : self.n_features_]
            sigma_xx = self.gmm.covariances_[k, : self.n_features_, : self.n_features_]
            sigma_xx = sigma_xx + self.reg_covar * eye
            log_pi = np.log(self.gmm.weights_[k] + LOG_WEIGHT_EPS)
            log_resp[:, k] = log_pi + multivariate_normal.logpdf(
                X_scaled,
                mean=mu_x,
                cov=sigma_xx,
                allow_singular=True,
            )

        log_resp -= np.max(log_resp, axis=1, keepdims=True)
        weights = np.exp(log_resp)
        denom = weights.sum(axis=1, keepdims=True) + RESP_NORM_EPS
        return weights / denom

    def predict_marginal_responsibilities(self, X: np.ndarray) -> np.ndarray:
        """Marginal mixture responsibilities ``beta_k(x)`` for each row of ``X``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_components)``.
        """
        self._check_fitted()
        assert self.scaler_x is not None
        X = _as_2d_X(X)
        assert_feature_matrix_shape(X, n_features=self.n_features_ or N_FEATURES)
        X_scaled = self.scaler_x.transform(X)
        return self._marginal_responsibilities(X_scaled)

    def _predict_residual_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict standardized residuals via GMR conditional expectation."""
        self._check_fitted()
        assert self.gmm is not None
        assert self.n_features_ is not None

        n_samples = X_scaled.shape[0]
        n_components = self.gmm.n_components
        beta = self._marginal_responsibilities(X_scaled)
        mu_cond = np.zeros((n_samples, n_components), dtype=float)

        eye = np.eye(self.n_features_, dtype=float)
        for k in range(n_components):
            mu_joint = self.gmm.means_[k]
            cov_joint = self.gmm.covariances_[k]

            mu_x = mu_joint[: self.n_features_]
            mu_y = float(mu_joint[self.n_features_])
            sigma_xx = cov_joint[: self.n_features_, : self.n_features_] + self.reg_covar * eye
            sigma_yx = cov_joint[self.n_features_ : self.n_features_ + 1, : self.n_features_]

            delta = X_scaled - mu_x.reshape(1, -1)
            solved = _solve_sigma_xx(sigma_xx, delta.T)
            mu_cond[:, k] = mu_y + (sigma_yx @ solved).ravel()

        return (beta * mu_cond).sum(axis=1)

    def predict_residual(self, X: np.ndarray) -> np.ndarray:
        """Predict ``ay_residual`` using Gaussian Mixture Regression.

        Given ``x``, the model returns ``E[y | X = x]`` as a mixture-weighted
        sum of component-wise conditional means.

        Parameters
        ----------
        X
            Feature matrix with shape ``(n_samples, n_features)`` or ``(n_features,)``.

        Returns
        -------
        numpy.ndarray
            Predicted residuals with shape ``(n_samples,)``.
        """
        self._check_fitted()
        assert self.scaler_x is not None
        assert self.scaler_y is not None

        X = _as_2d_X(X)
        assert_feature_matrix_shape(X, n_features=self.n_features_ or N_FEATURES)

        X_scaled = self.scaler_x.transform(X)
        y_scaled = self._predict_residual_scaled(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def predict_residual_smoothed(
        self,
        X: np.ndarray,
        smoothing: ResidualSmoothingConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict residual with temporal smoothing applied to GMR output only.

        Returns
        -------
        residual_raw
            Unsmoothed GMR residual.
        residual_smoothed
            Temporally smoothed residual (used for correction).
        """
        residual_raw = self.predict_residual(X)
        residual_smoothed = smooth_residual_temporal(residual_raw, smoothing)
        return residual_raw, residual_smoothed

    def predict_corrected_ay_smoothed(
        self,
        X: np.ndarray,
        smoothing: ResidualSmoothingConfig,
        *,
        gating: ResidualGatingConfig | None = DEFAULT_RESIDUAL_GATING_CONFIG,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict ``ay_corrected`` using smoothed, gate-weighted residual.

        Returns
        -------
        residual_raw
            Unsmoothed GMR residual.
        residual_smoothed
            Temporally smoothed residual (before gating).
        ay_corrected
            ``ay_bike + w * residual_smoothed``.
        """
        X = _as_2d_X(X)
        assert_feature_matrix_shape(X, n_features=self.n_features_ or N_FEATURES)
        residual_raw, residual_smoothed = self.predict_residual_smoothed(X, smoothing)
        ay_bike = extract_ay_bike_from_X(X, index=(self.n_features_ or X.shape[1]) - 1)
        ay_corrected, _, _ = apply_gated_correction(ay_bike, residual_smoothed, gating)
        return residual_raw, residual_smoothed, ay_corrected

    def predict_corrected_ay(
        self,
        X: np.ndarray,
        *,
        ay_bike_raw: np.ndarray | None = None,
        gating: ResidualGatingConfig | None = DEFAULT_RESIDUAL_GATING_CONFIG,
    ) -> np.ndarray:
        """Predict corrected lateral acceleration.

        ``ay_corrected = ay_bike_raw + w * ay_residual_pred``. When
        ``ay_bike_raw`` is provided it is used for the physics base; otherwise
        the last column of ``X`` is used (may be a smoothed feature copy).

        Parameters
        ----------
        X
            Feature matrix with shape ``(n_samples, n_features)`` or ``(n_features,)``.
        ay_bike_raw
            Optional raw nominal bicycle lateral acceleration for correction.
        gating
            Residual gating config; ``None`` uses :data:`DEFAULT_RESIDUAL_GATING_CONFIG`.

        Returns
        -------
        numpy.ndarray
            Corrected lateral acceleration with shape ``(n_samples,)``.
        """
        X = _as_2d_X(X)
        assert_feature_matrix_shape(X, n_features=self.n_features_ or N_FEATURES)
        idx = (self.n_features_ or X.shape[1]) - 1
        if ay_bike_raw is not None:
            ay_bike = np.asarray(ay_bike_raw, dtype=float).reshape(-1)
            if len(ay_bike) != X.shape[0]:
                raise ValueError("ay_bike_raw must have the same length as X.")
        else:
            ay_bike = extract_ay_bike_from_X(X, index=idx)
        residual_pred = self.predict_residual(X)
        ay_corrected, _, _ = apply_gated_correction(ay_bike, residual_pred, gating)
        return ay_corrected

    def save(self, path: str) -> None:
        """Persist the fitted model and scalers with joblib.

        Parameters
        ----------
        path
            Destination file path (``.joblib`` recommended).
        """
        self._check_fitted()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_class": "GMMResidualModel",
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "n_features_": self.n_features_,
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,
            "gmm": self.gmm,
        }
        joblib.dump(payload, out_path)

    @classmethod
    def load(cls, path: str) -> "GMMResidualModel":
        """Load a model saved by :meth:`save`.

        Parameters
        ----------
        path
            Path to a joblib artifact created by :meth:`save`.

        Returns
        -------
        GMMResidualModel
            Restored fitted model.
        """
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("model_class") != "GMMResidualModel":
            raise ValueError(
                f"File {path!r} is not a valid GMMResidualModel artifact."
            )

        model = cls(
            n_components=int(payload["n_components"]),
            covariance_type=str(payload["covariance_type"]),
            reg_covar=float(payload["reg_covar"]),
            random_state=payload["random_state"],
            max_iter=int(payload["max_iter"]),
            n_init=int(payload["n_init"]),
        )
        model.n_features_ = int(payload["n_features_"])
        model.scaler_x = payload["scaler_x"]
        model.scaler_y = payload["scaler_y"]
        model.gmm = payload["gmm"]
        model._check_fitted()
        return model
