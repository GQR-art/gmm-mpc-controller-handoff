"""Minimum viable residual learning for lateral dynamics (GMM/GMR).

Research-friendly layout: configure features in :mod:`residual_model.config`,
prepare data with :mod:`residual_model.data_utils`, fit/predict with
:class:`~residual_model.gmm_residual.GMMResidualModel`, and use CLI modules
for batch training and evaluation.

Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib, joblib, pytest.
"""

from __future__ import annotations

from residual_model import config, data_utils
from residual_model.config import (
    EXTENDED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LEGACY_FEATURE_COLUMNS,
    N_FEATURES,
    N_LEGACY_FEATURES,
    REQUIRED_DATA_COLUMNS,
)
from residual_model.steering_rate import SteeringRateConfig, SmoothingMethod
from residual_model.gmm_residual import GMMResidualModel

__all__ = [
    "GMMResidualModel",
    "config",
    "data_utils",
    "FEATURE_COLUMNS",
    "LEGACY_FEATURE_COLUMNS",
    "EXTENDED_FEATURE_COLUMNS",
    "N_FEATURES",
    "N_LEGACY_FEATURES",
    "REQUIRED_DATA_COLUMNS",
    "SteeringRateConfig",
    "SmoothingMethod",
]
