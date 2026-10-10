"""Configuration defaults for the lateral-acceleration residual GMM/GMR pipeline.

Centralizes column names and feature ordering so experiments stay reproducible and
easy to extend. The model uses ``FEATURE_COLUMNS`` (6 inputs including
``steering_rate``). Use ``LEGACY_FEATURE_COLUMNS`` for the previous 5-input MVP.

Stack: numpy, pandas, scikit-learn, scipy, matplotlib, joblib (see ``requirements.txt``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

REQUIRED_DATA_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "vx",
    "steering_angle",
    "steering_rate",
    "yaw_rate",
    "longitudinal_acceleration",
    "ay_bike",
    "ay_meas",
)

COLUMNS_BEFORE_STEERING_RATE: tuple[str, ...] = tuple(
    c for c in REQUIRED_DATA_COLUMNS if c != "steering_rate"
)

# Previous MVP (5 inputs) — used for A/B comparison experiments.
LEGACY_FEATURE_COLUMNS: tuple[str, ...] = (
    "vx",
    "steering_angle",
    "yaw_rate",
    "longitudinal_acceleration",
    "ay_bike",
)

# Pure GMM baseline (no ay_bike): z = [vx, delta, delta_dot, r, ax] -> ay_meas
PURE_GMM_FEATURE_COLUMNS: tuple[str, ...] = (
    "vx",
    "steering_angle",
    "steering_rate",
    "yaw_rate",
    "longitudinal_acceleration",
)

N_PURE_GMM_FEATURES: int = len(PURE_GMM_FEATURE_COLUMNS)

# z = [vx, steering_angle, steering_rate, yaw_rate, longitudinal_acceleration, ay_bike]
FEATURE_COLUMNS: tuple[str, ...] = (
    "vx",
    "steering_angle",
    "steering_rate",
    "yaw_rate",
    "longitudinal_acceleration",
    "ay_bike",
)

TARGET_COLUMN: str = "ay_residual"

N_FEATURES: int = len(FEATURE_COLUMNS)
N_LEGACY_FEATURES: int = len(LEGACY_FEATURE_COLUMNS)

AY_BIKE_COLUMN: str = "ay_bike"
AY_BIKE_COLUMN_INDEX: int = FEATURE_COLUMNS.index(AY_BIKE_COLUMN)
LEGACY_AY_BIKE_COLUMN_INDEX: int = LEGACY_FEATURE_COLUMNS.index(AY_BIKE_COLUMN)

# ---------------------------------------------------------------------------
# Future extension (research roadmap — not used by the MVP pipeline)
# ---------------------------------------------------------------------------

EXTENDED_FEATURE_COLUMNS: tuple[str, ...] = (
    "vx",
    "beta",
    "yaw_rate",
    "steering_angle",
    "steering_rate",
    "longitudinal_acceleration",
    "ay_bike",
)

# Training / evaluation CLI defaults (relative paths only in docs and examples).
DEFAULT_MIN_COMPONENTS: int = 1
DEFAULT_MAX_COMPONENTS: int = 8
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42

# Residual gating (inference): suppress correction in low-|ay_bike| regimes.
DEFAULT_RESIDUAL_GATING_AY_THRESHOLD_MPS2: float = 2.75
DEFAULT_RESIDUAL_GATING_K: float = 2.5

# Feature-space noise reduction (GMM input only; raw ay_bike preserved for residual target).
DEFAULT_STEERING_SAVGOL_WINDOW: int = 9
DEFAULT_STEERING_SAVGOL_POLYORDER: int = 2
AY_BIKE_FEATURE_SMOOTH_WINDOW: int = 7
AY_BIKE_FEATURE_SMOOTH_POLYORDER: int = 2

# Mild-regime GMM training regularization (|ay_bike_raw| below threshold).
MILD_REGIME_AY_BIKE_THRESHOLD_MPS2: float = 2.0
MILD_REGIME_SAMPLE_WEIGHT: float = 0.4


def feature_columns_label(columns: tuple[str, ...] | None = None) -> str:
    """Comma-separated feature names for error messages and logging."""
    return ", ".join(columns if columns is not None else FEATURE_COLUMNS)
