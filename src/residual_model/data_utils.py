"""CSV loading, validation, and construction of residual-learning arrays.

Assumes training/evaluation CSV files provide at least the columns listed in
``residual_model.config.REQUIRED_DATA_COLUMNS``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from residual_model.config import (
    AY_BIKE_COLUMN,
    AY_BIKE_COLUMN_INDEX,
    COLUMNS_BEFORE_STEERING_RATE,
    FEATURE_COLUMNS,
    N_FEATURES,
    REQUIRED_DATA_COLUMNS,
    TARGET_COLUMN,
    feature_columns_label,
)
from residual_model.feature_preprocessing import smooth_ay_bike_for_gmm_features
from residual_model.steering_rate import SteeringRateConfig, add_steering_rate_column


class MissingColumnsError(ValueError):
    """Raised when the input table is missing one or more required columns."""


def validate_required_columns(df: pd.DataFrame, required: tuple[str, ...] | None = None) -> None:
    """Ensure ``df`` contains every column in ``required``.

    Parameters
    ----------
    df
        Input table.
    required
        Column names that must be present. Defaults to ``REQUIRED_DATA_COLUMNS``.

    Raises
    ------
    MissingColumnsError
        If any required column is absent.
    """
    req = tuple(required) if required is not None else REQUIRED_DATA_COLUMNS
    missing = [c for c in req if c not in df.columns]
    if missing:
        have = ", ".join(map(str, sorted(df.columns.astype(str).tolist())))
        need = ", ".join(missing)
        raise MissingColumnsError(
            f"CSV is missing required column(s): {need}. Present columns: {have}"
        )


def assert_feature_matrix_shape(
    X: np.ndarray,
    *,
    n_features: int = N_FEATURES,
    feature_columns: tuple[str, ...] | None = None,
) -> None:
    """Check that ``X`` is 2-D with exactly ``n_features`` columns.

    Parameters
    ----------
    feature_columns
        Names used in error messages; defaults to ``FEATURE_COLUMNS`` when
        ``n_features == N_FEATURES``, otherwise generic labels.

    Raises
    ------
    ValueError
        If the shape is invalid.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2-D, got shape {X.shape}.")
    if X.shape[1] != n_features:
        if feature_columns is not None:
            label = feature_columns_label(feature_columns)
        elif n_features == N_FEATURES:
            label = feature_columns_label(FEATURE_COLUMNS)
        else:
            label = feature_columns_label(tuple(f"z{i}" for i in range(n_features)))
        raise ValueError(
            f"Expected X to have exactly {n_features} columns ({label}); "
            f"got n_features={X.shape[1]}."
        )


def extract_ay_bike_from_X(X: np.ndarray, index: int | None = None) -> np.ndarray:
    """Return the nominal bicycle ``ay_bike`` column from feature matrix ``X``.

  ``ay_bike`` is always the last column in both legacy and extended feature vectors.
    """
    col = (X.shape[1] - 1) if index is None else index
    return np.asarray(X, dtype=float)[:, col]


def load_vehicle_csv(csv_path: str) -> pd.DataFrame:
    """Load a vehicle dynamics CSV into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    csv_path
        Path to the CSV file (relative or absolute).

    Returns
    -------
    pandas.DataFrame
        Raw loaded frame (no cleaning).
    """
    return pd.read_csv(csv_path)


def drop_nonfinite_rows(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Drop rows where any of ``columns`` is NaN or non-finite.

    Parameters
    ----------
    df
        Input table.
    columns
        Columns to check for finite values.

    Returns
    -------
    pandas.DataFrame
        A copy with invalid rows removed.
    """
    work = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    arr = work.to_numpy(dtype=float, copy=False)
    mask = np.isfinite(arr).all(axis=1)
    out = df.loc[mask].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def compute_ay_residual(df: pd.DataFrame) -> pd.Series:
    """Compute ``ay_meas - ay_bike`` as a :class:`pandas.Series`."""
    ay_meas = pd.to_numeric(df["ay_meas"], errors="coerce")
    ay_bike = pd.to_numeric(df["ay_bike"], errors="coerce")
    return ay_meas - ay_bike


def extract_ay_bike_raw_from_df(df: pd.DataFrame) -> np.ndarray:
    """Return raw nominal bicycle ``ay_bike`` from a cleaned frame."""
    return pd.to_numeric(df[AY_BIKE_COLUMN], errors="coerce").to_numpy(dtype=float)


def build_feature_matrix_and_target(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] | None = None,
    *,
    smooth_ay_bike_feature: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``X`` and ``y`` arrays from a cleaned frame.

    ``df`` must already contain finite values for all feature and target columns.

    When ``smooth_ay_bike_feature`` is True, the last ``ay_bike`` column in ``X``
    is lightly Savitzky-Golay smoothed for GMM input only. The residual target
  ``y = ay_meas - ay_bike_raw`` always uses the raw ``ay_bike`` column in ``df``.

    Parameters
    ----------
    feature_columns
        Ordered input names; defaults to ``FEATURE_COLUMNS`` (minimum viable).
    smooth_ay_bike_feature
        If True, replace ``ay_bike`` in ``X`` with a lightly smoothed copy.

    Returns
    -------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix in ``feature_columns`` order.
    y : numpy.ndarray, shape (n_samples,)
        ``ay_residual``
    """
    cols = tuple(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    validate_required_columns(df)
    X = df[list(cols)].to_numpy(dtype=float, copy=True)
    y = df[TARGET_COLUMN].to_numpy(dtype=float, copy=True)

    if smooth_ay_bike_feature and AY_BIKE_COLUMN in cols:
        ay_idx = cols.index(AY_BIKE_COLUMN)
        ay_raw = extract_ay_bike_raw_from_df(df)
        X[:, ay_idx] = smooth_ay_bike_for_gmm_features(ay_raw)

    assert_feature_matrix_shape(X, n_features=len(cols), feature_columns=cols)
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y must be a 1-D array with the same length as X.")
    return X, y


def ensure_steering_rate(
    df: pd.DataFrame,
    *,
    steering_config: SteeringRateConfig | None = None,
) -> pd.DataFrame:
    """Sort by time (if needed) and add ``steering_rate`` when missing."""
    return add_steering_rate_column(
        df,
        config=steering_config,
        sort_by_time=True,
        overwrite=False,
    )


def prepare_residual_dataset(
    csv_path: str,
    *,
    steering_config: SteeringRateConfig | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load CSV, validate columns, drop invalid rows, and build ``X``, ``y``.

    This implements the data assumptions for training:

    - required columns present
    - rows with NaN/inf in any numeric field used for X/y are removed
    - ``ay_residual = ay_meas - ay_bike`` is computed and stored in the frame

    Parameters
    ----------
    csv_path
        Path to the CSV file.

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned table including ``ay_residual``.
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        Residual target.
    """
    df = load_vehicle_csv(csv_path)
    validate_required_columns(df, COLUMNS_BEFORE_STEERING_RATE)
    df = ensure_steering_rate(df, steering_config=steering_config)
    validate_required_columns(df)

    cols_for_finite_check: tuple[str, ...] = tuple(
        c for c in REQUIRED_DATA_COLUMNS if c != "timestamp"
    )
    df = drop_nonfinite_rows(df, cols_for_finite_check)

    df = df.copy()
    df[TARGET_COLUMN] = compute_ay_residual(df)

    X, y = build_feature_matrix_and_target(df)
    return df, X, y


def train_test_split_xy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(X, y)`` into train and test subsets (random shuffle)."""
    if X.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 samples after cleaning to split data; got n={X.shape[0]}."
        )
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def clean_residual_dataframe(
    df: pd.DataFrame,
    *,
    drop_duplicate_timestamps: bool = True,
    steering_config: SteeringRateConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop non-finite rows and optional duplicate timestamps; preserve time order.

    Returns
    -------
    df_clean
        Cleaned frame sorted by ``timestamp``.
    stats
        Counts: ``total``, ``removed_nonfinite``, ``removed_duplicates``, ``final``.
    """
    validate_required_columns(df, COLUMNS_BEFORE_STEERING_RATE)
    df = ensure_steering_rate(df, steering_config=steering_config)
    n_total = len(df)

    cols_numeric = tuple(c for c in REQUIRED_DATA_COLUMNS if c != "timestamp")
    df = drop_nonfinite_rows(df, cols_numeric)
    removed_nonfinite = n_total - len(df)

    if drop_duplicate_timestamps and "timestamp" in df.columns:
        n_before = len(df)
        df = df.sort_values("timestamp", kind="mergesort").drop_duplicates(
            subset=["timestamp"], keep="first"
        )
        removed_duplicates = n_before - len(df)
    else:
        df = df.sort_values("timestamp", kind="mergesort")
        removed_duplicates = 0

    df = df.reset_index(drop=True)
    if TARGET_COLUMN not in df.columns:
        df = df.copy()
        df[TARGET_COLUMN] = compute_ay_residual(df)

    stats = {
        "total": n_total,
        "removed_nonfinite": removed_nonfinite,
        "removed_duplicates": removed_duplicates,
        "final": len(df),
    }
    return df, stats


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-ordered split: first ``(1 - test_fraction)`` train, last fraction test (no shuffle)."""
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples for temporal split; got n={n}.")

    n_test = max(1, int(np.floor(n * test_fraction)))
    n_train = n - n_test
    if n_train < 1:
        raise ValueError(f"Temporal split left no training samples (n={n}, test_fraction={test_fraction}).")

    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def dataframe_to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build ``X``, ``y`` from a cleaned frame that includes ``ay_residual``."""
    X, y = build_feature_matrix_and_target(df)
    return X, y


def fit_standard_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[StandardScaler, StandardScaler]:
    """Fit separate :class:`~sklearn.preprocessing.StandardScaler` instances for X and y."""
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y.fit(y_train.reshape(-1, 1))
    return scaler_x, scaler_y


def transform_xy(
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fitted scalers to ``X`` and ``y``."""
    Xs = scaler_x.transform(X)
    ys = scaler_y.transform(y.reshape(-1, 1)).ravel()
    return Xs, ys


def prepare_scaled_train_test(
    csv_path: str,
    *,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """End-to-end data preparation: load, clean, split, and standardize.

    Standardization is fit **only** on the training split to avoid leakage.

    Returns
    -------
    dict
        Keys include ``df_clean``, ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``X_train_scaled``, ``X_test_scaled``, ``y_train_scaled``, ``y_test_scaled``,
        ``scaler_x``, ``scaler_y``, and ``feature_columns``.
    """
    df_clean, X, y = prepare_residual_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler_x, scaler_y = fit_standard_scalers(X_train, y_train)
    X_train_scaled, y_train_scaled = transform_xy(scaler_x, scaler_y, X_train, y_train)
    X_test_scaled, y_test_scaled = transform_xy(scaler_x, scaler_y, X_test, y_test)

    return {
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled,
        "y_test_scaled": y_test_scaled,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
    }


def resample_arrays_by_weight(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    random_state: int | None = 42,
    target_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample rows proportional to ``weights`` when sklearn GMM lacks ``sample_weight``.

    Parameters
    ----------
    X, y
        Training arrays.
    weights
        Non-negative per-sample weights.
    random_state
        RNG seed.
    target_size
        Output size; defaults to ``len(y)``.

    Returns
    -------
    X_resampled, y_resampled
        Resampled training arrays.
    """
    n = len(y)
    size = n if target_size is None else int(target_size)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if w.sum() <= 0:
        return X, y
    p = w / w.sum()
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=size, replace=True, p=p)
    return X[idx], y[idx]
