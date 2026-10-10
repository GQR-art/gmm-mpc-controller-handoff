"""Feature-space noise reduction and training-distribution regularization.

Provides light Savitzky-Golay smoothing for GMM input features (not residual
targets) and mild-regime sample weighting during GMM fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from residual_model.config import (
    AY_BIKE_COLUMN,
    AY_BIKE_FEATURE_SMOOTH_POLYORDER,
    AY_BIKE_FEATURE_SMOOTH_WINDOW,
    MILD_REGIME_AY_BIKE_THRESHOLD_MPS2,
    MILD_REGIME_SAMPLE_WEIGHT,
)


def _ensure_odd_window(window: int, n: int, polyorder: int) -> int:
    """Return an odd Savitzky-Golay window length in ``[polyorder+2, n]``."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(w, polyorder + 2)
    if n < w:
        max_odd = n if n % 2 == 1 else max(n - 1, 1)
        return max(1, max_odd)
    return min(w, n if n % 2 == 1 else n - 1)


def smooth_ay_bike_for_gmm_features(
    ay_bike_raw: np.ndarray,
    *,
    window_length: int = AY_BIKE_FEATURE_SMOOTH_WINDOW,
    polyorder: int = AY_BIKE_FEATURE_SMOOTH_POLYORDER,
) -> np.ndarray:
    """Lightly smooth ``ay_bike`` for GMM feature input only.

    The raw ``ay_bike`` used in residual definition and physics correction
    is **not** modified by this function.

    Parameters
    ----------
    ay_bike_raw
        Nominal bicycle lateral acceleration (m/s²).
    window_length
        Savitzky-Golay window length (odd).
    polyorder
        Savitzky-Golay polynomial order.

    Returns
    -------
    numpy.ndarray
        Smoothed copy suitable as the last GMM feature column.
    """
    x = np.asarray(ay_bike_raw, dtype=float).copy()
    n = len(x)
    if n < polyorder + 2:
        return x
    w = _ensure_odd_window(window_length, n, polyorder)
    if n < w:
        return x
    return signal.savgol_filter(x, w, polyorder, mode="interp")


@dataclass(frozen=True)
class MildRegimeConfig:
    """Downweight mild-regime samples during GMM fitting."""

    ay_bike_threshold_mps2: float = MILD_REGIME_AY_BIKE_THRESHOLD_MPS2
    mild_sample_weight: float = MILD_REGIME_SAMPLE_WEIGHT

    def __post_init__(self) -> None:
        if self.ay_bike_threshold_mps2 <= 0:
            raise ValueError("ay_bike_threshold_mps2 must be positive.")
        if not (0.0 < self.mild_sample_weight <= 1.0):
            raise ValueError("mild_sample_weight must be in (0, 1].")


def compute_mild_regime_sample_weights(
    ay_bike_raw: np.ndarray,
    *,
    config: MildRegimeConfig | None = None,
) -> np.ndarray:
    """Return per-sample GMM fit weights with reduced mild-regime influence.

    Mild regime: ``|ay_bike_raw| < threshold`` receives ``mild_sample_weight``.
    All other samples receive weight 1.0.

    Parameters
    ----------
    ay_bike_raw
        Raw nominal bicycle lateral acceleration (m/s²).

    Returns
    -------
    numpy.ndarray
        Sample weights with shape ``(n_samples,)``.
    """
    cfg = config or MildRegimeConfig()
    ay = np.asarray(ay_bike_raw, dtype=float)
    mild = np.abs(ay) < cfg.ay_bike_threshold_mps2
    weights = np.ones(len(ay), dtype=float)
    weights[mild] = cfg.mild_sample_weight
    return weights


def select_gmm_fit_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ay_bike_raw: np.ndarray,
    *,
    config: MildRegimeConfig | None = None,
    keep_fraction: float = MILD_REGIME_SAMPLE_WEIGHT,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample mild-regime training rows for GMM fitting.

    Returns
    -------
    X_fit, y_fit, fit_indices
        Subset used for ``GMMResidualModel.fit`` (aggressive rows kept; mild
        rows randomly subsampled).
    """
    idx = downsample_mild_regime_indices(
        ay_bike_raw,
        config=config,
        keep_fraction=keep_fraction,
        random_state=random_state,
    )
    return X_train[idx], y_train[idx], idx


def downsample_mild_regime_indices(
    ay_bike_raw: np.ndarray,
    *,
    config: MildRegimeConfig | None = None,
    keep_fraction: float = 0.4,
    random_state: int = 42,
) -> np.ndarray:
    """Randomly subsample mild-regime rows for GMM training.

    Parameters
    ----------
    ay_bike_raw
        Raw nominal bicycle lateral acceleration (m/s²).
    keep_fraction
        Fraction of mild-regime samples to retain (0–1).
    random_state
        RNG seed for reproducible subsampling.

    Returns
    -------
    numpy.ndarray
        Integer indices to keep (sorted ascending).
    """
    cfg = config or MildRegimeConfig()
    ay = np.asarray(ay_bike_raw, dtype=float)
    mild = np.abs(ay) < cfg.ay_bike_threshold_mps2
    aggressive = ~mild
    rng = np.random.default_rng(random_state)
    mild_idx = np.flatnonzero(mild)
    n_keep = max(1, int(np.round(len(mild_idx) * keep_fraction)))
    if len(mild_idx) > 0:
        kept_mild = rng.choice(mild_idx, size=min(n_keep, len(mild_idx)), replace=False)
    else:
        kept_mild = np.array([], dtype=int)
    kept = np.sort(np.concatenate([np.flatnonzero(aggressive), kept_mild]))
    return kept


def mild_regime_stats(
    ay_bike_raw: np.ndarray,
    *,
    config: MildRegimeConfig | None = None,
) -> dict[str, int | float]:
    """Summarize mild vs aggressive sample counts for logging."""
    cfg = config or MildRegimeConfig()
    ay = np.asarray(ay_bike_raw, dtype=float)
    mild = np.abs(ay) < cfg.ay_bike_threshold_mps2
    n_mild = int(mild.sum())
    n_agg = int((~mild).sum())
    return {
        "n_total": len(ay),
        "n_mild": n_mild,
        "n_aggressive": n_agg,
        "mild_fraction": float(n_mild / len(ay)) if len(ay) else 0.0,
        "threshold_mps2": cfg.ay_bike_threshold_mps2,
        "mild_weight": cfg.mild_sample_weight,
    }


def apply_ay_bike_feature_smoothing_to_frame(
    df: pd.DataFrame,
    *,
    window_length: int = AY_BIKE_FEATURE_SMOOTH_WINDOW,
    polyorder: int = AY_BIKE_FEATURE_SMOOTH_POLYORDER,
) -> pd.Series:
    """Return smoothed ``ay_bike`` series for GMM features; ``df['ay_bike']`` stays raw."""
    raw = pd.to_numeric(df[AY_BIKE_COLUMN], errors="coerce").to_numpy(dtype=float)
    smoothed = smooth_ay_bike_for_gmm_features(
        raw, window_length=window_length, polyorder=polyorder
    )
    return pd.Series(smoothed, index=df.index, name="ay_bike_feature")
