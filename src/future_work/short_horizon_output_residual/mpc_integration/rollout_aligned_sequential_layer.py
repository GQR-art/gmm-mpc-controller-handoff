"""Deployment-aligned, control-conditioned GMR prediction layer.

The Dynamic Bicycle model remains the sole state rollout.  GMR only corrects
the reported outputs and is never fed future measured vehicle states.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from future_work.short_horizon_output_residual.config import HORIZON_STEPS, SAMPLE_TIME_S
from future_work.short_horizon_output_residual.data_generation.bicycle_rollout import rollout
from future_work.short_horizon_output_residual.gmm_gmr.gmm_model import JointGMM
from future_work.short_horizon_output_residual.gmm_gmr.gmr_predictor import predict as gmr_predict

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "control_aware_rollout_gmr" / "outputs" / "selected_rollout_aligned_B.joblib"
TRAINING = ROOT / "control_aware_rollout_gmr" / "datasets" / "train_rollout_aligned_long_table.csv"


@dataclass(frozen=True)
class RolloutAlignedPrediction:
    prediction_time_s: np.ndarray
    Y_bicycle: np.ndarray
    mu_deltaY: np.ndarray
    Sigma_deltaY: np.ndarray
    Y_corrected: np.ndarray
    gmr_features: np.ndarray


class RolloutAlignedSequentialLayer:
    """Predict from a candidate control sequence without embedding an MPC solver."""

    def __init__(self, model_path: Path | str | None = None, training_path: Path | str | None = None):
        self.model = JointGMM.load(model_path or MODEL)
        self.training_path = Path(training_path or TRAINING)
        names = ["vx", "r_bicycle", "ay_bicycle", "delta_future", "delta_dot_future", "prediction_time_s"]
        training = pd.read_csv(self.training_path, usecols=names)
        self._training_values = {name: training[name].to_numpy(float) for name in names}
        self._training_bounds = {name: (float(values.min()), float(values.max())) for name, values in self._training_values.items()}

    @staticmethod
    def _rate(controls: np.ndarray, steering_now: float) -> np.ndarray:
        return np.r_[(controls[0] - steering_now) / SAMPLE_TIME_S, np.diff(controls) / SAMPLE_TIME_S]

    def predict(self, state: dict, condition: dict, controls, vx_preview=None, steering_rate_future=None) -> RolloutAlignedPrediction:
        """Evaluate one MPC candidate using only current state, preview, and controls.

        Required state keys are ``vy`` and ``r``.  ``condition`` provides ``vx``
        and current ``steering``; its measured r/ay are intentionally not used.
        """
        u = np.asarray(controls, float).reshape(-1)
        vx = np.full(HORIZON_STEPS, float(condition["vx"])) if vx_preview is None else np.asarray(vx_preview, float).reshape(-1)
        if len(u) != HORIZON_STEPS or len(vx) != HORIZON_STEPS:
            raise ValueError("exactly 20 controls and speed-preview values are required")
        bicycle = rollout(float(state["vy"]), float(state["r"]), vx, u)
        rate = self._rate(u, float(condition["steering"])) if steering_rate_future is None else np.asarray(steering_rate_future, float).reshape(-1)
        if len(rate) != HORIZON_STEPS:
            raise ValueError("steering_rate_future must have 20 values")
        tau = np.arange(1, HORIZON_STEPS + 1, dtype=float) * SAMPLE_TIME_S
        features = np.c_[vx, bicycle.yaw_rate, bicycle.ay, u, rate, tau]
        mu, covariance = self._conditional_prediction(features)
        y_bicycle = np.c_[bicycle.ay, bicycle.yaw_rate]
        # deltaY was trained as Y_real - Y_bicycle, so its mean is additive.
        result = RolloutAlignedPrediction(tau, y_bicycle, mu, covariance, y_bicycle + mu, features)
        if not all(np.isfinite(value).all() for value in (result.Y_bicycle, result.mu_deltaY, result.Sigma_deltaY, result.Y_corrected, result.gmr_features)):
            raise ValueError("non-finite rollout-aligned prediction")
        return result

    def _conditional_prediction(self, features):
        """Default frozen inference; isolated implementations may cache constants."""
        return gmr_predict(self.model, features)

    def audit_candidate_controls(self, controls, steering_now: float) -> dict:
        """Compare one candidate's delta/delta-dot support to train-only support."""
        u = np.asarray(controls, float).reshape(-1)
        if len(u) != HORIZON_STEPS:
            raise ValueError("exactly 20 controls are required")
        candidate = {"delta_future": u, "delta_dot_future": self._rate(u, float(steering_now))}
        result = {"candidate_length": HORIZON_STEPS}
        for name, values in candidate.items():
            reference = self._training_values[name]; lower, upper = self._training_bounds[name]
            below, above = values < lower, values > upper
            result[name] = {"training_min": lower, "training_max": upper,
                            "candidate_min": float(values.min()), "candidate_max": float(values.max()),
                            "wasserstein_distance": float(wasserstein_distance(reference, values)),
                            "ood_ratio": float((below | above).mean())}
        result["input_control_ood_free"] = bool(all(result[name]["ood_ratio"] == 0.0 for name in candidate))
        return result

    def audit_prediction_features(self, prediction: RolloutAlignedPrediction) -> dict:
        """Check all six deployed GMR features against train-only min/max support."""
        names = ["vx", "r_bicycle", "ay_bicycle", "delta_future", "delta_dot_future", "prediction_time_s"]
        report = {}
        for index, name in enumerate(names):
            values = prediction.gmr_features[:, index]
            lower, upper = self._training_bounds[name]
            ood = (values < lower) | (values > upper)
            report[name] = {"training_min": lower, "training_max": upper,
                            "candidate_min": float(values.min()), "candidate_max": float(values.max()),
                            "ood_ratio": float(ood.mean())}
        report["input_ood_free"] = bool(all(report[name]["ood_ratio"] == 0.0 for name in names))
        return report

    def iterate(self, state: dict, condition: dict, initial_controls, control_update, max_iterations: int = 3, threshold: float = 1e-4):
        """Generic external-MPC callback interface; no optimizer is implemented here."""
        controls = np.asarray(initial_controls, float).reshape(-1)
        history = []
        for _ in range(max_iterations):
            prediction = self.predict(state, condition, controls)
            updated = np.asarray(control_update(controls.copy(), prediction), float).reshape(-1)
            if len(updated) != HORIZON_STEPS:
                raise ValueError("control_update must return 20 values")
            change = float(np.linalg.norm(updated - controls))
            history.append({"controls": controls.copy(), "prediction": prediction, "control_change_norm": change})
            controls = updated
            if change < threshold:
                break
        return history
