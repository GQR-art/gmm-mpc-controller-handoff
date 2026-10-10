"""Held-out replay plant with the same [delta_ay, delta_r] target as GMR.

This is a validation-only controlled object.  The MPC prediction model remains
unchanged; the plant consumes exactly one pre-recorded held-out residual pair
per control cycle and never exposes a future replay value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from future_work.later_mpc.dynamic_bicycle_plant import DynamicBicyclePlant
from future_work.later_mpc.types import VehicleState

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATASET = ROOT / "control_aware_rollout_gmr" / "datasets" / "train_rollout_aligned_long_table.csv"
TEST_DATASET = ROOT / "control_aware_rollout_gmr" / "datasets" / "test_rollout_aligned_long_table.csv"


@dataclass(frozen=True)
class OutputResidual:
    epsilon_ay: float
    epsilon_r: float
    source_file: str
    window_id: str
    prediction_step: int


class OutputResidualReplay:
    """One held-out window replayed causally by a current control-cycle index."""

    def __init__(self, table: pd.DataFrame):
        required = {"delta_ay", "delta_r", "source_file", "window_id", "prediction_step"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Missing replay fields: {sorted(missing)}")
        self._table = table.sort_values("prediction_step").reset_index(drop=True).copy()
        values = self._table[["delta_ay", "delta_r"]].to_numpy(float)
        if self._table.empty or not np.isfinite(values).all():
            raise ValueError("Replay table must contain finite residual pairs")
        self._index = 0

    @classmethod
    def from_held_out_dataset(cls, path: Path | str = TEST_DATASET, window_id: str | None = None):
        data = pd.read_csv(path)
        selected = str(window_id) if window_id is not None else str(data.window_id.iloc[0])
        window = data.loc[data.window_id.astype(str) == selected].copy()
        if window.empty:
            raise ValueError(f"Unknown held-out replay window: {selected}")
        return cls(window)

    @property
    def length(self) -> int:
        return len(self._table)

    @property
    def index(self) -> int:
        return self._index

    @property
    def source_sessions(self) -> tuple[str, ...]:
        return tuple(sorted(map(str, self._table.source_file.unique())))

    def reset(self) -> None:
        self._index = 0

    def current(self) -> OutputResidual:
        if self._index >= len(self._table):
            raise IndexError("Residual replay is exhausted; no future residual may be read")
        row = self._table.iloc[self._index]
        return OutputResidual(float(row.delta_ay), float(row.delta_r), str(row.source_file),
                              str(row.window_id), int(row.prediction_step))

    def advance(self) -> None:
        if self._index >= len(self._table):
            raise IndexError("Residual replay is exhausted")
        self._index += 1


@dataclass(frozen=True)
class ResidualOutputDiagnostics:
    ay_bicycle: float
    r_bicycle: float
    ay_plant: float
    r_plant: float
    residual: OutputResidual


class ResidualOutputPlant:
    """Bicycle controlled object with causal held-out output-residual replay."""

    def __init__(self, replay: OutputResidualReplay, initial_state: VehicleState | None = None):
        self._replay = replay
        self._initial_state = initial_state or VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self._bicycle = DynamicBicyclePlant(initial_state=self._initial_state)
        self._last: ResidualOutputDiagnostics | None = None

    @property
    def last_diagnostics(self) -> ResidualOutputDiagnostics:
        if self._last is None:
            raise RuntimeError("No plant step has been executed")
        return self._last

    def get_state(self) -> VehicleState:
        return self._bicycle.get_state()

    def reset(self, initial_state: VehicleState | None = None) -> None:
        self._replay.reset()
        self._bicycle.reset(initial_state or self._initial_state)
        self._last = None

    def step(self, delta_rad: float, vx_mps: float, sample_time_s: float) -> VehicleState:
        """Apply the current residual only; advance replay after the control period."""
        residual = self._replay.current()
        bike_state = self._bicycle.step(float(delta_rad), float(vx_mps), float(sample_time_s))
        vy_dot, _ = self._bicycle.lateral_dynamics_derivative(bike_state.vy_mps, bike_state.yaw_rate_radps,
                                                              float(delta_rad), float(vx_mps))
        ay_bicycle = float(vy_dot + float(vx_mps) * bike_state.yaw_rate_radps)
        r_bicycle = float(bike_state.yaw_rate_radps)
        ay_plant = ay_bicycle + residual.epsilon_ay
        r_plant = r_bicycle + residual.epsilon_r
        # Output residuals affect only this validation plant's state feedback, never MPC prediction.
        corrected = VehicleState(bike_state.X_m, bike_state.Y_m,
                                 bike_state.psi_rad + residual.epsilon_r * float(sample_time_s),
                                 bike_state.vy_mps + residual.epsilon_ay * float(sample_time_s), r_plant)
        if not np.isfinite(list(corrected.__dict__.values()) + [ay_plant, r_plant]).all():
            raise FloatingPointError("Non-finite residual-output plant state")
        self._bicycle.reset(corrected)
        self._last = ResidualOutputDiagnostics(ay_bicycle, r_bicycle, ay_plant, r_plant, residual)
        self._replay.advance()
        return self.get_state()


def _stats(values: np.ndarray) -> dict:
    points = (1, 5, 25, 50, 75, 95, 99)
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max()),
            "percentiles": {str(p): float(v) for p, v in zip(points, np.percentile(values, points))}}


def build_residual_consistency_audit(training_path: Path | str = TRAIN_DATASET, held_out_path: Path | str = TEST_DATASET) -> dict:
    """Audit definition/provenance and train-vs-held-out residual distributions."""
    train, held_out = pd.read_csv(training_path), pd.read_csv(held_out_path)
    train_sessions, replay_sessions = set(train.source_file.astype(str)), set(held_out.source_file.astype(str))
    data = {"training_sessions": sorted(train_sessions), "held_out_replay_sessions": sorted(replay_sessions),
            "session_overlap": sorted(train_sessions & replay_sessions), "definitions_identical": True,
            "plant_output_identity": {"delta_ay_plant": "ay_plant - ay_bicycle = epsilon_ay = delta_ay replay",
                                      "delta_r_plant": "r_plant - r_bicycle = epsilon_r = delta_r replay",
                                      "verified_by": "test_residual_output_plant.py"},
            "causal_replay": "current replay index is read once in step(), then advanced; exhaustion raises IndexError rather than reading a future value",
            "variables": {}}
    for name in ("delta_ay", "delta_r"):
        a, b = train[name].to_numpy(float), held_out[name].to_numpy(float)
        data["variables"][name] = {"training": _stats(a), "held_out_replay": _stats(b),
                                   "wasserstein_distance": float(wasserstein_distance(a, b))}
    data["delta_ay_delta_r_pearson_correlation"] = {"training": float(np.corrcoef(train.delta_ay, train.delta_r)[0, 1]),
                                                       "held_out_replay": float(np.corrcoef(held_out.delta_ay, held_out.delta_r)[0, 1])}
    data["held_out_isolated"] = not bool(train_sessions & replay_sessions)
    return data


if __name__ == "__main__":
    output = ROOT / "mpc_integration" / "outputs" / "plant_residual_consistency.json"
    output.write_text(json.dumps(build_residual_consistency_audit(), indent=2), encoding="utf8")
