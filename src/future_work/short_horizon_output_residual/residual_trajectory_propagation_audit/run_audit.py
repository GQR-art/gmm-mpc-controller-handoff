"""Residual-to-Trajectory Propagation Feasibility Audit.

The module is diagnostic only.  It reconstructs previously recorded nominal
candidate sequences and closed-loop states, then evaluates three state-output
interpretations with the frozen Q/R/P objective.  It never solves or modifies
the MPC, and writes only below this audit directory.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from future_work.later_mpc.config import MPCConfig
from future_work.later_mpc.error_state import global_state_to_error_state
from future_work.later_mpc.nominal_bicycle_predictor import NominalBicyclePredictor
from future_work.later_mpc.scenarios import (
    constant_radius_turn,
    double_lane_change,
    straight_offset_recovery,
)
from future_work.short_horizon_output_residual.mpc_integration.residual_output_plant import (
    OutputResidualReplay,
    ResidualOutputPlant,
)
from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import (
    RolloutAlignedSequentialLayer,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = Path(__file__).resolve().parent
OUT = AUDIT_ROOT / "outputs"
FIGURES = OUT / "figures"
MODEL_PATH = ROOT / "control_aware_rollout_gmr" / "outputs" / "selected_rollout_aligned_B.joblib"
TRAIN_PATH = ROOT / "control_aware_rollout_gmr" / "datasets" / "train_rollout_aligned_long_table.csv"
TEST_PATH = ROOT / "control_aware_rollout_gmr" / "datasets" / "test_rollout_aligned_long_table.csv"
CLOSED_LOOP = ROOT / "mpc_integration" / "outputs" / "residual_output_plant_closed_loop"
SIGNALS_PATH = CLOSED_LOOP / "closed_loop_signals.csv"
TRACE_PATH = CLOSED_LOOP / "sequential_iteration_trace.csv"
COST_AUDIT_SUMMARY = ROOT / "mpc_integration" / "outputs" / "gmr_cost_sensitivity_audit" / "gmr_cost_sensitivity_summary.json"

SAMPLE_TIME_S = 0.05
HORIZON = 20
SPEED_MPS = 15.0
RANDOM_SEED = 20260826
SAMPLES_PER_SCENARIO = 12
CANDIDATES_PER_SAMPLE = 25
FEATURES = ["vx", "r_bicycle", "ay_bicycle", "delta_future", "delta_dot_future", "prediction_time_s"]
SCENARIOS = {
    "s1_straight_offset_recovery": straight_offset_recovery(),
    "s2_constant_radius_turn": constant_radius_turn(),
    "s3_double_lane_change": double_lane_change(),
}
REPLAY_SOURCE = {
    "s1_straight_offset_recovery": ("2023-11_5.csv", "Normal"),
    "s2_constant_radius_turn": ("2023-11_6.csv", "High Dynamic"),
    "s3_double_lane_change": ("2023-11_7.csv", "Aggressive"),
}


@dataclass(frozen=True)
class CostBreakdown:
    total: float
    stage_tracking: float
    terminal: float
    steering: float
    steering_rate: float
    ey_component: float
    epsi_component: float
    vy_component: float
    r_component: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    for name in ("r_bicycle", "ay_bicycle"):
        duplicate = f"{name}.1"
        if duplicate in table:
            if not np.allclose(table[name], table[duplicate], atol=1e-12, rtol=0.0):
                raise AssertionError(f"Historical duplicate columns disagree: {name}")
            table = table.drop(columns=duplicate)
    return table


def _replay_stream(test: pd.DataFrame, source_file: str, old_regime: str, count: int) -> pd.DataFrame:
    data = test.loc[(test.source_file.astype(str) == source_file) & (test.dynamic_regime == old_regime)].copy()
    data["target_time"] = np.round(data.timestamp + data.prediction_time_s, 9)
    data = data.sort_values(["target_time", "prediction_step"]).drop_duplicates("target_time", keep="first")
    stream = data.iloc[:count].copy().reset_index(drop=True)
    if len(stream) != count:
        raise AssertionError(f"Insufficient replay rows for {source_file}/{old_regime}")
    stream["prediction_step"] = np.arange(1, count + 1)
    stream["window_id"] = f"propagation-audit::{source_file}::{old_regime}"
    return stream


def project_controls(raw: np.ndarray, previous_delta: float, config: MPCConfig) -> np.ndarray:
    """Project one candidate onto the frozen amplitude and per-sample rate bounds."""
    bounds = config.steering_constraints
    result = np.empty_like(np.asarray(raw, float))
    prior = float(previous_delta)
    for index, value in enumerate(np.asarray(raw, float)):
        lower = max(bounds.delta_min_rad, prior + bounds.delta_rate_min_rad_per_sample)
        upper = min(bounds.delta_max_rad, prior + bounds.delta_rate_max_rad_per_sample)
        result[index] = np.clip(value, lower, upper)
        prior = result[index]
    return result


def candidate_set(nominal: np.ndarray, previous_delta: float, config: MPCConfig, rng) -> np.ndarray:
    """Create a deterministic feasible local cloud, larger than prior cost audits."""
    nominal = np.asarray(nominal, float)
    phase = np.linspace(0.0, 1.0, len(nominal))
    bases = np.vstack(
        [np.ones(len(nominal)), phase, np.sin(np.pi * phase), np.sin(2.0 * np.pi * phase), np.cos(np.pi * phase)]
    )
    candidates = [nominal.copy()]
    attempts = 0
    while len(candidates) < CANDIDATES_PER_SAMPLE:
        attempts += 1
        if attempts > 5000:
            raise RuntimeError("Could not form enough distinct feasible candidates")
        perturbation = rng.normal(0.0, 0.006, size=len(bases)) @ bases
        proposed = project_controls(nominal + perturbation, previous_delta, config)
        if all(np.linalg.norm(proposed - existing) > 1e-7 for existing in candidates):
            candidates.append(proposed)
    return np.asarray(candidates)


def pairwise_flip_fraction(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, float), np.asarray(second, float)
    flips = 0
    comparable = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            sign_first = np.sign(first[left] - first[right])
            sign_second = np.sign(second[left] - second[right])
            if sign_first == 0 or sign_second == 0:
                continue
            comparable += 1
            flips += int(sign_first != sign_second)
    return float(flips / comparable) if comparable else 0.0


def propagate_delta_r(
    nominal_states_with_initial: np.ndarray,
    delta_r: np.ndarray,
    vx_mps: float = SPEED_MPS,
    sample_time_s: float = SAMPLE_TIME_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate additive yaw-rate residual through epsi and exact ey kinematics.

    The nominal MPC conventions are ``ey > 0`` to the path left and
    ``epsi = psi - psi_ref``.  The incremental lateral-error velocity is

      vx[sin(epsi+dpsi)-sin(epsi)] + vy[cos(epsi+dpsi)-cos(epsi)].

    Trapezoidal integration guarantees a zero correction for zero ``delta_r``
    without replacing the nominal MPC discretization.
    """
    states = np.asarray(nominal_states_with_initial, float)
    residual = np.asarray(delta_r, float).reshape(-1)
    if states.shape != (len(residual) + 1, 4):
        raise ValueError("Expected nominal states with x0 and one state per residual step")
    delta_epsi = np.cumsum(residual) * float(sample_time_s)
    delta_epsi_with_initial = np.r_[0.0, delta_epsi]
    delta_ey = np.empty(len(residual), float)
    accumulated = 0.0
    for index in range(len(residual)):
        velocity_difference = []
        for state_index in (index, index + 1):
            epsi = states[state_index, 1]
            vy = states[state_index, 2]
            correction = delta_epsi_with_initial[state_index]
            velocity_difference.append(
                vx_mps * (np.sin(epsi + correction) - np.sin(epsi))
                + vy * (np.cos(epsi + correction) - np.cos(epsi))
            )
        accumulated += 0.5 * sample_time_s * (velocity_difference[0] + velocity_difference[1])
        delta_ey[index] = accumulated
    return delta_epsi, delta_ey


def corrected_states(nominal_states_with_initial: np.ndarray, delta_r: np.ndarray, include_ey: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_epsi, delta_ey = propagate_delta_r(nominal_states_with_initial, delta_r)
    states = np.asarray(nominal_states_with_initial[1:], float).copy()
    states[:, 1] += delta_epsi
    if include_ey:
        states[:, 0] += delta_ey
    return states, delta_epsi, delta_ey


def cost_breakdown(states: np.ndarray, controls: np.ndarray, previous_delta: float, config: MPCConfig) -> CostBreakdown:
    states = np.asarray(states, float)
    controls = np.asarray(controls, float)
    weights = config.state_weights
    component_weights = np.array([weights.ey, weights.epsi, weights.vy, weights.r], float)
    component = np.sum(states[:-1] ** 2 * component_weights[None, :], axis=0)
    terminal_components = weights.terminal_scale * component_weights * states[-1] ** 2
    stage = float(np.sum(component))
    terminal = float(np.sum(terminal_components))
    steering = float(config.input_weights.delta * controls @ controls)
    increments = np.diff(np.r_[float(previous_delta), controls])
    steering_rate = float(config.input_weights.delta_rate * increments @ increments)
    all_components = component + terminal_components
    total = stage + terminal + steering + steering_rate
    return CostBreakdown(total, stage, terminal, steering, steering_rate, *map(float, all_components))


def _matched_oracle_window(
    nominal_features: np.ndarray,
    source_file: str,
    old_regime: str,
    test_windows: dict,
    feature_scale: np.ndarray,
) -> tuple[str, np.ndarray, float]:
    candidates = test_windows.get((source_file, old_regime), [])
    if not candidates:
        candidates = [item for (source, _), groups in test_windows.items() if source == source_file for item in groups]
    if not candidates:
        raise AssertionError(f"No held-out oracle windows for {source_file}")
    best = None
    for window_id, features, true_delta_r in candidates:
        distance = float(np.sqrt(np.mean(((features - nominal_features) / feature_scale[None, :]) ** 2)))
        if best is None or distance < best[2]:
            best = (window_id, true_delta_r, distance)
    assert best is not None
    return best


def _oracle_windows(test: pd.DataFrame) -> dict:
    result: dict[tuple[str, str], list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for window_id, group in test.groupby("window_id", sort=False):
        group = group.sort_values("prediction_step")
        if len(group) != HORIZON:
            continue
        key = (str(group.source_file.iloc[0]), str(group.dynamic_regime.iloc[0]))
        result.setdefault(key, []).append(
            (str(window_id), group[FEATURES].to_numpy(float), group.delta_r.to_numpy(float))
        )
    return result


def _selected_steps(count: int) -> set[int]:
    return set(map(int, np.unique(np.linspace(5, count - 6, SAMPLES_PER_SCENARIO, dtype=int))))


def _prefix_cost(row: dict, prefix: str, cost: CostBreakdown) -> None:
    for field in CostBreakdown.__dataclass_fields__:
        row[f"{prefix}_{field}"] = getattr(cost, field)


def evaluate_candidate(
    layer: RolloutAlignedSequentialLayer,
    predictor: NominalBicyclePredictor,
    state,
    error_state: np.ndarray,
    curvature: np.ndarray,
    controls: np.ndarray,
    previous_delta: float,
    oracle_delta_r: np.ndarray,
    config: MPCConfig,
) -> tuple[dict, list[dict]]:
    nominal_with_initial = predictor.predict(
        error_state, controls, np.asarray(curvature, float)[:HORIZON], SPEED_MPS, SAMPLE_TIME_S
    )
    prediction = layer.predict(
        {"vy": state.vy_mps, "r": state.yaw_rate_radps},
        {"vx": SPEED_MPS, "steering": previous_delta},
        controls,
    )
    predicted_delta_r = prediction.mu_deltaY[:, 1]
    nominal_states = nominal_with_initial[1:].copy()
    current_states, predicted_delta_epsi, predicted_delta_ey = corrected_states(
        nominal_with_initial, predicted_delta_r, include_ey=False
    )
    full_states, _, full_delta_ey = corrected_states(nominal_with_initial, predicted_delta_r, include_ey=True)
    oracle_states, oracle_delta_epsi, oracle_delta_ey = corrected_states(
        nominal_with_initial, oracle_delta_r, include_ey=True
    )
    costs = {
        "nominal": cost_breakdown(nominal_states, controls, previous_delta, config),
        "current": cost_breakdown(current_states, controls, previous_delta, config),
        "full": cost_breakdown(full_states, controls, previous_delta, config),
        "oracle": cost_breakdown(oracle_states, controls, previous_delta, config),
    }
    row = {
        "control_l2": float(np.linalg.norm(controls)),
        "max_abs_predicted_delta_r": float(np.max(np.abs(predicted_delta_r))),
        "max_abs_predicted_delta_epsi": float(np.max(np.abs(predicted_delta_epsi))),
        "max_abs_predicted_delta_ey": float(np.max(np.abs(full_delta_ey))),
        "max_abs_oracle_delta_r": float(np.max(np.abs(oracle_delta_r))),
        "max_abs_oracle_delta_epsi": float(np.max(np.abs(oracle_delta_epsi))),
        "max_abs_oracle_delta_ey": float(np.max(np.abs(oracle_delta_ey))),
    }
    for name, cost in costs.items():
        _prefix_cost(row, name, cost)
    row.update(
        {
            "current_minus_nominal": costs["current"].total - costs["nominal"].total,
            "full_minus_nominal": costs["full"].total - costs["nominal"].total,
            "full_minus_current": costs["full"].total - costs["current"].total,
            "oracle_minus_nominal": costs["oracle"].total - costs["nominal"].total,
            "oracle_minus_full": costs["oracle"].total - costs["full"].total,
            "full_current_stage_tracking_change": costs["full"].stage_tracking - costs["current"].stage_tracking,
            "full_current_terminal_change": costs["full"].terminal - costs["current"].terminal,
        }
    )
    horizon_rows = []
    for index in range(HORIZON):
        horizon_rows.append(
            {
                "prediction_step": index + 1,
                "prediction_time_s": (index + 1) * SAMPLE_TIME_S,
                "predicted_delta_r": predicted_delta_r[index],
                "predicted_delta_epsi": predicted_delta_epsi[index],
                "predicted_delta_ey": full_delta_ey[index],
                "oracle_delta_r": oracle_delta_r[index],
                "oracle_delta_epsi": oracle_delta_epsi[index],
                "oracle_delta_ey": oracle_delta_ey[index],
            }
        )
    return row, horizon_rows


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    config = MPCConfig()
    if config.horizon_steps != HORIZON or config.sample_time_s != SAMPLE_TIME_S or config.longitudinal_speed_mps != SPEED_MPS:
        raise AssertionError("Audit constants must match frozen MPCConfig")
    test = _read_table(TEST_PATH)
    signals = pd.read_csv(SIGNALS_PATH)
    trace = pd.read_csv(TRACE_PATH)
    oracle_windows = _oracle_windows(test)
    layer = RolloutAlignedSequentialLayer(model_path=MODEL_PATH, training_path=TRAIN_PATH)
    predictor = NominalBicyclePredictor()
    feature_scale = layer.model.scaler.scale_[: layer.model.input_dim]
    rng = np.random.default_rng(RANDOM_SEED)
    candidate_rows, ranking_rows, oracle_rows, horizon_rows = [], [], [], []
    max_reconstruction_r_error = 0.0

    for scenario_name, scenario in SCENARIOS.items():
        source_file, old_regime = REPLAY_SOURCE[scenario_name]
        method_signals = signals.loc[(signals.scenario == scenario_name) & (signals.method == "Mean-GMR MPC")].sort_values("time_step")
        method_trace = trace.loc[(trace.scenario == scenario_name) & (trace.method == "Mean-GMR MPC") & (trace.iteration_index == 0)]
        if len(method_trace) != len(method_signals):
            raise AssertionError(f"Missing recorded nominal candidates for {scenario_name}")
        plant = ResidualOutputPlant(
            OutputResidualReplay(_replay_stream(test, source_file, old_regime, len(method_signals))),
            initial_state=scenario.initial_state,
        )
        selected = _selected_steps(len(method_signals))
        previous_delta = 0.0
        for signal in method_signals.itertuples(index=False):
            state = plant.get_state()
            if int(signal.time_step) in selected:
                trace_row = method_trace.loc[method_trace.time_step == signal.time_step].iloc[0]
                nominal_controls = np.asarray(json.loads(trace_row.U_j), float)
                error = global_state_to_error_state(state, scenario.reference_path)
                preview = scenario.reference_path.generate_horizon(
                    error.reference.s_m, HORIZON, SAMPLE_TIME_S, SPEED_MPS
                )
                nominal_prediction = layer.predict(
                    {"vy": state.vy_mps, "r": state.yaw_rate_radps},
                    {"vx": SPEED_MPS, "steering": previous_delta},
                    nominal_controls,
                )
                oracle_window_id, oracle_delta_r, oracle_match_distance = _matched_oracle_window(
                    nominal_prediction.gmr_features, source_file, old_regime, oracle_windows, feature_scale
                )
                candidates = candidate_set(nominal_controls, previous_delta, config, rng)
                cycle_rows = []
                for candidate_index, controls in enumerate(candidates):
                    metrics, candidate_horizon = evaluate_candidate(
                        layer, predictor, state, error.as_vector(), preview.curvature_per_m,
                        controls, previous_delta, oracle_delta_r, config,
                    )
                    metadata = {
                        "scenario": scenario_name,
                        "time_step": int(signal.time_step),
                        "time_s": float(signal.time_step) * SAMPLE_TIME_S,
                        "candidate_index": candidate_index,
                        "control_distance_from_nominal": float(np.linalg.norm(controls - nominal_controls)),
                        "control_sequence": json.dumps(controls.tolist()),
                        "oracle_window_id": oracle_window_id,
                        "oracle_match_standardized_rms": oracle_match_distance,
                    }
                    candidate_rows.append({**metadata, **metrics})
                    cycle_rows.append({**metadata, **metrics})
                    for horizon_row in candidate_horizon:
                        horizon_rows.append({**{key: metadata[key] for key in ("scenario", "time_step", "candidate_index")}, **horizon_row})
                cycle = pd.DataFrame(cycle_rows)
                costs = {name: cycle[f"{name}_total"].to_numpy(float) for name in ("nominal", "current", "full", "oracle")}
                best = {name: int(np.argmin(values)) for name, values in costs.items()}
                ranking = {
                    "scenario": scenario_name,
                    "time_step": int(signal.time_step),
                    "candidate_count": len(cycle),
                    "nominal_current_pairwise_flip_rate": pairwise_flip_fraction(costs["nominal"], costs["current"]),
                    "current_full_pairwise_flip_rate": pairwise_flip_fraction(costs["current"], costs["full"]),
                    "nominal_full_pairwise_flip_rate": pairwise_flip_fraction(costs["nominal"], costs["full"]),
                    "full_oracle_pairwise_flip_rate": pairwise_flip_fraction(costs["full"], costs["oracle"]),
                    "nominal_oracle_pairwise_flip_rate": pairwise_flip_fraction(costs["nominal"], costs["oracle"]),
                    "nominal_best_index": best["nominal"],
                    "current_best_index": best["current"],
                    "full_best_index": best["full"],
                    "oracle_best_index": best["oracle"],
                    "current_optimum_changed_vs_nominal": best["current"] != best["nominal"],
                    "full_optimum_changed_vs_current": best["full"] != best["current"],
                    "full_optimum_changed_vs_nominal": best["full"] != best["nominal"],
                    "oracle_optimum_changed_vs_full": best["oracle"] != best["full"],
                    "oracle_optimum_changed_vs_nominal": best["oracle"] != best["nominal"],
                    "current_full_best_control_l2": float(np.linalg.norm(candidates[best["current"]] - candidates[best["full"]])),
                    "nominal_full_best_control_l2": float(np.linalg.norm(candidates[best["nominal"]] - candidates[best["full"]])),
                    "full_oracle_best_control_l2": float(np.linalg.norm(candidates[best["full"]] - candidates[best["oracle"]])),
                    "oracle_window_id": oracle_window_id,
                    "oracle_match_standardized_rms": oracle_match_distance,
                }
                ranking_rows.append(ranking)
                oracle_rows.append(
                    {
                        **{key: ranking[key] for key in ("scenario", "time_step", "oracle_window_id", "oracle_match_standardized_rms")},
                        "predicted_full_best_index": best["full"],
                        "oracle_best_index": best["oracle"],
                        "best_index_mismatch": best["full"] != best["oracle"],
                        "best_control_l2_difference": ranking["full_oracle_best_control_l2"],
                        "pairwise_decision_gap": ranking["full_oracle_pairwise_flip_rate"],
                        "oracle_regret_at_predicted_best": float(costs["oracle"][best["full"]] - costs["oracle"][best["oracle"]]),
                        "predicted_regret_at_oracle_best": float(costs["full"][best["oracle"]] - costs["full"][best["full"]]),
                    }
                )
            next_state = plant.step(float(signal.delta_rad), SPEED_MPS, SAMPLE_TIME_S)
            max_reconstruction_r_error = max(max_reconstruction_r_error, abs(next_state.yaw_rate_radps - float(signal.r_plant)))
            previous_delta = float(signal.delta_rad)

    candidates = pd.DataFrame(candidate_rows)
    rankings = pd.DataFrame(ranking_rows)
    oracle_gap = pd.DataFrame(oracle_rows)
    raw_horizon = pd.DataFrame(horizon_rows)
    horizon_metrics = []
    for (step, tau), group in raw_horizon.groupby(["prediction_step", "prediction_time_s"], sort=True):
        row = {"prediction_step": int(step), "prediction_time_s": float(tau), "sample_count": len(group)}
        for source in ("predicted", "oracle"):
            for variable in ("delta_r", "delta_epsi", "delta_ey"):
                values = group[f"{source}_{variable}"].to_numpy(float)
                row[f"{source}_{variable}_mean"] = float(np.mean(values))
                row[f"{source}_abs_{variable}_mean"] = float(np.mean(np.abs(values)))
                row[f"{source}_abs_{variable}_median"] = float(np.median(np.abs(values)))
                row[f"{source}_abs_{variable}_p95"] = float(np.quantile(np.abs(values), 0.95))
                row[f"{source}_abs_{variable}_max"] = float(np.max(np.abs(values)))
        horizon_metrics.append(row)
    provenance = {
        "selected_cycles": len(rankings),
        "candidate_evaluations": len(candidates),
        "candidates_per_cycle": CANDIDATES_PER_SAMPLE,
        "samples_per_scenario": SAMPLES_PER_SCENARIO,
        "maximum_reconstructed_r_plant_error": max_reconstruction_r_error,
        "candidate_source": "existing Mean-GMR iteration-0 U(j) plus feasible projected perturbations",
        "nominal_optimizer_executed": False,
        "chrono_executed": False,
        "oracle_definition": "One nearest complete held-out Stanford residual window is matched to the nominal candidate and fixed across each cycle's candidate set.",
        "oracle_limitation": "Observed data cannot provide counterfactual true residuals for unexecuted candidate controls; this is a matched-window offline upper-bound proxy.",
    }
    return candidates, rankings, oracle_gap, pd.DataFrame(horizon_metrics), provenance


def _stats(values: Iterable[float]) -> dict:
    data = np.asarray(list(values), float)
    return {
        "mean": float(np.mean(data)), "std": float(np.std(data)), "min": float(np.min(data)),
        "median": float(np.median(data)), "p95": float(np.quantile(data, 0.95)), "max": float(np.max(data)),
    }


def summarize(candidates: pd.DataFrame, rankings: pd.DataFrame, oracle_gap: pd.DataFrame,
              horizon: pd.DataFrame, provenance: dict) -> dict:
    ranking = {
        "nominal_to_current": {
            "pairwise_flip_rate": float(rankings.nominal_current_pairwise_flip_rate.mean()),
            "optimum_change_rate": float(rankings.current_optimum_changed_vs_nominal.mean()),
        },
        "current_to_full": {
            "pairwise_flip_rate": float(rankings.current_full_pairwise_flip_rate.mean()),
            "optimum_change_rate": float(rankings.full_optimum_changed_vs_current.mean()),
            "best_control_l2": _stats(rankings.current_full_best_control_l2),
        },
        "nominal_to_full": {
            "pairwise_flip_rate": float(rankings.nominal_full_pairwise_flip_rate.mean()),
            "optimum_change_rate": float(rankings.full_optimum_changed_vs_nominal.mean()),
            "best_control_l2": _stats(rankings.nominal_full_best_control_l2),
        },
        "full_to_oracle": {
            "pairwise_flip_rate": float(rankings.full_oracle_pairwise_flip_rate.mean()),
            "optimum_change_rate": float(rankings.oracle_optimum_changed_vs_full.mean()),
            "best_control_l2": _stats(rankings.full_oracle_best_control_l2),
        },
        "nominal_to_oracle": {
            "pairwise_flip_rate": float(rankings.nominal_oracle_pairwise_flip_rate.mean()),
            "optimum_change_rate": float(rankings.oracle_optimum_changed_vs_nominal.mean()),
        },
    }
    cost_changes = {
        name: _stats(candidates[column])
        for name, column in {
            "current_minus_nominal": "current_minus_nominal",
            "full_minus_nominal": "full_minus_nominal",
            "full_minus_current": "full_minus_current",
            "oracle_minus_nominal": "oracle_minus_nominal",
            "oracle_minus_full": "oracle_minus_full",
            "full_current_stage_tracking_change": "full_current_stage_tracking_change",
            "full_current_terminal_change": "full_current_terminal_change",
        }.items()
    }
    full_material = ranking["current_to_full"]["pairwise_flip_rate"] >= 0.05 or ranking["current_to_full"]["optimum_change_rate"] >= 0.10
    full_natural = ranking["nominal_to_full"]["pairwise_flip_rate"] >= 0.05 or ranking["nominal_to_full"]["optimum_change_rate"] >= 0.10
    oracle_material = ranking["nominal_to_oracle"]["pairwise_flip_rate"] >= 0.05 or ranking["nominal_to_oracle"]["optimum_change_rate"] >= 0.10
    oracle_gap_material = ranking["full_to_oracle"]["pairwise_flip_rate"] >= 0.05 or ranking["full_to_oracle"]["optimum_change_rate"] >= 0.10
    if full_material and not oracle_gap_material:
        bottleneck = "incomplete residual propagation"
    elif full_material and oracle_gap_material:
        bottleneck = (
            "incomplete residual propagation (established primary bottleneck), with a secondary "
            "predicted-to-matched-oracle decision gap that is proxy-limited"
        )
    elif not full_material and oracle_material:
        bottleneck = "weak current propagation; the matched-oracle gap is proxy-limited"
    else:
        bottleneck = "MPC objective insensitivity to yaw-residual propagation"
    recommend = bool(full_natural and ranking["nominal_to_full"]["optimum_change_rate"] >= 0.10)
    return {
        "ranking_sensitivity": ranking,
        "cost_change_distributions": cost_changes,
        "propagation_endpoint_mean_abs": {
            key: float(horizon.iloc[-1][key])
            for key in ("predicted_abs_delta_r_mean", "predicted_abs_delta_epsi_mean", "predicted_abs_delta_ey_mean",
                        "oracle_abs_delta_r_mean", "oracle_abs_delta_epsi_mean", "oracle_abs_delta_ey_mean")
        },
        "oracle_decision_gap": {
            "pairwise_flip_rate": float(oracle_gap.pairwise_decision_gap.mean()),
            "best_candidate_mismatch_rate": float(oracle_gap.best_index_mismatch.mean()),
            "best_control_l2": _stats(oracle_gap.best_control_l2_difference),
            "oracle_regret_at_predicted_best": _stats(oracle_gap.oracle_regret_at_predicted_best),
            "match_distance": _stats(oracle_gap.oracle_match_standardized_rms),
        },
        "A_full_propagation_stronger_than_current": full_material,
        "B_original_ey_epsi_objective_naturally_changes_decision": full_natural,
        "C_oracle_gap_material": oracle_gap_material,
        "C_oracle_gap_interpretation": (
            "The gap is material, but it cannot be attributed solely to GMR prediction error: "
            "held-out observations do not contain counterfactual residuals for unexecuted controls, "
            "and the matched-window standardized RMS distance is nonzero."
        ),
        "D_primary_bottleneck": bottleneck,
        "E_recommend_full_propagated_closed_loop": recommend,
        "E_recommendation_rule": "Recommend only if nominal-to-full pairwise flip >=5% or optimum change >=10%, with optimum change itself >=10%.",
        "provenance": provenance,
    }


def save_figures(candidates: pd.DataFrame, rankings: pd.DataFrame, oracle_gap: pd.DataFrame, horizon: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    tau = horizon.prediction_time_s.to_numpy(float)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, variable, label in zip(axes, ("delta_r", "delta_epsi", "delta_ey"), ("|Delta r| (rad/s)", "|Delta epsi| (rad)", "|Delta ey| (m)")):
        axis.plot(tau, horizon[f"predicted_abs_{variable}_mean"], "o-", label="Predicted GMR")
        axis.plot(tau, horizon[f"oracle_abs_{variable}_mean"], "s--", label="Matched oracle")
        axis.set(xlabel="Prediction horizon (s)", ylabel=label, title=variable.replace("delta_", "Delta "))
        axis.grid(alpha=0.25); axis.legend()
    figure.suptitle("Delta-r to heading and lateral-error propagation"); figure.tight_layout()
    figure.savefig(FIGURES / "figure_1_residual_to_trajectory_horizon_propagation.png", dpi=320); plt.close(figure)

    sample = candidates.loc[(candidates.scenario == "s3_double_lane_change")]
    sample = sample.loc[sample.time_step == sample.time_step.iloc[len(sample) // 2]]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for name, style in zip(("nominal", "current", "full"), ("o-", "s--", "^-.")):
        axis.plot(sample.candidate_index, sample[f"{name}_total"], style, label=name)
    axis.set(xlabel="Candidate index", ylabel="Frozen MPC objective", title="Same candidate: nominal vs current vs full propagation")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(FIGURES / "figure_2_candidate_cost_comparison.png", dpi=320); plt.close(figure)

    labels = ["Nom→Current", "Current→Full", "Nom→Full", "Full→Oracle", "Nom→Oracle"]
    flip = [rankings.nominal_current_pairwise_flip_rate.mean(), rankings.current_full_pairwise_flip_rate.mean(),
            rankings.nominal_full_pairwise_flip_rate.mean(), rankings.full_oracle_pairwise_flip_rate.mean(),
            rankings.nominal_oracle_pairwise_flip_rate.mean()]
    optimum = [rankings.current_optimum_changed_vs_nominal.mean(), rankings.full_optimum_changed_vs_current.mean(),
               rankings.full_optimum_changed_vs_nominal.mean(), rankings.oracle_optimum_changed_vs_full.mean(),
               rankings.oracle_optimum_changed_vs_nominal.mean()]
    x = np.arange(len(labels)); width = 0.36
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x - width / 2, flip, width, label="Pairwise ranking flip")
    axis.bar(x + width / 2, optimum, width, label="Best candidate change")
    axis.set_xticks(x, labels, rotation=15); axis.set(ylabel="Fraction", title="Candidate ranking sensitivity")
    axis.axhline(0.05, color="black", linestyle=":", linewidth=1, label="5% reference")
    axis.grid(axis="y", alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(FIGURES / "figure_3_candidate_ranking_change.png", dpi=320); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(candidates.full_minus_nominal, candidates.oracle_minus_nominal, alpha=0.35, s=14)
    lo = min(candidates.full_minus_nominal.min(), candidates.oracle_minus_nominal.min())
    hi = max(candidates.full_minus_nominal.max(), candidates.oracle_minus_nominal.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xlabel="Predicted full Delta J", ylabel="Oracle full Delta J", title="Predicted vs oracle cost sensitivity")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(oracle_gap.oracle_match_standardized_rms, oracle_gap.pairwise_decision_gap, alpha=0.75)
    axes[1].set(xlabel="Oracle-window standardized RMS match distance", ylabel="Full-vs-oracle ranking gap", title="Oracle decision gap")
    axes[1].grid(alpha=0.25); figure.tight_layout()
    figure.savefig(FIGURES / "figure_4_predicted_vs_oracle_decision_sensitivity.png", dpi=320); plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 5) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=[np.number]):
        view[column] = view[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(summary: dict, rankings: pd.DataFrame, oracle_gap: pd.DataFrame, horizon: pd.DataFrame) -> None:
    sensitivity = summary["ranking_sensitivity"]
    report = f"""# Residual-to-Trajectory Propagation Feasibility Audit

## 1. 冻结范围

本审计没有修改或重新求解Chrono、Nominal MPC、正式optimizer、Q/R/P权重、约束、
GMR模型、Stanford数据或ResidualOutputPlant。候选中心来自既有Mean-GMR迭代日志的
iteration-0名义控制序列，并在冻结转角/转角增量约束内生成局部候选云。

## 2. 实际误差状态和传播定义

- `ey > 0`：车辆位于参考路径左侧；
- `epsi = psi - psi_ref`；
- Nominal模型：`ey_dot = vx*epsi + vy`，`epsi_dot = r - vx*kappa`；
- Current：只将`cumsum(mu_delta_r*Ts)`加到epsi；
- Full：在相同epsi修正基础上，通过Frenet横向误差运动学累积Delta-ey；
- 没有学习或人为添加Delta-ey；
- r和ay没有新增cost term，所有方法使用原始Q/R/P。

Full增量传播：

```text
Delta ey_dot = vx[sin(epsi + Delta epsi) - sin(epsi)]
             + vy[cos(epsi + Delta epsi) - cos(epsi)]
```

## 3. Horizon传播量级

{_markdown_table(horizon, ['prediction_time_s', 'predicted_abs_delta_r_mean', 'predicted_abs_delta_epsi_mean', 'predicted_abs_delta_ey_mean', 'oracle_abs_delta_r_mean', 'oracle_abs_delta_epsi_mean', 'oracle_abs_delta_ey_mean'])}

1秒端点平均绝对传播量：

`{json.dumps(summary['propagation_endpoint_mean_abs'], ensure_ascii=False)}`

## 4. Candidate ranking结果

`{json.dumps(sensitivity, ensure_ascii=False)}`

逐周期结果：

{_markdown_table(rankings, ['scenario', 'time_step', 'nominal_current_pairwise_flip_rate', 'current_full_pairwise_flip_rate', 'nominal_full_pairwise_flip_rate', 'full_oracle_pairwise_flip_rate', 'full_optimum_changed_vs_current', 'oracle_optimum_changed_vs_full'])}

## 5. Oracle上界说明

每个周期使用与名义候选六维rollout特征最近的同session/同工况完整held-out窗口，
读取其真实20步Delta-r，并在该周期全部候选间固定。由于观测数据不能提供未执行
候选控制的反事实真实残差，这只是matched-window离线上界代理，不是真实闭环oracle。

Oracle decision gap：

`{json.dumps(summary['oracle_decision_gap'], ensure_ascii=False)}`

## 6. 最终诊断

### A. Full传播是否比epsi-only更强？

`{summary['A_full_propagation_stronger_than_current']}`

### B. 是否可通过原始ey/epsi目标自然改变控制？

`{summary['B_original_ey_epsi_objective_naturally_changes_decision']}`

### C. Predicted与Oracle decision gap是否显著？

`{summary['C_oracle_gap_material']}`

解释：{summary['C_oracle_gap_interpretation']}

### D. 当前主要瓶颈

**{summary['D_primary_bottleneck']}**

### E. 是否建议进入Full Residual-Propagated Mean-GMR MPC闭环？

`{summary['E_recommend_full_propagated_closed_loop']}`

判据：{summary['E_recommendation_rule']}

本审计只回答传播和候选排序可行性，不声称ResidualOutputPlant是control-matched真实车辆，
也没有授权修改正式MPC。这里的Oracle差距不能被单独解释为GMR预测误差；它同时包含
非反事实matched-window代理误差，尤其在匹配距离较大的周期中应谨慎解释。
"""
    (OUT / "residual_propagation_report.md").write_text(report, encoding="utf-8")


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    frozen_paths = [
        MODEL_PATH, TRAIN_PATH, TEST_PATH, SIGNALS_PATH, TRACE_PATH,
        ROOT.parents[1] / "bicycle_model.py",
        ROOT.parent / "later_mpc" / "mpc_optimizer.py",
        ROOT.parent / "later_mpc" / "nominal_bicycle_predictor.py",
        ROOT / "mpc_integration" / "residual_output_plant.py",
    ]
    hashes_before = {str(path): _sha256(path) for path in frozen_paths}
    candidates, rankings, oracle_gap, horizon, provenance = run_analysis()
    summary = summarize(candidates, rankings, oracle_gap, horizon, provenance)
    summary.update(
        {
            "audit": "Residual-to-Trajectory Propagation Feasibility Audit",
            "frozen_Q_R_P": {
                "state_weights": MPCConfig().state_weights.__dict__,
                "input_weights": MPCConfig().input_weights.__dict__,
            },
            "new_cost_terms_added": False,
            "optimizer_executed_or_modified": False,
            "chrono_executed_or_modified": False,
            "gmr_modified": False,
            "dataset_modified": False,
            "residual_output_plant_modified": False,
            "previous_cost_audit_ranking_flip_fraction": json.loads(COST_AUDIT_SUMMARY.read_text(encoding="utf-8"))["ranking_flip_fraction"],
        }
    )
    save_figures(candidates, rankings, oracle_gap, horizon)
    candidates.to_csv(OUT / "residual_propagation_metrics.csv", index=False)
    rankings.to_csv(OUT / "candidate_ranking_comparison.csv", index=False)
    oracle_gap.to_csv(OUT / "oracle_decision_gap.csv", index=False)
    horizon.to_csv(OUT / "propagation_horizon_metrics.csv", index=False)
    hashes_after = {str(path): _sha256(path) for path in frozen_paths}
    summary["frozen_input_sha256_before"] = hashes_before
    summary["frozen_input_sha256_after"] = hashes_after
    summary["frozen_inputs_modified"] = hashes_before != hashes_after
    summary["figures"] = sorted(path.name for path in FIGURES.glob("*.png"))
    (OUT / "residual_propagation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, rankings, oracle_gap, horizon)
    print(json.dumps({
        "frozen_inputs_modified": summary["frozen_inputs_modified"],
        "selected_cycles": provenance["selected_cycles"],
        "candidate_evaluations": provenance["candidate_evaluations"],
        "ranking_sensitivity": summary["ranking_sensitivity"],
        "diagnosis": summary["D_primary_bottleneck"],
        "recommend_closed_loop": summary["E_recommend_full_propagated_closed_loop"],
        "output_directory": str(OUT),
    }, indent=2))
    return summary


if __name__ == "__main__":
    main()
