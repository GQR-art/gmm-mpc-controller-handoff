"""Protocol-only wrapper for paired Chrono Nominal-MPC baseline audits.

This module deliberately composes the frozen Plant, NominalMPCBridge, runner,
scenario objects, and steering calibration.  It does not reimplement the MPC
or change any Plant, controller, scenario, or calibration parameter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from future_work.chrono_mpc_integration.bridge.chrono_config import ChronoSedanConfig
from future_work.chrono_mpc_integration.bridge.chrono_mpc_controller import NominalMPCBridge
from future_work.chrono_mpc_integration.bridge.chrono_vehicle import ChronoSedanPlant
from future_work.chrono_mpc_integration.bridge.runner import (
    ClosedLoopSample,
    MultiRateChronoMPCRunner,
    MultiRateRunnerConfig,
    VehicleSettlingResult,
    settle_vehicle,
)
from future_work.chrono_mpc_integration.bridge.speed_controller import LongitudinalSpeedPI
from future_work.chrono_mpc_integration.bridge.steering_adapter import SteeringCalibration
from future_work.chrono_mpc_integration.scenarios import ChronoMPCScenario


CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "steering"
    / "golflike3d_v1_steering_calibration.json"
)
PHYSICS_DT_S = 0.002
MPC_DT_S = 0.05
TARGET_SPEED_MPS = 15.0


@dataclass(frozen=True, slots=True)
class Pose2D:
    """One global planar pose using the project X/Y/yaw convention."""

    X_m: float
    Y_m: float
    psi_rad: float


@dataclass(frozen=True, slots=True)
class InitializationAudit:
    """All pose and settling evidence required to compare protocol variants."""

    protocol: str
    requested_initial_pose: Pose2D
    probe_post_settling_pose: Pose2D | None
    formal_requested_initial_pose: Pose2D
    formal_post_settling_pose: Pose2D
    probe_settling: dict[str, object] | None
    formal_settling: dict[str, object]


@dataclass(frozen=True, slots=True)
class ProtocolRun:
    """One independent direct or staged closed-loop run."""

    scenario_name: str
    protocol: str
    samples: list[ClosedLoopSample]
    initialization: InitializationAudit


def inverse_settling_pose(probe_pose: Pose2D) -> Pose2D:
    """Return the pre-settling pose that cancels a measured rigid displacement.

    This is the exact inverse 2-D rigid-body transform used by the historical
    R=80 and double-lane-change scripts.  It does not overwrite any state after
    settling; it only defines the initial pose of a second, independent Plant.
    """

    c = math.cos(-probe_pose.psi_rad)
    s = math.sin(-probe_pose.psi_rad)
    return Pose2D(
        X_m=-(c * probe_pose.X_m - s * probe_pose.Y_m),
        Y_m=-(s * probe_pose.X_m + c * probe_pose.Y_m),
        psi_rad=-probe_pose.psi_rad,
    )


def _runner_config() -> MultiRateRunnerConfig:
    return MultiRateRunnerConfig(
        physics_dt_s=PHYSICS_DT_S,
        mpc_dt_s=MPC_DT_S,
        target_speed_mps=TARGET_SPEED_MPS,
    )


def _new_initialized_plant(pose: Pose2D, *, initial_speed_mps: float) -> ChronoSedanPlant:
    plant = ChronoSedanPlant(
        ChronoSedanConfig(
            vehicle_model="GolfLike3D_v1",
            physics_dt_s=PHYSICS_DT_S,
            initial_x_m=pose.X_m,
            initial_y_m=pose.Y_m,
            initial_yaw_rad=pose.psi_rad,
            initial_forward_speed_mps=initial_speed_mps,
        )
    )
    plant.initialize()
    plant.initialize_wheel_angular_speeds(initial_speed_mps)
    return plant


def _settle(plant: ChronoSedanPlant, calibration: SteeringCalibration, *, speed_mps: float) -> VehicleSettlingResult:
    result = settle_vehicle(
        plant,
        vx_reference_mps=speed_mps,
        physics_dt_s=PHYSICS_DT_S,
        normalized_steering=calibration.delta_rad_to_normalized(0.0),
    )
    if not result.passed:
        raise RuntimeError(f"Frozen settling protocol failed: {result}")
    return result


def _raw_pose(plant: ChronoSedanPlant) -> Pose2D:
    state = plant.get_raw_state()
    return Pose2D(state.X, state.Y, state.psi)


def run_protocol(
    *,
    scenario_name: str,
    scenario: ChronoMPCScenario,
    protocol: str,
) -> ProtocolRun:
    """Run one direct or staged initialization protocol using frozen modules.

    ``direct`` reproduces the release candidate's current setup: one Plant at
    the scenario pose, then settling. ``staged`` adds the historical probe and
    inverse-transform initialization before the formal settling and score.
    """

    if protocol not in {"direct", "staged"}:
        raise ValueError("protocol must be 'direct' or 'staged'.")
    if not math.isclose(scenario.initial_speed_mps, TARGET_SPEED_MPS, abs_tol=1e-12):
        raise ValueError("The frozen baseline audit requires a 15 m/s scenario.")

    calibration = SteeringCalibration.from_json(
        CALIBRATION_PATH, expected_vehicle_model="GolfLike3D_v1"
    )
    requested_pose = Pose2D(
        scenario.initial_x_m, scenario.initial_y_m, scenario.initial_psi_rad
    )

    probe_pose: Pose2D | None = None
    probe_settling: dict[str, object] | None = None
    formal_requested_pose = requested_pose
    if protocol == "staged":
        probe = _new_initialized_plant(requested_pose, initial_speed_mps=scenario.initial_speed_mps)
        probe_result = _settle(probe, calibration, speed_mps=scenario.initial_speed_mps)
        probe_pose = _raw_pose(probe)
        probe_settling = asdict(probe_result)
        formal_requested_pose = inverse_settling_pose(probe_pose)

    plant = _new_initialized_plant(
        formal_requested_pose, initial_speed_mps=scenario.initial_speed_mps
    )
    formal_settling = _settle(plant, calibration, speed_mps=scenario.initial_speed_mps)
    post_settling_pose = _raw_pose(plant)

    bridge = NominalMPCBridge(reference_path=scenario.reference_path)
    runner = MultiRateChronoMPCRunner(
        plant=plant,
        mpc_bridge=bridge,
        steering_calibration=calibration,
        speed_controller=LongitudinalSpeedPI(),
        config=_runner_config(),
    )
    samples = runner.run(scenario.duration_s)
    if not samples:
        raise RuntimeError("The frozen runner returned no closed-loop samples.")
    return ProtocolRun(
        scenario_name=scenario_name,
        protocol=protocol,
        samples=samples,
        initialization=InitializationAudit(
            protocol=protocol,
            requested_initial_pose=requested_pose,
            probe_post_settling_pose=probe_pose,
            formal_requested_initial_pose=formal_requested_pose,
            formal_post_settling_pose=post_settling_pose,
            probe_settling=probe_settling,
            formal_settling=asdict(formal_settling),
        ),
    )
