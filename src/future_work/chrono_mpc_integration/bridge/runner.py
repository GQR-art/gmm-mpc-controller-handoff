"""Deterministic multi-rate runner for Nominal MPC and a Chrono plant."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable

from future_work.later_mpc.controller import ControllerOutput
from future_work.later_mpc.error_state import global_state_to_error_state

from .chrono_mpc_controller import NominalMPCBridge
from .chrono_vehicle import ChronoSedanPlant
from .speed_controller import LongitudinalCommand, LongitudinalSpeedPI
from .steering_adapter import SteeringCalibration
from .state_mapper import chrono_raw_to_vehicle_state
from .types import ChronoRawState


@dataclass(frozen=True, slots=True)
class MultiRateRunnerConfig:
    physics_dt_s: float = 0.002
    mpc_dt_s: float = 0.05
    target_speed_mps: float = 15.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.physics_dt_s, self.mpc_dt_s, self.target_speed_mps)
        ):
            raise ValueError("Runner timing and target speed must be finite.")
        if self.physics_dt_s <= 0.0 or self.mpc_dt_s <= 0.0:
            raise ValueError("Runner time steps must be positive.")
        if self.target_speed_mps <= 0.0:
            raise ValueError("target_speed_mps must be positive.")
        ratio = self.mpc_dt_s / self.physics_dt_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("mpc_dt_s must be an integer multiple of physics_dt_s.")

    @property
    def physics_steps_per_mpc_update(self) -> int:
        return int(round(self.mpc_dt_s / self.physics_dt_s))


@dataclass(frozen=True, slots=True)
class ClosedLoopSample:
    time: float
    X: float
    Y: float
    psi: float
    vx: float
    vy: float
    yaw_rate: float
    ey: float
    epsi: float
    reference_X: float
    reference_Y: float
    reference_psi: float
    reference_curvature: float
    delta_mpc_rad: float
    delta_increment: float
    steering_normalized: float
    throttle: float
    brake: float
    solver_status: str
    solver_time: float
    mpc_updated: bool
    roll: float
    beta: float
    FL_Fz: float
    FR_Fz: float
    RL_Fz: float
    RR_Fz: float
    wheel_contact_count: int


@dataclass(frozen=True, slots=True)
class VehicleSettlingResult:
    passed: bool
    duration_s: float
    first_four_wheel_contact_s: float | None
    stable_window_s: float
    final_roll_rad: float
    final_pitch_rad: float
    final_vx_mps: float
    final_wheel_contact_count: int
    final_Fz_N: tuple[float, float, float, float]
    max_abs_longitudinal_slip: float


def settle_vehicle(
    plant: ChronoSedanPlant,
    *,
    vx_reference_mps: float,
    physics_dt_s: float = 0.002,
    normalized_steering: float = 0.0,
    timeout_s: float = 8.0,
    step_callback: Callable[[ChronoRawState, LongitudinalCommand], bool] | None = None,
) -> VehicleSettlingResult:
    """Apply the frozen Phase 2.15 state-based settling protocol without MPC."""
    if not plant.initialized:
        raise RuntimeError("Initialize the Chrono plant before settling.")
    if plant.vehicle_model != "GolfLike3D_v1":
        raise ValueError("The formal settling protocol requires GolfLike3D_v1.")
    if not math.isclose(physics_dt_s, 0.002, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The frozen settling protocol requires physics_dt_s=0.002.")

    speed_controller = LongitudinalSpeedPI()
    start_time = plant.get_raw_state().time
    stable_start: float | None = None
    first_four: float | None = None
    stable_window_s = 0.35
    passed = False

    while plant.get_raw_state().time - start_time < timeout_s - 1e-12:
        raw_state = plant.get_raw_state()
        speed_command = speed_controller.update(
            vx_reference_mps=vx_reference_mps,
            vx_measured_mps=raw_state.vx_body,
            dt_s=physics_dt_s,
        )
        plant.advance(
            physics_dt_s,
            normalized_steering=normalized_steering,
            throttle=speed_command.throttle,
            braking=speed_command.braking,
        )

        if step_callback is not None and not step_callback(
            plant.get_raw_state(), speed_command
        ):
            break

        elapsed = plant.get_raw_state().time - start_time
        suspension = plant.get_suspension_diagnostics()
        wheels = plant.get_wheel_diagnostics()
        contact_count = sum(item.contact for item in suspension)
        if contact_count == 4 and first_four is None:
            first_four = elapsed
        roll, pitch = plant.get_chassis_roll_pitch_rad()
        max_fz = max(item.force_z_tire_N for item in wheels)
        max_slip = max(abs(item.longitudinal_slip_sx) for item in wheels)
        vx_actual = plant.get_raw_state().vx_body
        stable = bool(
            elapsed >= 1.0
            and contact_count == 4
            and max_fz < 12000.0
            and abs(roll) < math.radians(2.0)
            and abs(pitch) < math.radians(5.0)
            and abs(vx_actual - vx_reference_mps) < 0.5
            and max_slip < 0.35
        )
        stable_start = elapsed if stable and stable_start is None else (
            stable_start if stable else None
        )
        if stable_start is not None and elapsed - stable_start >= stable_window_s:
            passed = True
            break

    raw_state = plant.get_raw_state()
    suspension = plant.get_suspension_diagnostics()
    wheels = plant.get_wheel_diagnostics()
    roll, pitch = plant.get_chassis_roll_pitch_rad()
    return VehicleSettlingResult(
        passed=passed,
        duration_s=raw_state.time - start_time,
        first_four_wheel_contact_s=first_four,
        stable_window_s=stable_window_s,
        final_roll_rad=roll,
        final_pitch_rad=pitch,
        final_vx_mps=raw_state.vx_body,
        final_wheel_contact_count=sum(item.contact for item in suspension),
        final_Fz_N=tuple(item.force_z_tire_N for item in wheels),  # type: ignore[arg-type]
        max_abs_longitudinal_slip=max(abs(item.longitudinal_slip_sx) for item in wheels),
    )


class MultiRateChronoMPCRunner:
    """Run integer-counter MPC scheduling over fixed Chrono physics steps."""

    def __init__(
        self,
        *,
        plant: ChronoSedanPlant,
        mpc_bridge: NominalMPCBridge,
        steering_calibration: SteeringCalibration,
        speed_controller: LongitudinalSpeedPI | None = None,
        config: MultiRateRunnerConfig | None = None,
    ) -> None:
        self.plant = plant
        self.mpc_bridge = mpc_bridge
        self.steering_calibration = steering_calibration
        self.speed_controller = speed_controller or LongitudinalSpeedPI()
        self.config = config or MultiRateRunnerConfig()
        if not math.isclose(self.config.physics_dt_s, 0.002, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Phase 3 requires physics_dt_s=0.002.")
        if not math.isclose(self.config.mpc_dt_s, 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Phase 3 requires mpc_dt_s=0.05.")
        if not math.isclose(
            self.plant.config.physics_dt_s,
            self.config.physics_dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Plant and runner physics time steps must match.")
        mpc_config = self.mpc_bridge.controller.optimizer.config
        if not math.isclose(
            mpc_config.sample_time_s,
            self.config.mpc_dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Frozen MPC and runner sample times must match.")
        if not math.isclose(
            mpc_config.longitudinal_speed_mps,
            self.config.target_speed_mps,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Frozen MPC speed and Chrono speed target must match.")

    def run(
        self,
        duration_s: float,
        *,
        step_callback: Callable[[ClosedLoopSample], bool] | None = None,
    ) -> list[ClosedLoopSample]:
        """Execute a closed loop beginning with an MPC update at ``t=0``."""
        if not self.plant.initialized:
            raise RuntimeError("Initialize the Chrono plant before running the loop.")
        if self.plant.vehicle_model != "GolfLike3D_v1":
            raise ValueError("Phase 3 runner requires vehicle_model='GolfLike3D_v1'.")
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive.")
        total_steps_float = duration_s / self.config.physics_dt_s
        if not math.isclose(
            total_steps_float, round(total_steps_float), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("duration_s must be an integer number of physics steps.")

        total_steps = int(round(total_steps_float))
        ratio = self.config.physics_steps_per_mpc_update
        if ratio != 25:
            raise ValueError(f"Phase 3 requires a 25:1 rate ratio, got {ratio}:1.")

        self.mpc_bridge.reset(0.0)
        self.speed_controller.reset()
        step_counter = 0
        raw_state = self.plant.get_raw_state()
        closed_loop_start_time = raw_state.time
        previous_delta_rad = self.mpc_bridge.previous_delta_rad
        mpc_output = self.mpc_bridge.compute_control(raw_state)
        delta_increment = mpc_output.delta_rad - previous_delta_rad
        normalized_steering = self.steering_calibration.delta_rad_to_normalized(
            mpc_output.delta_rad
        )
        speed_command = self._speed_command(raw_state)
        samples = [
            self._sample(
                raw_state,
                mpc_output,
                normalized_steering,
                speed_command,
                mpc_updated=True,
                closed_loop_start_time=closed_loop_start_time,
                delta_increment=delta_increment,
            )
        ]
        if step_callback is not None and not step_callback(samples[-1]):
            return samples

        while step_counter < total_steps:
            self.plant.advance(
                self.config.physics_dt_s,
                normalized_steering=normalized_steering,
                throttle=speed_command.throttle,
                braking=speed_command.braking,
            )
            step_counter += 1
            raw_state = self.plant.get_raw_state()

            mpc_updated = step_counter % ratio == 0
            delta_increment = 0.0
            if mpc_updated:
                previous_delta_rad = self.mpc_bridge.previous_delta_rad
                mpc_output = self.mpc_bridge.compute_control(raw_state)
                delta_increment = mpc_output.delta_rad - previous_delta_rad
                normalized_steering = self.steering_calibration.delta_rad_to_normalized(
                    mpc_output.delta_rad
                )
            speed_command = self._speed_command(raw_state)
            samples.append(
                self._sample(
                    raw_state,
                    mpc_output,
                    normalized_steering,
                    speed_command,
                    mpc_updated=mpc_updated,
                    closed_loop_start_time=closed_loop_start_time,
                    delta_increment=delta_increment,
                )
            )
            if step_callback is not None and not step_callback(samples[-1]):
                break
        return samples

    def _speed_command(self, raw_state: ChronoRawState) -> LongitudinalCommand:
        return self.speed_controller.update(
            vx_reference_mps=self.config.target_speed_mps,
            vx_measured_mps=raw_state.vx_body,
            dt_s=self.config.physics_dt_s,
        )

    def _sample(
        self,
        raw_state: ChronoRawState,
        mpc_output: ControllerOutput,
        normalized_steering: float,
        speed_command: LongitudinalCommand,
        *,
        mpc_updated: bool,
        closed_loop_start_time: float,
        delta_increment: float,
    ) -> ClosedLoopSample:
        error = global_state_to_error_state(
            chrono_raw_to_vehicle_state(raw_state), self.mpc_bridge.reference_path
        )
        roll, _ = self.plant.get_chassis_roll_pitch_rad()
        wheels = {item.wheel_id: item for item in self.plant.get_wheel_diagnostics()}
        suspension = self.plant.get_suspension_diagnostics()
        return ClosedLoopSample(
            time=raw_state.time - closed_loop_start_time,
            X=raw_state.X,
            Y=raw_state.Y,
            psi=raw_state.psi,
            vx=raw_state.vx_body,
            vy=raw_state.vy_body,
            yaw_rate=raw_state.yaw_rate_body,
            ey=error.ey_m,
            epsi=error.epsi_rad,
            reference_X=error.reference.X_m,
            reference_Y=error.reference.Y_m,
            reference_psi=error.reference.psi_rad,
            reference_curvature=error.reference.curvature_per_m,
            delta_mpc_rad=mpc_output.delta_rad,
            delta_increment=delta_increment,
            steering_normalized=float(normalized_steering),
            throttle=speed_command.throttle,
            brake=speed_command.braking,
            solver_status=mpc_output.solution.status,
            solver_time=mpc_output.solution.solve_time_s,
            mpc_updated=mpc_updated,
            roll=roll,
            beta=math.atan2(raw_state.vy_body, raw_state.vx_body),
            FL_Fz=wheels["FL"].force_z_tire_N,
            FR_Fz=wheels["FR"].force_z_tire_N,
            RL_Fz=wheels["RL"].force_z_tire_N,
            RR_Fz=wheels["RR"].force_z_tire_N,
            wheel_contact_count=sum(item.contact for item in suspension),
        )
