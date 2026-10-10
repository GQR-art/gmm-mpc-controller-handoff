"""Shared headless Chrono Sedan/GolfLike plant bridge.

This module contains no MPC, path following, steering calibration, or
longitudinal controller.  It only assembles a JSON Sedan plant, advances it
with explicitly supplied open-loop driver inputs, and exposes typed
measurements.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from collections.abc import Iterator
from typing import Any
import math

from .chrono_config import ChronoSedanConfig, VehicleModel
from .types import (
    ChronoRawState,
    ChronoSuspensionDiagnostic,
    ChronoVehicleMassProperties,
    ChronoWheelDiagnostic,
    ChronoWheelSpeedInitialization,
)


class ChronoSedanPlant:
    """Own one selectable Chrono JSON vehicle and a rigid-terrain plant.

    The historical class name is retained for compatibility.  ``Sedan`` is
    still the default model; pass ``vehicle_model="GolfLike"`` or
    ``vehicle_model="GolfLike3D_v1"`` to select a derived validation plant.
    """

    def __init__(
        self,
        config: ChronoSedanConfig | None = None,
        *,
        vehicle_model: VehicleModel | None = None,
    ) -> None:
        selected_config = config or ChronoSedanConfig()
        if vehicle_model is not None:
            selected_config = replace(selected_config, vehicle_model=vehicle_model)
        self.config = selected_config
        self._chrono: Any | None = None
        self._vehicle_module: Any | None = None
        self._vehicle: Any | None = None
        self._terrain: Any | None = None
        self._dll_directory_handle: Any | None = None

    @property
    def initialized(self) -> bool:
        """Whether vehicle and terrain have both been constructed."""
        return self._vehicle is not None and self._terrain is not None

    @property
    def vehicle_model(self) -> str:
        """Configured model name (Sedan or one of the derived GolfLike plants)."""
        return self.config.vehicle_model

    def initialize(self) -> None:
        """Construct the selected JSON vehicle with axle-specific tires."""
        if self.initialized:
            raise RuntimeError("Chrono vehicle plant is already initialized.")

        self.config.validate()
        chrono, vehicle_module = self._import_pychrono()
        self._chrono = chrono
        self._vehicle_module = vehicle_module

        # These process-local paths make the build-tree Python bindings and
        # their matching data tree explicit; no system environment is changed.
        chrono.SetChronoDataPath(self._trailing_separator(self.config.data_dir))
        vehicle_module.SetVehicleDataPath(
            self._trailing_separator(self.config.vehicle_data_dir)
        )

        # Chrono resolves nested component paths from its process-global
        # VehicleDataPath, not relative to the top-level vehicle JSON.  Keep
        # derived JSON files readable and location-independent, and materialize
        # their nested component references into a temporary runtime JSON.
        with self._runtime_vehicle_json() as runtime_vehicle_json:
            vehicle = vehicle_module.WheeledVehicle(
                str(runtime_vehicle_json), chrono.ChContactMethod_NSC
            )
        initial_pose = chrono.ChCoordsysd(
            chrono.ChVector3d(
                self.config.initial_x_m,
                self.config.initial_y_m,
                self.config.initial_z_m,
            ),
            chrono.QuatFromAngleZ(float(self.config.initial_yaw_rad)),
        )
        vehicle.Initialize(initial_pose, self.config.initial_forward_speed_mps)
        vehicle.GetChassis().SetFixed(False)
        vehicle.SetChassisVisualizationType(chrono.VisualizationType_NONE)
        vehicle.SetSuspensionVisualizationType(chrono.VisualizationType_NONE)
        vehicle.SetSteeringVisualizationType(chrono.VisualizationType_NONE)
        vehicle.SetWheelVisualizationType(chrono.VisualizationType_NONE)
        vehicle.SetTireVisualizationType(chrono.VisualizationType_NONE)

        engine = vehicle_module.ReadEngineJSON(str(self.config.sedan_engine_json))
        transmission = vehicle_module.ReadTransmissionJSON(
            str(self.config.sedan_transmission_json)
        )
        vehicle.InitializePowertrain(vehicle_module.ChPowertrainAssembly(engine, transmission))

        axles = vehicle.GetAxles()
        if len(axles) != 2:
            raise RuntimeError(
                "The Sedan/GolfLike bridge requires exactly two axles; "
                f"the selected vehicle exposes {len(axles)}."
            )
        for axle_index, axle in enumerate(axles):
            tire_json = self.config.tire_json_for_axle(axle_index)
            for wheel in axle.GetWheels():
                tire = vehicle_module.ReadTireJSON(str(tire_json))
                vehicle.InitializeTire(tire, wheel, chrono.VisualizationType_NONE)

        vehicle.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        terrain = vehicle_module.RigidTerrain(
            vehicle.GetSystem(), str(self.config.rigid_terrain_json)
        )
        terrain.Initialize()

        self._vehicle = vehicle
        self._terrain = terrain

    def advance(
        self,
        dt_s: float | None = None,
        *,
        normalized_steering: float = 0.0,
        throttle: float = 0.0,
        braking: float = 0.0,
        clutch: float = 0.0,
    ) -> None:
        """Advance one step with direct open-loop Chrono driver inputs.

        This is a transport operation, not a controller: callers must provide
        the already-normalized steering command and any actuator values.
        """
        self._require_initialized()
        step = self.config.physics_dt_s if dt_s is None else float(dt_s)
        if step <= 0.0:
            raise ValueError("dt_s must be positive.")
        if not -1.0 <= normalized_steering <= 1.0:
            raise ValueError("normalized_steering must be in [-1, 1].")
        if not 0.0 <= throttle <= 1.0:
            raise ValueError("throttle must be in [0, 1].")
        if not 0.0 <= braking <= 1.0:
            raise ValueError("braking must be in [0, 1].")
        if not 0.0 <= clutch <= 1.0:
            raise ValueError("clutch must be in [0, 1].")

        assert self._vehicle is not None
        assert self._terrain is not None
        assert self._vehicle_module is not None

        driver_input = self.make_driver_inputs(
            normalized_steering=normalized_steering,
            throttle=throttle,
            braking=braking,
            clutch=clutch,
        )
        time = self._vehicle.GetSystem().GetChTime()
        self._vehicle.Synchronize(time, driver_input, self._terrain)
        self._terrain.Synchronize(time)
        self._vehicle.Advance(step)
        self._terrain.Advance(step)

    def make_driver_inputs(
        self,
        *,
        normalized_steering: float = 0.0,
        throttle: float = 0.0,
        braking: float = 0.0,
        clutch: float = 0.0,
    ) -> Any:
        """Build the Chrono driver-input object used by plant and visualizer.

        This is a representation adapter only.  It does not calculate any
        actuator command and therefore keeps the control law outside the
        vehicle bridge.
        """
        self._require_initialized()
        if not -1.0 <= normalized_steering <= 1.0:
            raise ValueError("normalized_steering must be in [-1, 1].")
        for name, value in (
            ("throttle", throttle),
            ("braking", braking),
            ("clutch", clutch),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        assert self._vehicle_module is not None
        driver_input = self._vehicle_module.DriverInputs()
        driver_input.m_steering = float(normalized_steering)
        driver_input.m_throttle = float(throttle)
        driver_input.m_braking = float(braking)
        driver_input.m_clutch = float(clutch)
        return driver_input

    def create_irrlicht_visual_system(
        self,
        *,
        window_title: str,
        width: int = 1280,
        height: int = 720,
        chase_distance_m: float = 7.0,
        chase_height_m: float = 1.5,
    ) -> Any:
        """Attach a real Irrlicht vehicle visualizer to this plant.

        Visual assets and the visual system are deliberately opt-in so the
        existing headless experiments retain their exact behavior.
        """
        self._require_initialized()
        if width <= 0 or height <= 0:
            raise ValueError("Irrlicht window dimensions must be positive.")
        if chase_distance_m <= 0.0 or chase_height_m <= 0.0:
            raise ValueError("Chase-camera dimensions must be positive.")

        import pychrono.irrlicht  # noqa: F401 - validates the Irrlicht binding

        assert self._chrono is not None
        assert self._vehicle_module is not None
        assert self._vehicle is not None
        assert self._terrain is not None

        self._vehicle.SetChassisVisualizationType(
            self._chrono.VisualizationType_MESH
        )
        self._vehicle.SetSuspensionVisualizationType(
            self._chrono.VisualizationType_PRIMITIVES
        )
        self._vehicle.SetSteeringVisualizationType(
            self._chrono.VisualizationType_PRIMITIVES
        )
        self._vehicle.SetWheelVisualizationType(self._chrono.VisualizationType_MESH)
        self._vehicle.SetTireVisualizationType(self._chrono.VisualizationType_MESH)

        visual_system = self._vehicle_module.ChWheeledVehicleVisualSystemIrrlicht()
        visual_system.SetWindowTitle(str(window_title))
        visual_system.SetWindowSize(int(width), int(height))
        visual_system.SetChaseCamera(
            self._chrono.ChVector3d(0.0, 0.0, 0.75),
            float(chase_distance_m),
            float(chase_height_m),
        )
        visual_system.SetStepsize(float(self.config.physics_dt_s))
        visual_system.Initialize()
        visual_system.AddLightDirectional()
        visual_system.AddSkyBox()
        visual_system.AttachVehicle(self._vehicle)
        visual_system.AttachTerrain(self._terrain)
        visual_system.EnableStats(True)
        return visual_system

    def synchronize_irrlicht_visual_system(
        self,
        visual_system: Any,
        *,
        normalized_steering: float,
        throttle: float,
        braking: float,
        clutch: float = 0.0,
    ) -> None:
        """Synchronize an attached visual system with current plant inputs."""
        self._require_initialized()
        assert self._vehicle is not None
        driver_inputs = self.make_driver_inputs(
            normalized_steering=normalized_steering,
            throttle=throttle,
            braking=braking,
            clutch=clutch,
        )
        visual_system.Synchronize(
            float(self._vehicle.GetSystem().GetChTime()), driver_inputs
        )

    def get_front_wheel_angles_rad(self) -> tuple[float, float]:
        """Return actual front-left and front-right road-wheel angles in rad.

        Chrono defines positive values as a turn to the left.  The values are
        measured at the front axle (index 0), not inferred from the command.
        """
        self._require_initialized()
        assert self._vehicle is not None
        assert self._vehicle_module is not None
        return (
            float(self._vehicle.GetSteeringAngle(0, self._vehicle_module.LEFT)),
            float(self._vehicle.GetSteeringAngle(0, self._vehicle_module.RIGHT)),
        )

    def set_replay_initial_chassis_state(
        self,
        *,
        vx_body_mps: float,
        vy_body_mps: float,
        yaw_rate_body_radps: float,
    ) -> None:
        """Set only the initial lateral replay state of the chassis.

        Stanford replay uses measured ``vx``, ``vy``, and yaw rate as an
        initial condition so that Chrono and the Dynamic Bicycle plant start
        from the same lateral state.  This is intentionally not a vehicle
        controller and it does not apply steering, throttle, braking, or an
        engine-torque command.
        """
        self._require_initialized()
        self._validate_finite(
            vx_body_mps=vx_body_mps,
            vy_body_mps=vy_body_mps,
            yaw_rate_body_radps=yaw_rate_body_radps,
        )
        assert self._chrono is not None
        assert self._vehicle is not None

        body = self._vehicle.GetChassisBody()
        body_velocity_local = self._chrono.ChVector3d(
            float(vx_body_mps), float(vy_body_mps), 0.0
        )
        body.SetPosDt(body.TransformDirectionLocalToParent(body_velocity_local))
        body.SetAngVelLocal(
            self._chrono.ChVector3d(0.0, 0.0, float(yaw_rate_body_radps))
        )

    def prescribe_replay_longitudinal_speed(self, vx_body_mps: float) -> None:
        """Impose the exogenous longitudinal input used by lateral replay.

        The Dynamic Bicycle plant accepts ``u = [delta_road, vx]`` directly.
        To give the Chrono plant the identical longitudinal input without a
        separate speed controller, this method updates only the chassis
        body-frame longitudinal velocity before a physics step.  It preserves
        the simulated lateral and vertical body velocities and uses no
        throttle, brake, clutch, or engine-torque command.
        """
        self._require_initialized()
        self._validate_finite(vx_body_mps=vx_body_mps)
        if vx_body_mps <= 0.0:
            raise ValueError("Replay longitudinal speed must be positive.")
        assert self._chrono is not None
        assert self._vehicle is not None

        body = self._vehicle.GetChassisBody()
        previous_local_velocity = body.TransformDirectionParentToLocal(
            body.GetPosDt()
        )
        prescribed_local_velocity = self._chrono.ChVector3d(
            float(vx_body_mps),
            float(previous_local_velocity.y),
            float(previous_local_velocity.z),
        )
        body.SetPosDt(
            body.TransformDirectionLocalToParent(prescribed_local_velocity)
        )

    def advance_lateral_replay(
        self,
        dt_s: float,
        *,
        normalized_steering: float,
        vx_body_mps: float,
    ) -> None:
        """Advance the plant using only ``[normalized_steering, vx]``.

        This narrow replay interface deliberately omits the powertrain driver
        channels.  It first imposes the exogenous ``vx`` input, then advances
        the vehicle with steering only; ``advance`` consequently retains its
        zero default throttle, braking, and clutch values.
        """
        self.prescribe_replay_longitudinal_speed(vx_body_mps)
        self.advance(dt_s, normalized_steering=normalized_steering)

    def get_raw_state(self) -> ChronoRawState:
        """Read world pose/velocity and body-frame lateral/yaw quantities."""
        self._require_initialized()
        assert self._vehicle is not None

        body = self._vehicle.GetChassisBody()
        position = body.GetPos()
        world_velocity = body.GetPosDt()
        body_velocity = body.TransformDirectionParentToLocal(world_velocity)
        body_angular_velocity = body.GetAngVelLocal()
        cardan_xyz = body.GetRot().GetCardanAnglesXYZ()

        return ChronoRawState(
            time=float(self._vehicle.GetSystem().GetChTime()),
            X=float(position.x),
            Y=float(position.y),
            psi=float(cardan_xyz.z),
            vx_world=float(world_velocity.x),
            vy_world=float(world_velocity.y),
            vx_body=float(body_velocity.x),
            vy_body=float(body_velocity.y),
            yaw_rate_body=float(body_angular_velocity.z),
        )

    def get_mass_properties(self) -> ChronoVehicleMassProperties:
        """Return aggregate initialized-vehicle mass, COM, and inertia evidence."""
        self._require_initialized()
        assert self._vehicle is not None

        inertia = self._vehicle.GetInertia().to_numpy()
        com = self._vehicle.GetCOMFrame().GetPos()
        return ChronoVehicleMassProperties(
            mass_kg=float(self._vehicle.GetMass()),
            com_x_m=float(com.x),
            com_y_m=float(com.y),
            com_z_m=float(com.z),
            inertia_xx_kg_m2=float(inertia[0, 0]),
            inertia_yy_kg_m2=float(inertia[1, 1]),
            inertia_zz_kg_m2=float(inertia[2, 2]),
        )

    def get_wheelbase_m(self) -> float:
        """Return the initialized vehicle's declared wheelbase in metres."""
        self._require_initialized()
        assert self._vehicle is not None
        return float(self._vehicle.GetWheelbase())

    def get_chassis_roll_pitch_rad(self) -> tuple[float, float]:
        """Return chassis roll and pitch in Chrono's body XYZ convention."""
        self._require_initialized()
        assert self._vehicle is not None
        cardan_xyz = self._vehicle.GetChassisBody().GetRot().GetCardanAnglesXYZ()
        return float(cardan_xyz.x), float(cardan_xyz.y)

    def get_chassis_roll_pitch_rates_radps(self) -> tuple[float, float]:
        """Return chassis body-frame roll and pitch angular rates."""
        self._require_initialized()
        assert self._vehicle is not None
        angular_velocity = self._vehicle.GetChassisBody().GetAngVelLocal()
        return float(angular_velocity.x), float(angular_velocity.y)

    def get_chassis_roll_pitch_accelerations_radps2(self) -> tuple[float, float]:
        """Return chassis body-frame roll and pitch angular accelerations."""
        self._require_initialized()
        assert self._vehicle is not None
        angular_acceleration = self._vehicle.GetChassisBody().GetAngAccLocal()
        return float(angular_acceleration.x), float(angular_acceleration.y)

    def get_body_acceleration_mps2(self) -> tuple[float, float, float]:
        """Return chassis-origin acceleration resolved in the chassis frame."""
        self._require_initialized()
        assert self._vehicle is not None
        body = self._vehicle.GetChassisBody()
        acceleration = body.TransformDirectionParentToLocal(body.GetPosDt2())
        return float(acceleration.x), float(acceleration.y), float(acceleration.z)

    def get_aggregate_com_acceleration_body_mps2(
        self,
    ) -> tuple[float, float, float]:
        """Return current aggregate-COM acceleration in chassis axes."""
        self._require_initialized()
        assert self._vehicle is not None
        acceleration = self._vehicle.GetPointAcceleration(
            self._vehicle.GetCOMFrame().GetPos()
        )
        return float(acceleration.x), float(acceleration.y), float(acceleration.z)

    def get_chassis_reference_world_m(self) -> tuple[float, float, float]:
        """Return the chassis reference-frame origin in world axes."""
        self._require_initialized()
        assert self._vehicle is not None
        position = self._vehicle.GetChassisBody().GetFrameRefToAbs().GetPos()
        return float(position.x), float(position.y), float(position.z)

    def get_chassis_com_world_m(self) -> tuple[float, float, float]:
        """Return the rigid chassis component COM in world axes."""
        self._require_initialized()
        assert self._vehicle is not None
        position = self._vehicle.GetChassisBody().GetFrameCOMToAbs().GetPos()
        return float(position.x), float(position.y), float(position.z)

    def get_aggregate_com_world_m(self) -> tuple[float, float, float]:
        """Return the current aggregate vehicle COM position in world axes."""
        self._require_initialized()
        assert self._vehicle is not None
        body = self._vehicle.GetChassisBody()
        com_world = body.GetFrameRefToAbs().TransformPointLocalToParent(
            self._vehicle.GetCOMFrame().GetPos()
        )
        return float(com_world.x), float(com_world.y), float(com_world.z)

    def get_wheel_tracks_m(self) -> tuple[float, float]:
        """Return initialized front and rear wheel tracks."""
        self._require_initialized()
        assert self._vehicle is not None
        return float(self._vehicle.GetWheeltrack(0)), float(self._vehicle.GetWheeltrack(1))

    def get_suspension_diagnostics(
        self,
    ) -> tuple[ChronoSuspensionDiagnostic, ...]:
        """Return read-only double-wishbone travel and contact evidence."""
        self._require_initialized()
        assert self._vehicle is not None
        assert self._vehicle_module is not None

        vehicle = self._vehicle
        module = self._vehicle_module
        chassis_frame = vehicle.GetChassisBody().GetFrameRefToAbs()
        ids = (("FL", "FR"), ("RL", "RR"))
        sides = (("left", module.LEFT), ("right", module.RIGHT))
        wheel_forces = {item.wheel_id: item.force_z_tire_N for item in self.get_wheel_diagnostics()}
        records: list[ChronoSuspensionDiagnostic] = []
        for axle_index, axle in enumerate(vehicle.GetAxles()):
            suspension = module.CastToChDoubleWishbone(axle.m_suspension)
            track = float(vehicle.GetWheeltrack(axle_index))
            for side_index, (side_name, side_enum) in enumerate(sides):
                wheel = axle.GetWheels()[side_index]
                tire = wheel.GetTire()
                force_element = module.CastToChForceElementTire(tire)
                contact_data = force_element.ReportTireContactData()
                spindle_world = vehicle.GetSpindlePos(axle_index, side_enum)
                spindle = chassis_frame.TransformPointParentToLocal(spindle_world)
                wheel_id = ids[axle_index][side_index]
                records.append(
                    ChronoSuspensionDiagnostic(
                        wheel_id=wheel_id,
                        axle_index=axle_index,
                        side=side_name,
                        track_m=track,
                        spindle_x_m=float(spindle.x),
                        spindle_y_m=float(spindle.y),
                        spindle_z_m=float(spindle.z),
                        spring_length_m=float(suspension.GetSpringLength(side_enum)),
                        spring_deformation_m=float(
                            suspension.GetSpringDeformation(side_enum)
                        ),
                        spring_force_N=float(suspension.GetSpringForce(side_enum)),
                        shock_length_m=float(suspension.GetShockLength(side_enum)),
                        shock_velocity_mps=float(
                            suspension.GetShockVelocity(side_enum)
                        ),
                        tire_deflection_m=float(tire.GetDeflection()),
                        tire_effective_radius_m=float(tire.GetRadius()),
                        contact=bool(contact_data.in_contact),
                        contact_depth_m=max(0.0, float(contact_data.depth)),
                        contact_normal_force_N=max(
                            0.0, float(contact_data.normal_force)
                        ),
                    )
                )
        return tuple(records)

    def initialize_wheel_angular_speeds(
        self, vx_body_mps: float
    ) -> tuple[ChronoWheelSpeedInitialization, ...]:
        """Set rolling-consistent spindle speeds once before replay.

        Each wheel uses its tire model's current effective rolling radius.  The
        write is limited to spindle-local Y angular velocity and is intended
        only for initialization/settling, never continuous replay tracking.
        """
        self._require_initialized()
        self._validate_finite(vx_body_mps=vx_body_mps)
        if vx_body_mps < 0.0:
            raise ValueError("Wheel-speed initialization requires non-negative vx.")
        assert self._chrono is not None
        assert self._vehicle is not None

        records: list[ChronoWheelSpeedInitialization] = []
        ids = (("FL", "FR"), ("RL", "RR"))
        for axle_index, axle in enumerate(self._vehicle.GetAxles()):
            for wheel_index, wheel in enumerate(axle.GetWheels()):
                radius = float(wheel.GetTire().GetRadius())
                if not math.isfinite(radius) or radius <= 0.0:
                    raise RuntimeError(
                        f"Wheel {ids[axle_index][wheel_index]} has invalid "
                        f"effective rolling radius {radius}."
                    )
                requested = float(vx_body_mps) / radius
                spindle = wheel.GetSpindle()
                existing = spindle.GetAngVelLocal()
                spindle.SetAngVelLocal(
                    self._chrono.ChVector3d(float(existing.x), requested, float(existing.z))
                )
                records.append(
                    ChronoWheelSpeedInitialization(
                        wheel_id=ids[axle_index][wheel_index],
                        effective_radius_m=radius,
                        requested_omega_radps=requested,
                        actual_omega_radps=float(wheel.GetState().omega),
                    )
                )
        return tuple(records)

    def get_wheel_diagnostics(self) -> tuple[ChronoWheelDiagnostic, ...]:
        """Return a read-only four-wheel TMeasy diagnostic snapshot.

        This method reports states already computed by Chrono.  It neither
        advances the system nor changes tire, wheel, chassis, or driver state.
        """
        self._require_initialized()
        assert self._chrono is not None
        assert self._vehicle is not None
        assert self._vehicle_module is not None
        assert self._terrain is not None

        body = self._vehicle.GetChassisBody()
        records: list[ChronoWheelDiagnostic] = []
        ids = (("FL", "FR"), ("RL", "RR"))
        sides = (
            ("left", self._vehicle_module.LEFT),
            ("right", self._vehicle_module.RIGHT),
        )
        for axle_index, axle in enumerate(self._vehicle.GetAxles()):
            wheels = axle.GetWheels()
            if len(wheels) != 2:
                raise RuntimeError(
                    f"Axle {axle_index} exposes {len(wheels)} wheels; "
                    "four-wheel diagnostics require one left and one right wheel."
                )
            for wheel_index, ((side_name, side_enum), wheel) in enumerate(
                zip(sides, wheels, strict=True)
            ):
                tire = wheel.GetTire()
                tmeasy = self._vehicle_module.CastToChTMeasyTire(tire)
                tire_frame = self._chrono.ChCoordsysd()
                local_force = tire.ReportTireForceLocal(self._terrain, tire_frame).force
                global_force = tire.ReportTireForce(self._terrain).force
                body_force = body.TransformDirectionParentToLocal(global_force)
                internal_alpha = float(tmeasy.GetSlipAngle_internal())
                records.append(
                    ChronoWheelDiagnostic(
                        wheel_id=ids[axle_index][wheel_index],
                        axle_index=axle_index,
                        side=side_name,
                        slip_angle_rad=float(tire.GetSlipAngle()),
                        tmeasy_slip_angle_rad=internal_alpha,
                        lateral_slip_sy=float(-math.tan(internal_alpha)),
                        longitudinal_slip_sx=float(
                            tmeasy.GetLongitudinalSlip_internal()
                        ),
                        force_x_tire_N=float(local_force.x),
                        force_y_tire_N=float(local_force.y),
                        force_z_tire_N=float(local_force.z),
                        force_x_body_N=float(body_force.x),
                        force_y_body_N=float(body_force.y),
                        force_z_body_N=float(body_force.z),
                        wheel_omega_radps=float(wheel.GetState().omega),
                        effective_radius_m=float(tire.GetRadius()),
                        road_wheel_angle_rad=float(
                            self._vehicle.GetSteeringAngle(axle_index, side_enum)
                        ),
                    )
                )
        return tuple(records)

    def _import_pychrono(self) -> tuple[Any, Any]:
        release_dir = str(self.config.chrono_release_dir)
        if release_dir not in sys.path:
            sys.path.insert(0, release_dir)
        if hasattr(os, "add_dll_directory"):
            self._dll_directory_handle = os.add_dll_directory(release_dir)

        import pychrono as chrono
        import pychrono.vehicle as vehicle_module

        return chrono, vehicle_module

    @contextmanager
    def _runtime_vehicle_json(self) -> Iterator[Path]:
        """Yield a Chrono-resolvable top-level JSON without changing sources."""
        if self.config.vehicle_model == "Sedan":
            yield self.config.vehicle_json
            return

        payload = json.loads(self.config.vehicle_json.read_text(encoding="utf-8"))
        if self.config.vehicle_model == "GolfLike3D_v1":
            component_paths = {
                "chassis": self.config.golf_like_3d_chassis_json,
                "suspension": self.config.golf_like_3d_suspension_json,
                "front_arb": self.config.golf_like_3d_front_antirollbar_json,
                "rear_arb": self.config.golf_like_3d_rear_antirollbar_json,
            }
            try:
                relative = {
                    name: Path(os.path.relpath(path, self.config.vehicle_data_dir)).as_posix()
                    for name, path in component_paths.items()
                }
            except ValueError as exc:
                raise ValueError(
                    "GolfLike3D_v1 and Chrono VehicleDataPath must be on the "
                    "same Windows drive so Chrono can resolve derived JSON files."
                ) from exc
            payload["Chassis"]["Input File"] = relative["chassis"]
            for axle_index, axle in enumerate(payload["Axles"]):
                axle["Suspension Input File"] = relative["suspension"]
                axle["Antirollbar Input File"] = relative[
                    "front_arb" if axle_index == 0 else "rear_arb"
                ]
        else:
            try:
                relative_chassis = os.path.relpath(
                    self.config.golf_like_chassis_json,
                    self.config.vehicle_data_dir,
                )
            except ValueError as exc:
                raise ValueError(
                    "GolfLike model and Chrono VehicleDataPath must be on the same "
                    "Windows drive so Chrono can resolve the derived chassis JSON."
                ) from exc
            payload["Chassis"]["Input File"] = Path(relative_chassis).as_posix()

        with tempfile.TemporaryDirectory(prefix="chrono_golf_like_") as temp_dir:
            runtime_json = Path(temp_dir) / f"{self.config.vehicle_model}.runtime.json"
            runtime_json.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            yield runtime_json

    @staticmethod
    def _trailing_separator(path: Path) -> str:
        return f"{path}{os.sep}"

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("Call initialize() before advancing or reading state.")

    @staticmethod
    def _validate_finite(**values: float) -> None:
        invalid = [name for name, value in values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Replay state values must be finite: {invalid}")
