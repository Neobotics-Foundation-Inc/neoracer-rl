"""
Put more than one NeoRacer on the same track.

The simulator attaches a car to a track by grafting its "car" body into the
track's spec. MjSpec.attach_body takes a name prefix, so attaching the same car
file twice under different prefixes gives two complete, independent cars in one
model: a_motor and b_motor, a_imu_accel and b_imu_accel, and so on. Nothing in
the simulator looks a part up by a global name, it binds spec elements, so each
car's parts bind to the same model without clashing.

    scene = build("loop_corridor", ("a", "b"))
    data = mujoco.MjData(scene.model)
    drive(scene, data, "a", throttle=0.3, steer=0.1)
    obs = observe(scene, data, "a")        # the same 20 numbers NeoRacerEnv gives

Each car also gets its own truth sensors, named with its prefix, because the
simulator's truth.add_sensors writes one fixed set of names and two cars would
collide. The sensors here are built the same way as that function builds them.

Frame: +X forward, +Y left, +Z up.
"""

from dataclasses import dataclass, fields

import mujoco
import numpy as np
from neoracer_mujoco import assets, contract, sensors, sim, truth
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import Car
from neoracer_rl.envs import neoracer_env

import tracks

# How far apart two cars start along the track, in metres. A NeoRacer is
# 0.40 m long, so 1.2 m leaves room to accelerate without an immediate nudge.
STAGGER_M = 1.2
# Half the gap between two cars placed side by side, in metres. The corridor
# is 1.0 m wide and the car is 0.235 m across, so 0.22 keeps both off the walls.
ABREAST_M = 0.22


@dataclass(frozen=True)
class MultiScene:
    """A compiled track with several cars on it, each found by its prefix."""

    spec: mujoco.MjSpec
    model: mujoco.MjModel
    cars: dict[str, Car]
    limits: dict[str, contract.ActuatorLimits]

    @property
    def prefixes(self) -> tuple[str, ...]:
        return tuple(self.cars)


def _add_truth(spec: mujoco.MjSpec, prefix: str, body: str) -> truth.TruthSensors:
    """Truth sensors for one car, named with its prefix.

    truth.add_sensors always writes truth_pos, truth_zaxis and
    truth_wall_contact, so a second car would reuse the first car's sensors.
    These are the same three sensors with the prefix in front of each name.
    """
    xbody = mujoco.mjtObj.mjOBJ_XBODY
    pos = spec.add_sensor(
        name=f"{prefix}_{truth.POS}",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=xbody,
        objname=body,
    )
    zaxis = spec.add_sensor(
        name=f"{prefix}_{truth.ZAXIS}",
        type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
        objtype=xbody,
        objname=body,
    )
    wall_contact = spec.add_sensor(
        name=f"{prefix}_{truth.WALL_CONTACT}",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=xbody,
        objname=body,
        reftype=mujoco.mjtObj.mjOBJ_BODY,
        refname=truth.WALLS_BODY,
        intprm=[1 << mujoco.mjtConDataField.mjCONDATA_FOUND.value, 0, 1],
    )
    return truth.TruthSensors(pos=pos, zaxis=zaxis, wall_contact=wall_contact)


def _find_car(spec: mujoco.MjSpec, model: mujoco.MjModel, prefix: str) -> Car:
    """The parts of the car attached under prefix.

    This is neoracer_mujoco.scene.find_car with the prefix in front of every
    name. It stays here rather than upstream because only this folder attaches
    more than one car.
    """
    name = lambda part: f"{prefix}_{part}"  # noqa: E731
    body = spec.body(name("car"))
    if body is None:
        raise ValueError(f"no car body called {name('car')!r}")
    joints = body.joints
    if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{name('car')} needs exactly one free joint")
    car_sensors = sensors.CarSensors(
        **{f.name: spec.sensor(name(f.name)) for f in fields(sensors.SensorReadings)}
    )
    wheels = tuple(spec.geom(name(w)) for w in contract.WHEEL_GEOMS)
    return Car(
        body=body,
        joint=joints[0],
        drive=spec.actuator(name(contract.DRIVE_ACTUATOR)),
        steer=spec.actuator(name(contract.STEER_ACTUATOR)),
        wheels=wheels,
        lidar=spec.site(name(contract.LIDAR_SITE)),
        sensors=car_sensors,
        truth=truth.TruthSensors(
            pos=spec.sensor(name(truth.POS)),
            zaxis=spec.sensor(name(truth.ZAXIS)),
            wall_contact=spec.sensor(name(truth.WALL_CONTACT)),
        ),
    )


def start_positions(
    prefixes: tuple[str, ...], abreast: bool = False
) -> tuple[tuple[float, float, float], ...]:
    """Where each car starts on loop_corridor, in metres.

    The ring's bottom straight runs along y = -2.5, between an inner wall at
    y = -2 and an outer wall at y = -3, and the cars face +X. Staggered, they
    line up one behind the other; abreast, they sit side by side.
    """
    base_y = tracks.SPAWN["loop_corridor"][1]
    if abreast:
        offsets = np.linspace(-ABREAST_M, ABREAST_M, len(prefixes))
        return tuple((0.0, base_y + float(dy), 0.0) for dy in offsets)
    return tuple(
        (-STAGGER_M * i, base_y, 0.0) for i in range(len(prefixes))
    )


def build(
    track: str = "loop_corridor",
    prefixes: tuple[str, ...] = ("a", "b"),
    abreast: bool = False,
) -> MultiScene:
    """Compile track with one car per prefix. See tracks.py for track names."""
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"the prefixes must differ: {prefixes}")

    generated = tracks.parse_generated(track)
    spec = (
        tracks.generated_spec(*generated)
        if generated is not None
        else mujoco.MjSpec.from_file(tracks.track_path(track))
    ).copy()
    if spec.body(truth.WALLS_BODY) is None:
        spec.worldbody.add_body(name=truth.WALLS_BODY)

    for prefix, pos in zip(prefixes, start_positions(prefixes, abreast), strict=True):
        car = mujoco.MjSpec.from_file(assets.DEFAULT_CAR)
        frame = spec.worldbody.add_frame()
        frame.pos = np.array(pos)
        frame.attach_body(car.body("car"), f"{prefix}_", "")
        _add_truth(spec, prefix, f"{prefix}_car")

    model = spec.compile()
    cars = {p: _find_car(spec, model, p) for p in prefixes}
    limits = {p: contract.actuator_limits(model, c) for p, c in cars.items()}
    return MultiScene(spec=spec, model=model, cars=cars, limits=limits)


def observe(
    scene: MultiScene, data: mujoco.MjData, prefix: str
) -> np.ndarray:
    """The 20 numbers NeoRacerEnv would give this car. Same layout, same order,
    so a policy trained in that environment reads it unchanged."""
    return neoracer_env.observation(scene.model, scene.cars[prefix], data.sensordata)


def drive(
    scene: MultiScene, data: mujoco.MjData, prefix: str, throttle: float, steer: float
) -> None:
    """Command one car. throttle is in [-1, 1] as a fraction of the battery
    voltage, steer is in [-1, 1] as a fraction of the steering limit, which is
    the same pair of numbers NeoRacerEnv takes as its action."""
    command = ControlCommand(
        throttle=float(np.clip(throttle, -1.0, 1.0)),
        steer=float(np.clip(steer, -1.0, 1.0)) * scene.limits[prefix].steer_rad,
    )
    sim.apply_command(scene.cars[prefix], data, command)


def pose_of(data: mujoco.MjData, car: Car) -> tuple[float, float, float]:
    """A car's x, y (m) and heading (rad, 0 along +X, growing anticlockwise).

    Takes a Car rather than a prefix, so it also reads the single car of a
    NeoRacerEnv through its public scene.
    """
    body = data.bind_scalar(car.body)
    x, y = float(body.xpos[0]), float(body.xpos[1])
    w, qx, qy, qz = (float(v) for v in body.xquat)
    yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, float(yaw)


def pose(scene: MultiScene, data: mujoco.MjData, prefix: str) -> tuple[float, float, float]:
    """The pose of one car in a multi-car scene."""
    return pose_of(data, scene.cars[prefix])


def touching_wall(scene: MultiScene, data: mujoco.MjData, prefix: str) -> bool:
    """True if any part of this car is touching a track wall."""
    sensor = scene.cars[prefix].truth.wall_contact
    return float(data.bind_scalar(sensor).data[0]) > 0.0


def gap_between(
    scene: MultiScene, data: mujoco.MjData, a: str, b: str
) -> float:
    """Straight-line distance between two cars, in metres."""
    ax, ay, _ = pose(scene, data, a)
    bx, by, _ = pose(scene, data, b)
    return float(np.hypot(ax - bx, ay - by))


if __name__ == "__main__":
    scene = build("loop_corridor", ("a", "b"))
    data = mujoco.MjData(scene.model)
    mujoco.mj_forward(scene.model, data)
    print(f"two cars on loop_corridor: nq={scene.model.nq} nu={scene.model.nu} "
          f"nsensor={scene.model.nsensor}")
    for p in scene.prefixes:
        x, y, yaw = pose(scene, data, p)
        obs = observe(scene, data, p)
        print(f"  car {p}: start ({x:+.2f}, {y:+.2f}) heading {np.degrees(yaw):+.1f} deg, "
              f"obs {obs.shape}, first lidar beam {obs[0]:.2f} m")
    for _ in range(90):
        drive(scene, data, "a", 0.3, 0.0)
        drive(scene, data, "b", 0.2, 0.0)
        mujoco.mj_step(scene.model, data)
    print(f"  after 90 steps the gap is {gap_between(scene, data, 'a', 'b'):.2f} m")
