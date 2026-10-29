"""
Gymnasium environment for the NeoRacer MuJoCo car.

This module gives the NeoRacer simulator the Gymnasium Env API. It uses the
neoracer_mujoco modules (assets, batch, sim, sensors, contract, speed_loop)
and adds no physics, sensors or actuator mappings. The classical controllers in
neoracer_mujoco.control drive the same simulator.

Frame: +X forward, +Y left, +Z up (see neoracer-mujoco's assets/neoracer.xml).
Quantities are in SI units unless a name says otherwise.

Track (config.tracks): LoopCorridor, assets/tracks/loop_corridor.xml, or
GeneratedTracks, a pool of tracks from neoracer_mujoco.track_generation. With
a pool, reset() draws the track of each episode. A pool leaves out the
tracks where a spawn could put a car against a wall (arenas_of). Progress is
the angle around the center of the loop on loop_corridor, and the distance
along the centerline on a generated track, as an angle: 2 pi is one lap on
both.

Action space -- Box([-1,-1],[1,1]), float32:
    action[0] = normalized speed, + = forward
    action[1] = normalized steer, + = LEFT

    The signs are the ones in contract.py: + speed = forward, + steer = left.
    racecar_core uses + angle = right, so this env does not use
    racecar_core_adapter.MujocoDrive, which negates the steer. step() clamps
    the action and scales the steer by the steer servo's limit
    (contract.actuator_limits). The car's computer commands a speed (its
    spec's Command.drive). On the NeoRacer, action[0] asks the firmware's
    speed loop (config.car.speed_loop, neoracer_mujoco.speed_loop) for that
    share of config.max_speed times the loop's max speed, as racecar_core's
    set_speed_angle asks the car, and the loop sets the throttle at the
    firmware's ticks. The RACECAR Neo V2's model holds its own speed (its
    car has no speed loop), so there action[0] times max_speed is the
    throttle, which its model takes as a speed.

The task (task.py): config.observe, config.terminate and config.reward are
functions of a task.Step, what the env knows after a step. The defaults,
CarObservation, LapEnds and ProgressReward, follow.

Observation space -- CarObservation: Box(20,), float32, built only from
sensors.SensorReadings (no raw qpos/qvel). Another config.observe gives an
unbounded Box of its size.
    [0:8]   lidar_ranges   m, -1 = no hit or a bearing the car blocks
            (sensors.lidar_ranges). The model's beams read as the LiDAR's
            driver reports them: the last complete turn of its head, so a
            range is up to two turns old.
    [8:11]  imu_accel      m/s^2, body frame, gravity included
    [11:14] imu_gyro       rad/s, body frame
    [14:18] imu_quat       orientation quaternion (w,x,y,z), world frame
    [18]    steer_cmd_pos  rad, virtual steering input
    [19]    estimated_speed  m/s, from motor_encoder_rad_s: the mean of the
            4 wheel speeds, like the one drive motor of the real car, at the
            car's wheel radius (contract.wheel_radius). It is not the true
            speed imu_linvel.

Control rate: 30 Hz, the target LiDAR scan rate of the real hardware.
__init__ computes the physics steps per action from model.opt.timestep.
step() drives the car with neoracer_mujoco's batch.drive, as NeoRacerBatch
does, on a batch.DataBatch of the env's MjData, so a row of NeoRacerBatch on
the CPU backend steps the same as this env, bit for bit. reset() starts the
speed loop over, its ticks from then on.

Reward (ProgressReward, all weights explicit and UNTUNED):
    r_t = progress_weight * delta_theta_t
    plus one-time terminal adjustments:
        + lap_bonus        on successful lap completion
        - rollover_penalty on rollover termination
        - crash_penalty    on crash termination
        - stuck_penalty    on stuck termination
    delta_theta_t is the signed net change in cumulative angle (radians)
    around the loop's center this tick (see task.ProgressRows). crash_penalty
    equals stuck_penalty, so an episode that makes progress and then crashes
    returns more than one in which the car never moves.

Physics randomization (config.physics, neoracer_mujoco.physics):
    reset() draws a Physics from config.physics, after the spawn offset, and
    the episode steps with it: factors on the wheel friction, the chassis
    mass, the motor torque and the steering stiffness. The default ranges
    change nothing and take no random draw. The car settles at its spawn
    with the physics of the XML, so with another mass its suspension settles
    again in the first steps of the episode.

Termination (terminated=True) vs. truncation (truncated=True):
    terminated: an end of config.terminate. LapEnds's are rollover
    (upright_cos below the threshold), crash (a car geom touched a wall),
    stuck (no net progress over progress_window_s), and a full lap
    (cumulative angle >= 2 pi in the forward direction of the track, which
    is +).
    truncated: the episode reached max_episode_steps.
"""

import copy
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, assert_never

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from neoracer_mujoco import (
    assets,
    batch,
    contract,
    physics,
    sim,
    speed_loop,
)
from neoracer_mujoco.physics import PhysicsRanges
from neoracer_mujoco.racecar import CarParams, neoracer_params
from neoracer_mujoco.scene import Car, Scene
from neoracer_mujoco.sim import SpawnRange
from neoracer_mujoco.track_generation import (
    Difficulty,
    NoValidTrack,
    Track,
    WallParams,
    generate_track,
    lap_distance,
    pose_along,
    settings_for_difficulty,
    to_spec,
)

from .task import (
    CarObservation,
    LapEnds,
    Observe,
    ProgressReward,
    Reward,
    SensorData,
    Step,
    Terminate,
    advance_rows,
    encoder_speed,
    first_end,
    first_step,
    start_rows,
)

_TRACK = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
"""loop_corridor.xml. The progress around the world origin, the lap and the
reward are specific to its geometry."""

WALL_BOXES = 192
"""The wall boxes of a generated track: more than the 161 that any of 240
tracks needed, so that every track of a pool has the same sizes, as a batch
with one track per row needs."""

NEORACER = neoracer_params()
"""The NeoRacer's settings beside its model, from its spec."""

_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)


class EnvInfo(TypedDict):
    """The info dict of reset() and step(). gymnasium requires a dict."""

    progress_rad: float
    """The signed progress in this episode."""
    lap_frac: float
    """progress_rad / 2 pi in laps, negative if the car drove backward."""
    speed_ms: float
    """The speed that the car's motor encoder measures (task.encoder_speed)."""
    upright_cos: float
    """cos(tilt) of the car body."""
    contact: bool
    """A car geom touched a wall in this step."""
    episode_time_s: float
    termination_reason: str | None
    """The end of config.terminate that ended the episode in this step."""


@dataclass(frozen=True)
class LoopCorridor:
    """assets/tracks/loop_corridor.xml, a closed corridor about 1 m wide."""


@dataclass(frozen=True)
class GeneratedTracks:
    """A pool of generated tracks: the first count seeds, from first_seed
    on, that give a track of this difficulty whose start has room for the
    env's cars (arenas_of)."""

    difficulty: Difficulty = 1
    """From 0, gentle and wide, to 3, tight and narrow."""
    count: int = 16
    first_seed: int = 0

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"a pool needs at least one track, not {self.count}")


type Tracks = LoopCorridor | GeneratedTracks


@dataclass(frozen=True)
class NeoRacerEnvConfig:
    """The car, its tracks, the control rate, and the task: what a policy
    sees, when its episode ends and what it earns (task.py)."""

    car_path: str = assets.DEFAULT_CAR
    tracks: Tracks = LoopCorridor()
    control_hz: float = 30.0
    max_speed: float = 1.0
    """The share of the car's full speed that action[0] = 1 asks for, in
    (0, 1], as racecar_core's set_max_speed sets it. The NeoRacer's full speed
    is its speed loop's max speed, more than its motor reaches on the battery
    (about 3.5 m/s), so there action[0] = 1 holds full throttle."""
    car: CarParams = NEORACER
    """The settings of the car of car_path beside its model, from its spec:
    the NeoRacer's. Its speed loop runs beside the model; a car whose model
    holds its own speed has none, as the RACECAR Neo V2's does
    (racecar_neo_v2_params)."""
    max_episode_steps: int = 1800
    """60 s at 30 Hz."""
    spawn: SpawnRange = SpawnRange()
    physics: PhysicsRanges = PhysicsRanges()
    """The range of each factor of the car's physics, drawn at each reset."""
    progress_window_s: float = 4.0
    """How far back a Step's progress.recent looks."""
    observe: Observe = CarObservation()
    terminate: Terminate = LapEnds()
    reward: Reward = ProgressReward()

    def __post_init__(self) -> None:
        if not 0.0 < self.max_speed <= 1.0:
            raise ValueError(f"max_speed must be in (0, 1], not {self.max_speed}")


@dataclass(frozen=True)
class Arena:
    """One track of the env, with the car on it, at the car's spawn."""

    scene: Scene
    layout: LoopCorridor | Track
    """What lap_angle measures progress on."""


def arenas(config: NeoRacerEnvConfig, visuals: bool) -> tuple[Arena, ...]:
    """The tracks of config.tracks, with the car of config at the start, as
    arenas_of gives them."""
    return arenas_of(config, (((0.0, 0.0), 0.0),), visuals)


def arenas_of(
    config: NeoRacerEnvConfig,
    starts: Sequence[tuple[tuple[float, float], float]],
    visuals: bool,
) -> tuple[Arena, ...]:
    """The tracks of config.tracks, with a car of config at each of starts,
    an offset and a heading: the car stands offset[0] m along the track's
    centerline from its start, + forward, and offset[1] m across it, + left,
    turned by heading (rad) from the track's direction there, + to the left
    (pose_along). One car keeps its names
    unprefixed; several are behind the prefixes car0/, car1/, ...
    (assets.compose_cars). On loop_corridor, whose start is on a straight,
    the offsets and headings are in the start's frame.

    A generated track where a spawn of config.spawn could put a car against
    a wall (sim.spawn_clears_walls) is left out, as a seed that gives no
    track is. Raise ValueError if one could on loop_corridor, or if the
    seeds give too few tracks."""
    prefixes = [""] if len(starts) == 1 else [f"car{k}/" for k in range(len(starts))]

    def scene_of(
        spec: mujoco.MjSpec, poses: Sequence[tuple[tuple[float, float], float]]
    ) -> Scene:
        entries = [
            assets.Entry(prefix, offset, heading, config.car_path)
            for prefix, (offset, heading) in zip(prefixes, poses, strict=True)
        ]
        return assets.compose_cars(spec, entries, visuals)

    match config.tracks:
        case LoopCorridor() as loop:
            spec = mujoco.MjSpec.from_file(_TRACK)
            scene = scene_of(spec, starts)
            if not sim.spawn_clears_walls(scene, config.spawn):
                raise ValueError(
                    f"a spawn of {config.spawn} can put a car at {starts} against "
                    "a wall of loop_corridor"
                )
            return (Arena(scene, loop),)
        case GeneratedTracks(difficulty=difficulty, count=count, first_seed=first):
            settings = settings_for_difficulty(difficulty)
            walls = WallParams(boxes_per_wall=WALL_BOXES)
            pool: list[Arena] = []
            for seed in range(first, first + 4 * count):
                match generate_track(seed, settings):
                    case Track() as track:
                        poses = []
                        for offset, heading in starts:
                            xy, turn = pose_along(track, *offset)
                            poses.append((xy, turn + heading))
                        scene = scene_of(to_spec(track, walls), poses)
                        if sim.spawn_clears_walls(scene, config.spawn):
                            pool.append(Arena(scene, track))
                    case NoValidTrack():
                        pass
                    case other:
                        assert_never(other)
                if len(pool) == count:
                    return tuple(pool)
            raise ValueError(
                f"seeds {first} to {first + 4 * count - 1} gave only {len(pool)} "
                f"tracks of difficulty {difficulty} with room for cars at {starts}, "
                f"not {count}"
            )
        case other:
            assert_never(other)


def lap_angle(
    arena: Arena, car: Car, sensordata: SensorData
) -> npt.NDArray[np.float64]:
    """How far around the track car, a car of the arena, is in sensordata
    (..., nsensordata), as an angle (rad), shape (...,). One lap is 2 pi."""
    model = arena.scene.model
    match arena.layout:
        case LoopCorridor():
            return angle_around_center(model, car, sensordata)
        case Track() as track:
            xy = batch.read(model, sensordata, car.truth.pos)[..., :2]
            return 2.0 * np.pi * lap_distance(track, xy) / track.total_length
        case other:
            assert_never(other)


def loop_scene(config: NeoRacerEnvConfig, visuals: bool) -> Scene:
    """The car of config on loop_corridor.xml, at the track's start. Without
    visuals, the car's meshes are left out: the physics and the sensors are
    the same, and the model is smaller and faster (assets.compose). Raise
    ValueError if the car has no free joint or misses an actuator or sensor."""
    return assets.compose(
        mujoco.MjSpec.from_file(_TRACK), config.car_path, visuals=visuals
    )


def substeps_per_action(config: NeoRacerEnvConfig, model: mujoco.MjModel) -> int:
    """The physics steps in one env step: the closest whole number to one
    control period."""
    return max(1, round((1.0 / config.control_hz) / model.opt.timestep))


def progress_window_len(config: NeoRacerEnvConfig) -> int:
    """The totals that a Step's progress keeps: one more than the steps in
    progress_window_s."""
    return max(1, round(config.progress_window_s * config.control_hz)) + 1


def angle_around_center(
    model: mujoco.MjModel, car: Car, sensordata: SensorData
) -> npt.NDArray[np.float64]:
    """The angle (rad) of the car's position around the world origin, the
    center of the loop, shape (...,)."""
    pos = batch.read(model, sensordata, car.truth.pos)
    return np.arctan2(pos[..., 1], pos[..., 0])


class NeoRacerEnv(gym.Env):
    """Gymnasium environment driving a car around the tracks of
    config.tracks.

    __init__ compiles one MjModel, and no method changes it. reset() writes
    only MjData (mj_resetData, settle, spawn perturbation), so one episode
    cannot change the model of the next. All random numbers come from
    self.np_random, the seeded generator of this instance.
    """

    # The same type as gym.Env.metadata, so not a ClassVar.
    metadata: dict[str, Any] = {"render_modes": ["human"], "render_fps": 30}  # noqa: RUF012

    def __init__(
        self,
        config: NeoRacerEnvConfig = NeoRacerEnvConfig(),
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        self._arenas = arenas(config, visuals=render_mode is not None)
        self._arena = self._arenas[0]
        self.scene = self._arena.scene
        """The track of the episode. scene.car holds the car's parts as spec
        elements, which the env binds to self.data where it reads or writes
        them. scene.model keeps the physics of the XML."""
        self.model = copy.deepcopy(self.scene.model)
        """scene.model with the physics of the episode."""
        self.data = mujoco.MjData(self.model)
        """The simulation's state. All tracks have the same sizes, so it fits
        every one."""
        self.physics = physics.Physics()
        self._car = self.scene.car

        self._substeps_per_action = substeps_per_action(config, self.model)
        self.effective_control_dt = self._substeps_per_action * float(
            self.model.opt.timestep
        )
        """The true control period, a whole number of physics steps. It can
        differ slightly from 1/control_hz."""

        self._limits = contract.actuator_limits(self.model, self._car)
        self.loop: speed_loop.SpeedLoopState | None = None
        """The car's speed loop (config.car.speed_loop) in the episode, a
        batch of one row. reset() starts it over."""

        self.action_space = spaces.Box(
            low=_ACTION_LOW, high=_ACTION_HIGH, dtype=np.float32
        )
        self._window_len = progress_window_len(config)
        self._progress = start_rows(np.zeros(1), self._window_len)
        """The episode's progress, a batch of one row."""
        match config.observe:
            case CarObservation() as observe:
                self.observation_space = observe.space()
            case observe:
                start = sim.settle(self.scene.model)
                angle = lap_angle(self._arena, self._car, start.sensordata[None])
                step = first_step(
                    self.model,
                    self._car,
                    start.sensordata[None],
                    start_rows(angle, self._window_len),
                )
                shape = observe(step).shape[1:]
                self.observation_space = spaces.Box(-np.inf, np.inf, shape, np.float32)

        self._viewer = None
        self._viewer_model = self.model

    # --- internal helpers -----------------------------------------------

    def _lap_angle(self) -> float:
        return float(lap_angle(self._arena, self._car, self.data.sensordata))

    def _drive_angle(self) -> float:
        """The drivetrain's angle (rad) now, at the last step's end."""
        return float(batch.drive_angle(self.model, self._car, self.data.qpos))

    def _drive(self, speed: float, steer: float) -> bool:
        """Step one control period with speed (action[0]) and steer (rad)
        commanded, the car's speed loop, if it has one, holding the speed.
        Return whether a car geom touched a wall."""
        model, car, n = self.model, self._car, self._substeps_per_action
        one, steers = batch.DataBatch(model), np.array([steer])
        params, loop = self.config.car.speed_loop, self.loop
        if params is None or loop is None:
            throttle = np.array([speed * self.config.max_speed])
            ctrl = batch.command(model, car, throttle, steers)
            _, sensordata = one.step(self.data, ctrl, n)
        else:
            commanded = np.asarray(
                speed_loop.target(params, np.array([speed]), self.config.max_speed)
            )
            _, sensordata, self.loop = batch.drive(
                one, model, car, params, self.data, loop, commanded, steers, n
            )
        return bool((batch.read(model, sensordata, car.truth.wall_contact) > 0).any())

    def _observe(self, step: Step) -> npt.NDArray[np.float32]:
        return np.asarray(self.config.observe(step)[0], np.float32)

    def _info(self, step: Step, reason: str | None) -> EnvInfo:
        total = float(step.progress.total[0])
        return {
            "progress_rad": total,
            "lap_frac": total / (2.0 * np.pi),
            "speed_ms": float(encoder_speed(step)[0]),
            "upright_cos": float(step.read(self._car.truth.zaxis)[0, 2]),
            "contact": bool(step.contact[0]),
            "episode_time_s": int(step.progress.steps[0]) * self.effective_control_dt,
            "termination_reason": reason,
        }

    # --- Gymnasium API ----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        if len(self._arenas) > 1:  # one track takes no random draw
            self._arena = self._arenas[int(self.np_random.integers(len(self._arenas)))]
        if self._arena.scene is not self.scene:
            self.scene, self._car = self._arena.scene, self._arena.scene.car
            self.model = copy.deepcopy(self.scene.model)
        nominal = self.scene.model
        sim.settle(nominal, self.data)
        sim.perturb_spawn(
            nominal, self.data, self._car, self.np_random, self.config.spawn
        )
        self.physics = physics.sample(self.config.physics, self.np_random)
        physics.apply_physics(self.model, self.scene, self.physics)
        params = self.config.car.speed_loop
        if params is not None:
            self.loop = speed_loop.start(params, np.array([self._drive_angle()]))

        self._progress = start_rows(np.array([self._lap_angle()]), self._window_len)
        step = first_step(
            self.model, self._car, self.data.sensordata[None].copy(), self._progress
        )
        obs, info = self._observe(step), self._info(step, None)
        return obs, info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), _ACTION_LOW, _ACTION_HIGH
        )
        steer = float(action[1]) * self._limits.steer_rad
        contact = self._drive(float(action[0]), steer)
        self._progress = advance_rows(self._progress, np.array([self._lap_angle()]))
        step = Step(
            self.model,
            self._car,
            self.data.sensordata[None].copy(),
            np.array([contact]),
            self._progress,
            action[None],
        )
        codes, ends = first_end(self.config.terminate(step), 1)
        reason = list(ends)[codes[0] - 1] if codes[0] else None
        terminated = reason is not None
        truncated = (not terminated) and int(
            self._progress.steps[0]
        ) >= self.config.max_episode_steps
        step_reward = float(self.config.reward(step, ends)[0])
        obs, info = self._observe(step), self._info(step, reason)
        return obs, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is not None and self._viewer_model is not self.model:
            self._viewer.close()  # a new track: show its model
            self._viewer = None
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer_model = self.model
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
