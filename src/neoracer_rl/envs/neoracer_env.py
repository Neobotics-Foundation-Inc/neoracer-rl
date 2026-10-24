"""
Gymnasium environment for the NeoRacer MuJoCo car.

This module gives the NeoRacer simulator the Gymnasium Env API. It uses the
neoracer_mujoco modules (assets, sim, sensors, collision, contract) and adds
no physics, sensors or actuator mappings. The classical controllers in
neoracer_mujoco.control drive the same simulator.

Frame: +X forward, +Y left, +Z up (see neoracer-mujoco's assets/neoracer.xml).
Quantities are in SI units unless a name says otherwise.

Track (config.tracks): LoopCorridor, assets/tracks/loop_corridor.xml, or
GeneratedTracks, a pool of tracks from neoracer_mujoco.track_generation. With
a pool, reset() draws the track of each episode. Progress is the angle around
the center of the loop on loop_corridor, and the distance along the
centerline on a generated track, as an angle: 2 pi is one lap on both.

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

Observation space -- Box(20,), float32, built only from sensors.SensorReadings
(no raw qpos/qvel):
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
Each step, and each tick of the speed loop, starts MuJoCo's constraint
solver cold (qacc_warmstart = 0), as mujoco.rollout does at the start of each
piece of batch.drive, so a row of NeoRacerBatch on the CPU backend steps the
same as this env, bit for bit. reset() starts the speed loop over, its ticks
from then on.

Reward (RewardConfig, all weights explicit and UNTUNED, see reward):
    r_t = progress_weight * delta_theta_t
    plus one-time terminal adjustments:
        + lap_bonus        on successful lap completion
        - rollover_penalty on rollover termination
        - crash_penalty    on crash termination
        - stuck_penalty    on stuck termination
    delta_theta_t is the signed net change in cumulative angle (radians)
    around the loop's center this tick (see Progress). crash_penalty equals
    stuck_penalty, so an episode that makes progress and then crashes
    returns more than one in which the car never moves.

Physics randomization (config.physics, neoracer_mujoco.physics):
    reset() draws a Physics from config.physics, after the spawn offset, and
    the episode steps with it: factors on the wheel friction, the chassis
    mass, the motor torque and the steering stiffness. The default ranges
    change nothing and take no random draw. The car settles at its spawn
    with the physics of the XML, so with another mass its suspension settles
    again in the first steps of the episode.

Termination (terminated=True) vs. truncation (truncated=True):
    terminated: rollover (upright_cos below the threshold), crash (a car
    geom touched a wall), stuck (no net progress in a window of time), or a
    full lap (cumulative angle >= 2 pi in the forward direction of the
    track, which is +). See termination.
    truncated: the episode reached max_episode_steps.
"""

import copy
import os
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, assert_never

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from neoracer_mujoco import (
    assets,
    batch,
    collision,
    contract,
    physics,
    sensors,
    sim,
    speed_loop,
)
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.physics import PhysicsRanges
from neoracer_mujoco.racecar_core_adapter import CarParams, neoracer_params
from neoracer_mujoco.scene import Car, Scene
from neoracer_mujoco.sim import SpawnRange
from neoracer_mujoco.track_generation import (
    Difficulty,
    NoValidTrack,
    Track,
    WallParams,
    generate_track,
    lap_distance,
    settings_for_difficulty,
    shortest_loop_distance,
    to_spec,
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

OBS_DIM = 20
type Observation = npt.NDArray[np.float32]
"""Shape (OBS_DIM,), in the layout of the module docstring."""
type SensorData = npt.NDArray[np.float64]
"""Shape (..., nsensordata)."""

_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)

type TerminationReason = Literal["rollover", "crash", "stuck", "lap_complete"]


class EnvInfo(TypedDict):
    """The info dict of reset() and step(). gymnasium requires a dict."""

    progress_rad: float
    """The signed progress in this episode."""
    lap_frac: float
    """progress_rad / 2 pi in laps, negative if the car drove backward."""
    speed_ms: float
    """estimated_speed, observation [19]."""
    upright_cos: float
    """cos(tilt) of the car body."""
    contact: bool
    """A car geom touched a wall in this step."""
    episode_time_s: float
    termination_reason: TerminationReason | None


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights. Nobody has tuned them yet. See the module docstring for
    the formula."""

    progress_weight: float = 1.0
    """Reward per radian of signed progress around the track."""
    rollover_penalty: float = 10.0
    """One-time penalty on rollover termination."""
    crash_penalty: float = 5.0
    """One-time penalty on crash termination. It equals stuck_penalty, so a
    car that tries the first corner and hits the wall returns more than a car
    that does not move."""
    stuck_penalty: float = 5.0
    """One-time penalty on stuck termination."""
    lap_bonus: float = 10.0
    """One-time bonus on a completed lap."""


@dataclass(frozen=True)
class TerminationConfig:
    """Thresholds that end an episode. See termination."""

    upright_rollover_cos: float = 0.5
    """The car has rolled over when cos(tilt) is below this."""
    stuck_window_s: float = 4.0
    stuck_progress_rad: float = 0.05
    """The car is stuck when its net progress over the window is less."""


@dataclass(frozen=True)
class ObservationBounds:
    """Bounds of the observation space, with a margin over the peaks in 160
    episodes on loop_corridor.xml at full throttle at the full battery voltage
    with random steering, each run to its end (all end in a wall crash), with
    the elliptic friction cone and impratio=10.
    If the track, the spawn range or the car changes, measure the peaks again."""

    lidar_m: float = 50.0
    accel_m_s2: float = 1000.0
    """The peak, in a crash, is about 290 m/s^2."""
    gyro_rad_s: float = 100.0
    """The peak is about 7 rad/s."""
    steer_rad: float = 2.0
    """The peak is about 0.43 rad. MuJoCo's 0.45 rad joint limit is soft, so
    a crash could push past it."""
    speed_m_s: float = 40.0
    """The same limit as test_top_speed_bounded in neoracer-mujoco's
    validation/test_conformance.py. The peak is about 3.5 m/s; the motor's
    free speed is 4.3 m/s."""


@dataclass(frozen=True)
class LoopCorridor:
    """assets/tracks/loop_corridor.xml, a closed corridor about 1 m wide."""


@dataclass(frozen=True)
class GeneratedTracks:
    """A pool of generated tracks: the first count seeds, from first_seed
    on, that give a track of this difficulty."""

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
    """The car, its tracks, the control rate, the reward and the ends of an
    episode."""

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
    obs_bounds: ObservationBounds = ObservationBounds()
    spawn: SpawnRange = SpawnRange()
    termination: TerminationConfig = TerminationConfig()
    reward: RewardConfig = RewardConfig()
    physics: PhysicsRanges = PhysicsRanges()
    """The range of each factor of the car's physics, drawn at each reset."""

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
    """The tracks of config.tracks, with the car of config. Raise ValueError
    if the seeds give too few generated tracks."""
    match config.tracks:
        case LoopCorridor() as loop:
            return (Arena(loop_scene(config, visuals), loop),)
        case GeneratedTracks(difficulty=difficulty, count=count, first_seed=first):
            settings = settings_for_difficulty(difficulty)
            walls = WallParams(boxes_per_wall=WALL_BOXES)
            pool: list[Arena] = []
            for seed in range(first, first + 4 * count):
                match generate_track(seed, settings):
                    case Track() as track:
                        spec = to_spec(track, walls)
                        scene = assets.compose(spec, config.car_path, visuals=visuals)
                        pool.append(Arena(scene, track))
                    case NoValidTrack():
                        pass
                    case other:
                        assert_never(other)
                if len(pool) == count:
                    return tuple(pool)
            raise ValueError(
                f"seeds {first} to {first + 4 * count - 1} gave only {len(pool)} "
                f"tracks of difficulty {difficulty}, not {count}"
            )
        case other:
            assert_never(other)


def lap_angle(arena: Arena, sensordata: SensorData) -> npt.NDArray[np.float64]:
    """How far around the track the car in sensordata (..., nsensordata) is,
    as an angle (rad), shape (...,). One lap is 2 pi."""
    model, car = arena.scene.model, arena.scene.car
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


def stuck_window_len(config: NeoRacerEnvConfig) -> int:
    """The number of totals in the stuck window: one more than the steps in
    stuck_window_s."""
    return max(1, round(config.termination.stuck_window_s * config.control_hz)) + 1


def _observation_space(b: ObservationBounds) -> spaces.Box:
    low = np.concatenate(
        [
            np.full(8, -1.0),  # lidar: -1 is the sensor's own "no hit" value
            np.full(3, -b.accel_m_s2),  # imu_accel
            np.full(3, -b.gyro_rad_s),  # imu_gyro
            np.full(4, -1.0),  # imu_quat (unit quaternion component range)
            np.array([-b.steer_rad]),  # steer_cmd_pos
            np.array([-b.speed_m_s]),  # estimated_speed
        ]
    ).astype(np.float32)
    high = np.concatenate(
        [
            np.full(8, b.lidar_m),
            np.full(3, b.accel_m_s2),
            np.full(3, b.gyro_rad_s),
            np.full(4, 1.0),
            np.array([b.steer_rad]),
            np.array([b.speed_m_s]),
        ]
    ).astype(np.float32)
    return spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)


def observation(
    model: mujoco.MjModel, car: Car, sensordata: SensorData
) -> npt.NDArray[np.float32]:
    """The observation (..., OBS_DIM) in sensordata (..., nsensordata), from
    the car's sensors only. See the module docstring for the layout."""
    s = car.sensors

    def read(*sensors: mujoco.MjsSensor) -> SensorData:
        return np.concatenate([batch.read(model, sensordata, x) for x in sensors], -1)

    beams = (
        s.lidar_000,
        s.lidar_045,
        s.lidar_090,
        s.lidar_135,
        s.lidar_180,
        s.lidar_225,
        s.lidar_270,
        s.lidar_315,
    )
    # A beam that the car blocks reads no return, as sensors.read gives it.
    blocked = np.array([beam.name in s.blocked for beam in beams])
    lidar = np.where(blocked, sensors.NO_RETURN, read(*beams))
    wheels = read(s.fl_wheel_vel, s.fr_wheel_vel, s.rl_wheel_vel, s.rr_wheel_vel)
    # The mean wheel speed (rad/s) is sensors.motor_encoder_rad_s, the reading
    # of the car's one motor encoder, and the speed (m/s) is that times the
    # car's wheel radius.
    motor = (wheels[..., 0] + wheels[..., 1] + wheels[..., 2] + wheels[..., 3]) / 4
    speed = (motor * contract.wheel_radius(model, car))[..., None]
    parts = (lidar, read(s.imu_accel, s.imu_gyro, s.imu_quat, s.steer_cmd_pos), speed)
    return np.concatenate([p.astype(np.float32) for p in parts], -1)


def angle_around_center(
    model: mujoco.MjModel, car: Car, sensordata: SensorData
) -> npt.NDArray[np.float64]:
    """The angle (rad) of the car's position around the world origin, the
    center of the loop, shape (...,)."""
    pos = batch.read(model, sensordata, car.truth.pos)
    return np.arctan2(pos[..., 1], pos[..., 0])


@dataclass(frozen=True)
class Progress:
    """Signed progress of the car around the track, as lap_angle measures it:
    2 pi is one lap. + is the forward-travel direction."""

    angle: float
    """The car's lap_angle at the last step."""
    total: float
    """The progress in this episode."""
    delta: float
    """The progress in the last step."""
    window: tuple[float, ...]
    """total at each of the last steps, oldest first."""


def start_progress(angle: float) -> Progress:
    return Progress(angle=angle, total=0.0, delta=0.0, window=(0.0,))


def advance(p: Progress, angle: float, window_len: int) -> Progress:
    """Progress after the car moved to angle (rad). Keeps the last
    window_len totals."""
    delta = float(shortest_loop_distance(np.asarray(angle - p.angle), 2.0 * np.pi))
    total = p.total + delta
    return Progress(
        angle=angle, total=total, delta=delta, window=(*p.window, total)[-window_len:]
    )


def is_stuck(p: Progress, window_len: int, min_progress: float) -> bool:
    if len(p.window) < window_len:
        return False
    return abs(p.total - p.window[0]) < min_progress


def termination(
    cfg: TerminationConfig,
    upright_cos: float,
    contact: bool,
    stuck: bool,
    total_progress: float,
) -> TerminationReason | None:
    """The reason the episode ends at this step, or None if it continues."""
    if upright_cos < cfg.upright_rollover_cos:
        return "rollover"
    if contact:
        return "crash"
    if stuck:
        return "stuck"
    if total_progress >= 2.0 * np.pi:
        return "lap_complete"
    return None


def reward(cfg: RewardConfig, delta: float, reason: TerminationReason | None) -> float:
    r = cfg.progress_weight * delta
    match reason:
        case None:
            pass
        case "rollover":
            r -= cfg.rollover_penalty
        case "crash":
            r -= cfg.crash_penalty
        case "stuck":
            r -= cfg.stuck_penalty
        case "lap_complete":
            r += cfg.lap_bonus
        case _:
            assert_never(reason)
    return float(r)


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
        """The car's speed loop (config.car.speed_loop) in the episode.
        reset() starts it over."""

        self.action_space = spaces.Box(
            low=_ACTION_LOW, high=_ACTION_HIGH, dtype=np.float32
        )
        self.observation_space = _observation_space(config.obs_bounds)

        self._window_len = stuck_window_len(config)

        self._viewer = None
        self._viewer_model = self.model

        # Episode state. gymnasium requires reset() before step(), and
        # reset() sets these values again.
        self._step_count = 0
        self._progress = start_progress(0.0)

    # --- internal helpers -----------------------------------------------

    def _lap_angle(self) -> float:
        return float(lap_angle(self._arena, self.data.sensordata))

    def _drive_angle(self) -> float:
        """The drivetrain's angle (rad) now, at the last step's end."""
        return float(batch.drive_angle(self.model, self._car, self.data.qpos))

    def _drive(self, speed: float, steer: float) -> bool:
        """Step one control period with speed (action[0]) and steer (rad)
        commanded, the car's speed loop, if it has one, holding the speed.
        Return whether a car geom touched a wall."""
        params, dt = self.config.car.speed_loop, float(self.model.opt.timestep)
        if params is None or self.loop is None:
            commanded, throttle = 0.0, speed * self.config.max_speed
        else:
            commanded = float(speed_loop.target(params, speed, self.config.max_speed))
            throttle = float(self.loop.throttle)
        contact, fresh = False, True
        for _ in range(self._substeps_per_action):
            if fresh:
                self.data.qacc_warmstart[:] = 0
                command = ControlCommand(throttle=throttle, steer=steer)
                sim.apply_command(self._car, self.data, command)
            mujoco.mj_step(self.model, self.data)
            if collision.car_touches_wall(self.data, self._car):
                contact = True
            fresh = False
            if params is not None and self.loop is not None:
                angle = self._drive_angle()
                self.loop = speed_loop.step(params, self.loop, commanded, angle, dt)
                fresh, throttle = (
                    self.loop.measure.span == 0.0,
                    float(self.loop.throttle),
                )
        return contact

    def _build_observation(self) -> tuple[Observation, float]:
        """The observation, and the estimated speed (m/s) in it."""
        obs = observation(self.model, self._car, self.data.sensordata)
        return obs, float(obs[OBS_DIM - 1])

    def _build_info(
        self,
        speed: float,
        contact: bool,
        upright_cos: float,
        reason: TerminationReason | None,
    ) -> EnvInfo:
        return {
            "progress_rad": self._progress.total,
            "lap_frac": self._progress.total / (2.0 * np.pi),
            "speed_ms": speed,
            "upright_cos": upright_cos,
            "contact": contact,
            "episode_time_s": self._step_count * self.effective_control_dt,
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
            self.loop = speed_loop.start(params, self._drive_angle())

        self._step_count = 0
        self._progress = start_progress(self._lap_angle())

        obs, speed = self._build_observation()
        upright_cos = sim.car_upright_cos(self.data, self._car)
        info = self._build_info(
            speed, contact=False, upright_cos=upright_cos, reason=None
        )
        return obs, info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), _ACTION_LOW, _ACTION_HIGH
        )
        steer = float(action[1]) * self._limits.steer_rad
        contact_this_tick = self._drive(float(action[0]), steer)

        self._step_count += 1
        window_len = self._window_len
        self._progress = advance(self._progress, self._lap_angle(), window_len)

        upright_cos = sim.car_upright_cos(self.data, self._car)
        stuck = is_stuck(
            self._progress, window_len, self.config.termination.stuck_progress_rad
        )
        reason = termination(
            self.config.termination,
            upright_cos,
            contact_this_tick,
            stuck,
            self._progress.total,
        )
        terminated = reason is not None
        truncated = (
            not terminated
        ) and self._step_count >= self.config.max_episode_steps

        step_reward = reward(self.config.reward, self._progress.delta, reason)
        obs, speed = self._build_observation()
        info = self._build_info(speed, contact_this_tick, upright_cos, reason)

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
