"""
Gymnasium environment for the NeoRacer MuJoCo car.

This module gives the NeoRacer simulator the Gymnasium Env API. It uses the
neoracer_mujoco modules (assets, sim, sensors, collision, contract) and adds
no physics, sensors or actuator mappings. The classical controllers in
neoracer_mujoco.control drive the same simulator.

Frame: +X forward, +Y left, +Z up (see neoracer-mujoco's assets/neoracer.xml).

Track: assets/tracks/loop_corridor.xml only.

Action space -- Box([-1,-1],[1,1]), float32:
    action[0] = normalized throttle, + = forward
    action[1] = normalized steer,    + = LEFT

    The signs are the ones in contract.py: + torque = forward, + steer = left.
    racecar_core uses + angle = right, so this env does not use
    racecar_core_adapter.MujocoDrive, which negates the steer. step() clamps
    the action and scales it by the actuator limits of the compiled model
    (contract.actuator_limits), and the throttle also by throttle_scale.

Observation space -- Box(20,), float32, built only from sensors.SensorReadings
(no raw qpos/qvel):
    [0:8]   lidar_ranges   m, -1 = no hit (sensors.lidar_ranges)
    [8:11]  imu_accel      m/s^2, body frame, gravity included
    [11:14] imu_gyro       rad/s, body frame
    [14:18] imu_quat       orientation quaternion (w,x,y,z), world frame
    [18]    steer_cmd_pos  rad, virtual steering input
    [19]    estimated_speed  m/s, from motor_encoder_rad_s: the mean of the
            4 wheel speeds, like the one drive motor of the real car. It is
            not the true speed imu_linvel.

Control rate: 30 Hz, the target LiDAR scan rate of the real hardware.
__init__ computes the physics steps per action from model.opt.timestep.
Each step starts MuJoCo's constraint solver cold (qacc_warmstart = 0), as
mujoco.rollout does, so a row of NeoRacerBatch on the CPU backend steps the
same as this env, bit for bit.

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
from neoracer_mujoco import assets, batch, collision, contract, physics, sensors, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.physics import PhysicsRanges
from neoracer_mujoco.scene import Car, Scene

# The spawn offset, the progress around the world origin, the lap and the
# reward are specific to the geometry of this one track.
_TRACK = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")

# Spawn offset (m) on loop_corridor.xml: its outer wall is at y=-3 and its
# inner wall at y=-2, so the 1.0 m wide bottom straight is centered at
# y=-2.5. At the origin, the car would spawn inside the inner wall.
_LOOP_SPAWN_OFFSET = (0.0, -2.5, 0.0)

OBS_DIM = 20
type Observation = npt.NDArray[np.float32]  # shape (OBS_DIM,), see module docstring
type SensorData = npt.NDArray[np.float64]  # shape (..., nsensordata)

_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)

type TerminationReason = Literal["rollover", "crash", "stuck", "lap_complete"]


class EnvInfo(TypedDict, closed=True):
    """The info dict of reset() and step(). gymnasium requires a dict."""

    progress_rad: float  # rad; cumulative signed progress in this episode
    lap_frac: float  # laps; progress_rad / 2 pi, negative if the car drove backward
    speed_ms: float  # m/s; estimated_speed, observation [19]
    upright_cos: float  # cos(tilt) of the car body
    contact: bool  # a car geom touched a wall in this step
    episode_time_s: float  # s
    termination_reason: TerminationReason | None


def _compose_yaw_perturbation(q: np.ndarray, yaw: float) -> np.ndarray:
    """q * dq, where dq is a rotation of yaw (rad) about +Z. q and the
    result are quaternions (w, x, y, z)."""
    half = yaw / 2.0
    dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    out = np.empty(4)
    mujoco.mju_mulQuat(out, q, dq)
    return out


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights. Nobody has tuned them yet. See the module docstring for
    the formula."""

    progress_weight: float = 1.0  # reward per radian of signed progress around the loop
    rollover_penalty: float = 10.0  # one-time penalty on rollover termination
    # One-time penalties on crash and stuck termination. Equal, so a car that
    # tries the first corner and hits the wall returns more than a car that
    # does not move.
    crash_penalty: float = 5.0
    stuck_penalty: float = 5.0
    lap_bonus: float = 10.0  # one-time bonus on successful lap completion


@dataclass(frozen=True)
class SpawnRange:
    """Bounds of the random spawn perturbation at reset(). Each value is
    sampled uniformly in [-bound, +bound]."""

    lateral_m: float = 0.15  # m, along world Y
    yaw_rad: float = 0.15  # rad, about +Z
    resettle_steps: int = (
        50  # physics steps after the perturbation, to settle the suspension
    )


@dataclass(frozen=True)
class TerminationConfig:
    """Thresholds that end an episode. See termination."""

    upright_rollover_cos: float = 0.5  # cos(tilt) below this = rolled over
    stuck_window_s: float = 4.0  # s
    stuck_progress_rad: float = 0.05  # rad, min net progress over the window


@dataclass(frozen=True)
class ObservationBounds:
    """Bounds of the observation space, with a margin over the peaks in 160
    episodes on loop_corridor.xml at full throttle at the full motor limit
    (throttle_scale=1) with random steering, each run to its end (most end in
    a wall crash), with the elliptic friction cone and impratio=10.
    If the track, the spawn range or the car changes, measure the peaks again."""

    lidar_m: float = 50.0  # m
    accel_m_s2: float = 1000.0  # m/s^2; the peak, in a crash, is ~590 m/s^2
    gyro_rad_s: float = 100.0  # rad/s; the peak is ~14 rad/s
    # rad; the peak is ~0.83 rad. MuJoCo's 0.45 rad joint limit is soft, so a
    # crash can push past it.
    steer_rad: float = 2.0
    # m/s; the same limit as test_top_speed_bounded in neoracer-mujoco's
    # validation/test_conformance.py. The car passes 27 m/s on a long straight.
    speed_m_s: float = 40.0


@dataclass(frozen=True)
class NeoRacerEnvConfig:
    """Environment configuration. The track is always loop_corridor.xml."""

    car_path: str = assets.DEFAULT_CAR
    control_hz: float = 30.0  # Hz
    # The fraction of the motor torque limit that action[0] = 1 gives, in
    # (0, 1]. At the full limit the car passes 27 m/s, and loop_corridor needs
    # a few m/s: a LiDAR centering controller laps at 0.1 of the limit
    # (about 4 m/s) and hits the wall at 0.2 (about 5.7 m/s).
    throttle_scale: float = 0.1
    max_episode_steps: int = 1800  # 60 s at 30 Hz
    obs_bounds: ObservationBounds = ObservationBounds()
    spawn: SpawnRange = SpawnRange()
    termination: TerminationConfig = TerminationConfig()
    reward: RewardConfig = RewardConfig()
    # The range of each factor of the car's physics, drawn at each reset.
    physics: PhysicsRanges = PhysicsRanges()

    def __post_init__(self) -> None:
        if not 0.0 < self.throttle_scale <= 1.0:
            raise ValueError(
                f"throttle_scale must be in (0, 1], not {self.throttle_scale}"
            )


def loop_scene(config: NeoRacerEnvConfig, visuals: bool) -> Scene:
    """The car of config on loop_corridor.xml, at the spawn position. Without
    visuals, the car's meshes are left out: the physics and the sensors are
    the same, and the model is smaller and faster (assets.compose)."""
    # compose raises ValueError if the car has no free joint or misses an
    # actuator or sensor.
    return assets.compose(
        mujoco.MjSpec.from_file(_TRACK),
        config.car_path,
        _LOOP_SPAWN_OFFSET,
        visuals=visuals,
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

    lidar = read(
        s.lidar_000,
        s.lidar_045,
        s.lidar_090,
        s.lidar_135,
        s.lidar_180,
        s.lidar_225,
        s.lidar_270,
        s.lidar_315,
    )
    wheels = read(s.fl_wheel_vel, s.fr_wheel_vel, s.rl_wheel_vel, s.rr_wheel_vel)
    # The mean wheel speed (rad/s) is sensors.motor_encoder_rad_s, the reading
    # of the car's one motor encoder, and the speed (m/s) is
    # sensors.estimated_linear_speed_ms of it.
    motor = (wheels[..., 0] + wheels[..., 1] + wheels[..., 2] + wheels[..., 3]) / 4
    speed = (motor * sensors.WHEEL_RADIUS)[..., None]
    parts = (lidar, read(s.imu_accel, s.imu_gyro, s.imu_quat, s.steer_cmd_pos), speed)
    return np.concatenate([p.astype(np.float32) for p in parts], -1)


def angle_around_center(
    model: mujoco.MjModel, car: Car, sensordata: SensorData
) -> npt.NDArray[np.float64]:
    """The angle (rad) of the car's position around the world origin, the
    center of the loop, shape (...,)."""
    pos = batch.read(model, sensordata, car.truth.pos)
    return np.arctan2(pos[..., 1], pos[..., 0])


def apply_spawn_offset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    car: Car,
    spawn: SpawnRange,
    rng: np.random.Generator,
) -> None:
    """Move the car in data sideways and turn it, by random amounts within
    spawn, then step spawn.resettle_steps physics steps to settle the
    suspension."""
    lateral = float(rng.uniform(-spawn.lateral_m, spawn.lateral_m))
    yaw = float(rng.uniform(-spawn.yaw_rad, spawn.yaw_rad))
    # The car's free joint in qpos, a view: (x, y, z, qw, qx, qy, qz).
    qpos = data.bind_scalar(car.joint).qpos
    qpos[1] += lateral
    qpos[3:7] = _compose_yaw_perturbation(qpos[3:7], yaw)
    mujoco.mj_forward(model, data)
    for _ in range(spawn.resettle_steps):
        mujoco.mj_step(model, data)


@dataclass(frozen=True)
class Progress:
    """Signed progress of the car around the loop center (the world origin;
    the track is symmetric about it). + is the forward-travel direction."""

    angle: float  # rad; angle of the car around the center, at the last step
    total: float  # rad; cumulative progress in this episode
    delta: float  # rad; progress in the last step
    window: tuple[float, ...]  # total at each of the last steps, oldest first


def start_progress(angle: float) -> Progress:
    return Progress(angle=angle, total=0.0, delta=0.0, window=(0.0,))


def advance(p: Progress, angle: float, window_len: int) -> Progress:
    """Progress after the car moved to angle (rad). Keeps the last
    window_len totals."""
    delta = angle - p.angle
    if delta > np.pi:
        delta -= 2.0 * np.pi
    elif delta < -np.pi:
        delta += 2.0 * np.pi
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
    """Gymnasium environment driving the NeoRacer car around
    assets/tracks/loop_corridor.xml.

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

        # scene.car holds the car's parts as spec elements, which the env
        # binds to self.data where it reads or writes them. scene.model keeps
        # the physics of the XML; self.model has the physics of the episode.
        self.scene = loop_scene(config, visuals=render_mode is not None)
        self.model = copy.deepcopy(self.scene.model)
        self.data = mujoco.MjData(self.model)
        self.physics = physics.Physics()
        self._car = self.scene.car

        self._substeps_per_action = substeps_per_action(config, self.model)
        # The true control period, after rounding to a whole number of
        # physics substeps. It can differ slightly from 1/control_hz.
        self.effective_control_dt = self._substeps_per_action * float(
            self.model.opt.timestep
        )

        self._limits = contract.actuator_limits(self.model, self._car)
        self._max_torque = config.throttle_scale * self._limits.torque_nm  # N·m

        self.action_space = spaces.Box(
            low=_ACTION_LOW, high=_ACTION_HIGH, dtype=np.float32
        )
        self.observation_space = _observation_space(config.obs_bounds)

        self._window_len = stuck_window_len(config)

        self._viewer = None

        # Episode state. gymnasium requires reset() before step(), and
        # reset() sets these values again.
        self._step_count = 0
        self._progress = start_progress(0.0)

    # --- internal helpers -----------------------------------------------

    def _angle_around_center(self) -> float:
        return float(angle_around_center(self.model, self._car, self.data.sensordata))

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

        nominal = self.scene.model
        mujoco.mj_resetData(nominal, self.data)
        sim.apply_ctrl_and_step(nominal, self.data, 0.0, sim.SETTLE_STEPS)
        apply_spawn_offset(
            nominal, self.data, self._car, self.config.spawn, self.np_random
        )
        self.physics = physics.sample(self.config.physics, self.np_random)
        physics.apply_physics(self.model, self.scene, self.physics)

        self._step_count = 0
        self._progress = start_progress(self._angle_around_center())

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
        sim.apply_command(
            self._car,
            self.data,
            ControlCommand(
                throttle=float(action[0]) * self._max_torque,
                steer=float(action[1]) * self._limits.steer_rad,
            ),
        )

        self.data.qacc_warmstart[:] = 0
        contact_this_tick = False
        for _ in range(self._substeps_per_action):
            mujoco.mj_step(self.model, self.data)
            if collision.car_touches_wall(self.data, self._car):
                contact_this_tick = True

        self._step_count += 1
        window_len = self._window_len
        self._progress = advance(
            self._progress, self._angle_around_center(), window_len
        )

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
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
