"""
The task: what a policy sees, when its episode ends, and what it earns. An
env takes three functions of a Step, which holds what the env knows after a
step:

    observe(step) -> observations, shape (rows, n)
    terminate(step) -> ends: each way an episode can end, by name, as a mask (rows,)
    reward(step, ends) -> rewards, shape (rows,)

NeoRacerEnv calls them with one row, and NeoRacerBatch with one row per
episode, so a function serves both. An episode that meets several ends in a
step has the first of them, in terminate's order, and reward sees only that
one. The defaults, CarObservation, LapEnds and ProgressReward, are the lap
task of the README. Each is a frozen dataclass with __call__, so a config
that holds them can be compared, printed and saved, as a function cannot. A
new task passes its own functions in NeoRacerEnvConfig, and a function can
build on the defaults:

    def reward(step, ends):
        return ProgressReward()(step, ends) - 0.01 * np.abs(step.action[:, 1])

Quantities are in SI units unless a name says otherwise.
"""

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from neoracer_mujoco import batch, contract, sensors
from neoracer_mujoco.scene import Car
from neoracer_mujoco.track_generation import shortest_loop_distance

type Rows = npt.NDArray[np.float64]
"""Shape (rows,)."""
type Mask = npt.NDArray[np.bool_]
"""Shape (rows,)."""
type Ints = npt.NDArray[np.int64]
"""Shape (rows,)."""
type Observations = npt.NDArray[np.float32]
"""Shape (rows, n)."""
type SensorData = npt.NDArray[np.float64]
"""Shape (..., nsensordata)."""
type Ends = dict[str, Mask]
"""The ways an episode can end, by name, each true in the rows that end that
way, in the order that decides which end a row that meets several has."""


@dataclass(frozen=True)
class ProgressRows:
    """How far around the track each row has gone in its episode, as
    neoracer_env.lap_angle measures it: 2 pi is one lap, and + is forward."""

    angle: Rows
    """The lap angle at the last step."""
    total: Rows
    """The progress in this episode."""
    delta: Rows
    """The progress in the last step."""
    steps: Ints
    """The steps in this episode."""
    window: npt.NDArray[np.float64]
    """The totals of the last window_len steps, shape (rows, window_len), as
    a ring: column k % window_len holds the total after step k of the
    episode."""

    @property
    def recent(self) -> Rows:
        """The progress over the last window_len - 1 steps, inf in a row whose
        episode has had fewer."""
        window_len = self.window.shape[1]
        oldest = self.window[np.arange(len(self.steps)), (self.steps + 1) % window_len]
        return np.where(self.steps >= window_len - 1, self.total - oldest, np.inf)


def start_rows(angle: Rows, window_len: int) -> ProgressRows:
    """Every row at the start of an episode, at lap angle angle."""
    zeros = np.zeros_like(angle)
    return ProgressRows(
        angle=angle,
        total=zeros,
        delta=zeros,
        steps=np.zeros(angle.shape, np.int64),
        window=np.zeros((*angle.shape, window_len)),
    )


def advance_rows(p: ProgressRows, angle: Rows) -> ProgressRows:
    """p after every row's car moved to lap angle angle. A step's progress is
    in [-pi, pi): a car moves less than half a lap in a step."""
    delta = shortest_loop_distance(angle - p.angle, 2.0 * np.pi)
    total = p.total + delta
    steps = p.steps + 1
    window = p.window.copy()
    window[np.arange(len(steps)), steps % window.shape[1]] = total
    return ProgressRows(angle, total, delta, steps, window)


def restart_rows(p: ProgressRows, done: Mask, angle: Rows) -> ProgressRows:
    """p, with the rows where done back at the start, at lap angle angle."""
    return ProgressRows(
        angle=np.where(done, angle, p.angle),
        total=np.where(done, 0.0, p.total),
        delta=np.where(done, 0.0, p.delta),
        steps=np.where(done, 0, p.steps),
        window=np.where(done[:, None], 0.0, p.window),
    )


def lead(own: ProgressRows, other: ProgressRows) -> Rows:
    """How far ahead of own other is, as progress (rad): the gap between
    their starts, the short way around the track, plus other's progress less
    own's. Below 0 where own is ahead."""
    starts = (other.angle - other.total) - (own.angle - own.total)
    return shortest_loop_distance(starts, 2.0 * np.pi) + other.total - own.total


@dataclass(frozen=True)
class Rival:
    """Another car of a race (race.py), as the Step of one of its cars sees
    it."""

    car: Car
    progress: ProgressRows
    """How far around the track it has gone, as the Step's own progress."""
    ahead: Rows
    """How far ahead of the Step's car it is (lead)."""
    contact: Mask
    """It touched the Step's car during the step."""


@dataclass(frozen=True)
class Step:
    """What the env knows after a step, or at the start of an episode, for
    each of its rows."""

    model: mujoco.MjModel
    car: Car
    sensordata: SensorData
    """Every sensor at the end of the step, shape (rows, nsensordata): the
    car's own, which the real car has too, and the truth sensors that only a
    simulator has (car.truth). read gives one of them."""
    contact: Mask
    """A car geom touched a wall during the step."""
    progress: ProgressRows
    action: npt.NDArray[np.float32]
    """The action of the step, speed and steer in [-1, 1], shape (rows, 2):
    zeros at the start of an episode."""
    rivals: tuple[Rival, ...] = ()
    """The other cars of a race, in the race's order: none for a car alone."""

    def read(self, sensor: mujoco.MjsSensor) -> npt.NDArray[np.float64]:
        """The reading of sensor in every row, shape (rows, dim)."""
        return batch.read(self.model, self.sensordata, sensor)


def first_step(
    model: mujoco.MjModel, car: Car, sensordata: SensorData, progress: ProgressRows
) -> Step:
    """The Step at the start of an episode, before any contact or action."""
    rows = len(sensordata)
    return Step(
        model,
        car,
        sensordata,
        np.zeros(rows, np.bool_),
        progress,
        np.zeros((rows, 2), np.float32),
    )


type Observe = Callable[[Step], Observations]
type Terminate = Callable[[Step], Ends]
type Reward = Callable[[Step, Ends], Rows]


def first_end(ends: Ends, rows: int) -> tuple[Ints, Ends]:
    """Each row's end, as a code, 0 while its episode goes on, else k + 1 for
    the k-th of ends, the first it meets; and ends with only those."""
    codes = np.zeros(rows, np.int64)
    for k, ended in enumerate(ends.values(), 1):
        codes = np.where((codes == 0) & ended, k, codes)
    return codes, {name: codes == k for k, name in enumerate(ends, 1)}


def encoder_speed(step: Step) -> Rows:
    """The speed (m/s) that the car's motor encoder measures: the mean turn
    rate of its four wheels (sensors.motor_encoder_rad_s) at the car's wheel
    radius. It is not the true speed when a wheel slips."""
    s = step.car.sensors
    wheels = (s.fl_wheel_vel, s.fr_wheel_vel, s.rl_wheel_vel, s.rr_wheel_vel)
    w0, w1, w2, w3 = (step.read(w)[..., 0] for w in wheels)
    return (w0 + w1 + w2 + w3) / 4 * contract.wheel_radius(step.model, step.car)


OBS_DIM = 20
"""The size of CarObservation's observation."""


@dataclass(frozen=True)
class ObservationBounds:
    """Bounds of CarObservation, with a margin over the peaks in 160 episodes
    on loop_corridor.xml at full throttle at the full battery voltage with
    random steering, each run to its end (all end in a wall crash), with the
    elliptic friction cone and impratio=10.
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
class CarObservation:
    """The 20 numbers of neoracer_env's module docstring, from the car's own
    sensors only."""

    bounds: ObservationBounds = ObservationBounds()

    def __call__(self, step: Step) -> Observations:
        s = step.car.sensors

        def read(*sensors: mujoco.MjsSensor) -> SensorData:
            return np.concatenate([step.read(x) for x in sensors], -1)

        order = sensors.LIDAR_BEAM_ORDER
        # A beam that the car blocks reads no return, as sensors.read gives it.
        blocked = np.array([name in s.blocked for name in order])
        lidar = np.where(
            blocked, sensors.NO_RETURN, read(*(getattr(s, n) for n in order))
        )
        imu = read(s.imu_accel, s.imu_gyro, s.imu_quat, s.steer_cmd_pos)
        parts = (lidar, imu, encoder_speed(step)[..., None])
        return np.concatenate([p.astype(np.float32) for p in parts], -1)

    def space(self) -> spaces.Box:
        b = self.bounds
        high = np.concatenate(
            [
                np.full(8, b.lidar_m),
                np.full(3, b.accel_m_s2),
                np.full(3, b.gyro_rad_s),
                np.full(4, 1.0),
                [b.steer_rad, b.speed_m_s],
            ]
        ).astype(np.float32)
        # -1 is the LiDAR's own "no hit", and a quaternion's parts are in [-1, 1].
        low = -high
        low[:8] = sensors.NO_RETURN
        return spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)


@dataclass(frozen=True)
class LapEnds:
    """The ends of the lap task, in this order: rollover, the car tilts so far
    that the cosine of its tilt is below upright_rollover_cos; crash, a car
    geom touches a wall; stuck, less than stuck_progress_rad of progress over
    the env's progress window; lap_complete, 2 pi of progress forward."""

    upright_rollover_cos: float = 0.5
    stuck_progress_rad: float = 0.05

    def __call__(self, step: Step) -> Ends:
        return {
            "rollover": step.read(step.car.truth.zaxis)[:, 2]
            < self.upright_rollover_cos,
            "crash": step.contact,
            "stuck": np.abs(step.progress.recent) < self.stuck_progress_rad,
            "lap_complete": step.progress.total >= 2.0 * np.pi,
        }


@dataclass(frozen=True)
class ProgressReward:
    """progress_weight per radian of progress, and a one-time amount at each
    end of LapEnds. Nobody has tuned these weights yet."""

    progress_weight: float = 1.0
    rollover_penalty: float = 10.0
    crash_penalty: float = 5.0
    """It equals stuck_penalty, so a car that tries the first corner and hits
    the wall returns more than a car that does not move."""
    stuck_penalty: float = 5.0
    lap_bonus: float = 10.0

    def __call__(self, step: Step, ends: Ends) -> Rows:
        at_end = {
            "rollover": -self.rollover_penalty,
            "crash": -self.crash_penalty,
            "stuck": -self.stuck_penalty,
            "lap_complete": self.lap_bonus,
        }
        once = np.zeros(len(step.progress.delta))
        for name, amount in at_end.items():
            if name in ends:
                once = once + np.where(ends[name], amount, 0.0)
        return self.progress_weight * step.progress.delta + once
