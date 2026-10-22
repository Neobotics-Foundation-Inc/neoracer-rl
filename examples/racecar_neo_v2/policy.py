"""
The RACECAR Neo V2's driving policy: what it sees, how it acts, and the step
that driver.py takes on the car. It needs numpy only, so it runs on the car.

Frame: the car's, +X forward, +Y left. An angle is counterclockwise from the
front, unless its name says otherwise; a racecar_core scan runs clockwise.

The policy sees OBS_DIM numbers, all of which the car measures:
- the range (m) in each of BEAM_DEGREES, from the LiDAR's scan: the nearest
  return within WINDOW_DEGREES, or MAX_RANGE for none or farther;
- the speed (m/s) from the drive board's encoder;
- its last action.
Every CONTROL_PERIOD it acts with two numbers in [-1, 1]: the throttle, which
the car's speed controller turns into a speed, and the steering, +1 full lock
to the left. Driving clockwise, it sees and steers mirrored, left for right.
"""

from dataclasses import dataclass, replace
from typing import Literal, assert_never

import numpy as np
import numpy.typing as npt

type Ranges = npt.NDArray[np.float64]  # (..., len(BEAM_DEGREES)) m
type Observation = npt.NDArray[np.float64]  # (..., OBS_DIM)
type Action = npt.NDArray[np.float64]  # (..., 2): throttle, steering (+ = left)
type Direction = Literal["counterclockwise", "clockwise"]

BEAM_DEGREES = (0, 45, 90, 135, 180, 225, 270, 315)
MAX_RANGE = 5.0  # m
WINDOW_DEGREES = 2.0
CONTROL_PERIOD = 1 / 30  # s
OBS_DIM = len(BEAM_DEGREES) + 3
# The reference car's steady speed (m/s) at each throttle, from
# neoracer_spec.racecar_neo_v2's Drive.steady_speed.
STEADY_SPEED = ((0.0, 0.0), (0.25, 0.721), (0.5, 0.833), (1.0, 1.07))
# A scan that has not changed for this long (s) is stale: the LiDAR has
# stopped, and racecar_core repeats its last scan. The head turns in 0.135 s,
# and a real scan's noise changes it every turn.
STALE_SCAN = 0.5


def ranges_from_scan(samples_cm: npt.ArrayLike) -> Ranges:
    """The ranges (m) of a racecar_core scan: samples in cm, clockwise from
    the front, 0 for no return."""
    samples = np.asarray(samples_cm, dtype=np.float64) / 100.0
    n = len(samples)
    half = max(1, round(WINDOW_DEGREES / 360.0 * n))
    ranges = np.full(len(BEAM_DEGREES), MAX_RANGE)
    for k, degrees in enumerate(BEAM_DEGREES):
        center = round((360 - degrees) % 360 / 360.0 * n)
        window = samples[(center + np.arange(-half, half + 1)) % n]
        returns = window[window > 0.0]
        if returns.size:
            ranges[k] = min(float(returns.min()), MAX_RANGE)
    return ranges


def clip_ranges(ranges_m: npt.ArrayLike) -> Ranges:
    """Ranges (m) with no return, -1 as a MuJoCo rangefinder reads it, or
    farther than MAX_RANGE read as MAX_RANGE."""
    r = np.asarray(ranges_m, dtype=np.float64)
    return np.where(r < 0.0, MAX_RANGE, np.minimum(r, MAX_RANGE))


def mirror_ranges(ranges: Ranges) -> Ranges:
    """The ranges of the scan mirrored left for right: the beam at a degrees
    reads the range at -a."""
    return ranges[..., [-k % len(BEAM_DEGREES) for k in range(len(BEAM_DEGREES))]]


def observation(
    ranges: Ranges, speed: npt.ArrayLike, last_action: Action
) -> Observation:
    """The policy's input: ranges (..., 8) m, speed (...) m/s and the last
    action (..., 2)."""
    speed = np.broadcast_to(np.asarray(speed, dtype=np.float64), ranges.shape[:-1])
    return np.concatenate([ranges, speed[..., None], last_action], axis=-1)


def steady_speed(throttle: float) -> float:
    """The speed (m/s) that the reference car held at a throttle in [-1, 1]."""
    x, y = zip(*STEADY_SPEED, strict=True)
    return float(np.sign(throttle) * np.interp(abs(throttle), x, y))


@dataclass(frozen=True)
class Layer:
    weight: npt.NDArray[np.float64]  # (inputs, outputs)
    bias: npt.NDArray[np.float64]  # (outputs,)


@dataclass(frozen=True)
class Policy:
    """A trained policy: an MLP with tanh hidden layers over the normalized
    observation. throttle_scale is the throttle that action 1 commanded in
    training (NeoRacerEnvConfig.throttle_scale)."""

    layers: tuple[Layer, ...]
    obs_mean: npt.NDArray[np.float64]  # (OBS_DIM,)
    obs_var: npt.NDArray[np.float64]  # (OBS_DIM,)
    obs_clip: float  # standard deviations
    throttle_scale: float


def load(path: str) -> Policy:
    """The policy in an .npz that train_v2.py wrote. Raises KeyError if an
    array is missing."""
    with np.load(path) as f:
        count = sum(
            1 for k in f.files if k.startswith("actor.") and k.endswith(".weight")
        )
        if count == 0:
            raise KeyError(f"{path} has no actor layers")
        return Policy(
            layers=tuple(
                Layer(f[f"actor.{i}.weight"], f[f"actor.{i}.bias"])
                for i in range(count)
            ),
            obs_mean=f["obs_mean"],
            obs_var=f["obs_var"],
            obs_clip=float(f["obs_clip"]),
            throttle_scale=float(f["throttle_scale"]),
        )


def act(policy: Policy, obs: Observation) -> Action:
    """The action in [-1, 1] for each observation: the policy's mean."""
    x = (obs - policy.obs_mean) / np.sqrt(policy.obs_var + 1e-8)
    x = np.clip(x, -policy.obs_clip, policy.obs_clip)
    for layer in policy.layers[:-1]:
        x = np.tanh(x @ layer.weight + layer.bias)
    out = policy.layers[-1]
    return np.clip(x @ out.weight + out.bias, -1.0, 1.0)


@dataclass(frozen=True)
class Driver:
    """The car program between two updates."""

    action: Action  # (2,) the policy's last action
    wait: float  # s until the next one; at most 0 means now


START = Driver(action=np.zeros(2), wait=0.0)


@dataclass(frozen=True)
class Scan:
    """The LiDAR's last scan and how long it has stayed the same."""

    samples_cm: npt.NDArray[np.float64]
    age: float  # s since it changed


NO_SCAN = Scan(samples_cm=np.zeros(0), age=0.0)


def watch(scan: Scan, samples_cm: npt.ArrayLike, dt: float) -> Scan:
    """The scan after an update dt (s) after the last, when racecar_core
    gives samples_cm."""
    samples = np.array(samples_cm, dtype=np.float64)
    if np.array_equal(samples, scan.samples_cm):
        return replace(scan, age=scan.age + dt)
    return Scan(samples, 0.0)


def stale(scan: Scan) -> bool:
    """Whether the LiDAR has stopped sending scans."""
    return scan.age > STALE_SCAN


def control(
    policy: Policy,
    driver: Driver,
    dt: float,
    scan_cm: npt.ArrayLike,
    speed: float,
    direction: Direction,
) -> Driver:
    """The driver after an update dt (s) after the last, with the scan and
    the speed (m/s) the car gives then. It acts at the first update after
    each deadline, and the deadlines are CONTROL_PERIOD apart, whatever the
    rate of the updates. After a stall, an update more than half a period
    after a deadline, it acts at once and sets the next deadline a full
    period later. So two actions are at least half a period apart."""
    wait = driver.wait - dt
    if wait > 0.0:
        return replace(driver, wait=wait)
    ranges = ranges_from_scan(scan_cm)
    match direction:
        case "counterclockwise":
            action = act(policy, observation(ranges, speed, driver.action))
        case "clockwise":
            flip = np.array([1.0, -1.0])
            seen = observation(mirror_ranges(ranges), speed, driver.action * flip)
            action = act(policy, seen) * flip
        case _:
            assert_never(direction)
    stalled = wait < -CONTROL_PERIOD / 2
    return Driver(action, CONTROL_PERIOD if stalled else wait + CONTROL_PERIOD)


def command(policy: Policy, action: Action, speed_limit: float) -> tuple[float, float]:
    """racecar_core's (speed, angle) for an action, with the car's max speed
    1: the throttle times policy.throttle_scale and speed_limit, and the
    steering turned to racecar_core's sign, + to the right."""
    return float(action[0]) * policy.throttle_scale * speed_limit, -float(action[1])
