"""
Laws for the RACECAR Neo V2's car program, examples/racecar_neo_v2: what
the policy reads in a racecar_core scan, the mirrored drive, the timing of
its actions and its command, without the simulator.
"""

import ast
import importlib.resources
import itertools
import runpy
import types
from collections.abc import Iterable

import numpy as np
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2

from examples import drive_v2
from examples.racecar_neo_v2 import policy

CAR_PYTHON = (3, 12)  # the RACECAR Neo V2's Ubuntu 24.04


def _random_policy(seed: int) -> policy.Policy:
    """A policy with random weights, whose action depends on every input."""
    rng = np.random.default_rng(seed)
    sizes = (policy.OBS_DIM, 16, 2)
    return policy.Policy(
        layers=tuple(
            policy.Layer(rng.normal(size=(n_in, n_out)), rng.normal(size=n_out))
            for n_in, n_out in itertools.pairwise(sizes)
        ),
        obs_mean=rng.normal(size=policy.OBS_DIM),
        obs_var=rng.uniform(0.5, 2.0, policy.OBS_DIM),
        obs_clip=10.0,
        throttle_scale=0.8,
    )


def _mirror_scan(scan: np.ndarray) -> np.ndarray:
    """A racecar_core scan mirrored left for right: slot i reads slot -i."""
    return np.roll(scan[::-1], 1)


def test_a_return_shows_in_the_beam_that_points_at_it_and_in_no_other():
    for slots in (360, 720, 1080):
        for k, degrees in enumerate(policy.BEAM_DEGREES):
            # Slot i of a scan points i * 360 / slots degrees clockwise.
            scan = np.zeros(slots)
            scan[round((-degrees) % 360 * slots / 360)] = 120.0  # cm
            expected = np.full(len(policy.BEAM_DEGREES), policy.MAX_RANGE)
            expected[k] = 1.2
            np.testing.assert_array_equal(
                policy.ranges_from_scan(scan), expected, err_msg=f"{slots} {degrees}"
            )


def test_a_beam_reads_the_nearest_return_in_its_window_up_to_max_range():
    scan = np.full(1080, 300.0)  # cm
    scan[3] = 150.0  # 1 degree clockwise of the front
    ranges = policy.ranges_from_scan(scan)
    assert ranges[0] == 1.5
    np.testing.assert_array_equal(ranges[1:], 3.0)
    far = policy.ranges_from_scan(np.full(1080, 100.0 * policy.MAX_RANGE + 50.0))
    np.testing.assert_array_equal(far, policy.MAX_RANGE)


def test_mirroring_the_scan_mirrors_the_ranges():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scan = rng.uniform(10.0, 700.0, 1080) * (rng.random(1080) > 0.1)
        ranges = policy.ranges_from_scan(scan)
        np.testing.assert_array_equal(
            policy.ranges_from_scan(_mirror_scan(scan)), policy.mirror_ranges(ranges)
        )
        np.testing.assert_array_equal(
            policy.mirror_ranges(policy.mirror_ranges(ranges)), ranges
        )


def test_driving_clockwise_is_the_mirror_of_driving_counterclockwise():
    rng = np.random.default_rng(1)
    flip = np.array([1.0, -1.0])
    for seed in range(20):
        car = _random_policy(seed)
        scan = rng.uniform(10.0, 700.0, 1080)
        speed, last = rng.uniform(-0.5, 1.2), rng.uniform(-1.0, 1.0, 2)
        ccw = policy.control(
            car, policy.Driver(last, 0.0), 0.01, scan, speed, "counterclockwise"
        )
        cw = policy.control(
            car,
            policy.Driver(last * flip, 0.0),
            0.01,
            _mirror_scan(scan),
            speed,
            "clockwise",
        )
        np.testing.assert_allclose(cw.action, ccw.action * flip, atol=1e-12)


COUNT = 1e-3  # what _counting_policy adds to its action each time it acts


def _counting_policy() -> policy.Policy:
    """A policy whose action is its last action plus COUNT, so that its
    action counts the times it acted."""
    weight = np.zeros((policy.OBS_DIM, 2))
    weight[-2:] = np.eye(2)  # the last action is the last two inputs
    zeros, ones = np.zeros(policy.OBS_DIM), np.ones(policy.OBS_DIM)
    return policy.Policy(
        (policy.Layer(weight, np.full(2, COUNT)),), zeros, ones, 1e9, 1.0
    )


def _acts(driver: policy.Driver) -> int:
    return round(driver.action[0] / COUNT)


def _action_times(car: policy.Policy, dts: Iterable[float]) -> list[float]:
    """The times (s) at which the program acts, with updates dts (s) apart."""
    scan = np.full(1080, 100.0)
    driver, now, times = policy.START, 0.0, []
    for dt in dts:
        now += dt
        acts = _acts(driver)
        driver = policy.control(car, driver, dt, scan, 0.5, "counterclockwise")
        if _acts(driver) > acts:
            times.append(now)
    return times


def test_the_program_acts_once_a_control_period_whatever_its_update_rate():
    car = _counting_policy()
    period, seconds = policy.CONTROL_PERIOD, 3.0
    # Each case repeats its updates; the last two jitter around the period.
    for frames in (
        (1 / 360,),
        (1 / 100,),
        (1 / 60,),
        (period,),
        (0.1,),
        (period - 0.001, period + 0.001),
        (period - 0.015, period + 0.015),
    ):
        dts = itertools.islice(
            itertools.cycle(frames), round(seconds / np.mean(frames))
        )
        acts = len(_action_times(car, dts))
        # Updates slower than a control period each act.
        expected = seconds / max(np.mean(frames), period)
        assert abs(acts - expected) <= 1, (frames, acts, expected)


def test_two_actions_are_at_least_half_a_period_apart():
    rng = np.random.default_rng(2)
    car = _counting_policy()
    for high in (0.02, 0.04, 0.08):
        times = _action_times(car, rng.uniform(1e-4, high, 2000))
        assert np.min(np.diff(times)) >= policy.CONTROL_PERIOD / 2 - 1e-12, high


def test_after_a_stall_the_program_acts_at_once_and_next_a_period_later():
    car = _counting_policy()
    scan = np.full(1080, 100.0)
    period = policy.CONTROL_PERIOD
    # Each stall comes more than half a period after the next deadline.
    for stall in (1.5 * period, 2 * period, 0.1, 1.0):
        for frame in (1e-3, 1 / 360, 1 / 100):
            driver = policy.START
            for dt in (frame, stall):
                driver = policy.control(car, driver, dt, scan, 0.5, "counterclockwise")
            assert _acts(driver) == 2, (stall, frame)
            waited = 0.0
            while _acts(driver) == 2:
                driver = policy.control(
                    car, driver, frame, scan, 0.5, "counterclockwise"
                )
                waited += frame
            assert period <= waited < period + frame, (stall, frame, waited)


class _Racecar:
    """racecar_core as driver.py uses it: the LiDAR gives samples, and the
    car records each command, None for a stop."""

    DT = 1 / 60  # s, the time between two updates

    def __init__(self) -> None:
        self.samples = np.zeros(0, dtype=np.float32)
        self.commands: list[tuple[float, float] | None] = []
        self.drive = types.SimpleNamespace(
            set_max_speed=lambda speed: None,
            stop=lambda: self.commands.append(None),
            set_speed_angle=lambda speed, angle: self.commands.append((speed, angle)),
        )
        self.lidar = types.SimpleNamespace(get_samples=lambda: self.samples)
        self.physics = types.SimpleNamespace(get_encoder_speed=lambda: 0.5)
        self.controller = types.SimpleNamespace(
            Button=types.SimpleNamespace(RB=5), is_down=lambda button: True
        )

    def get_delta_time(self) -> float:
        return self.DT


def test_the_car_stops_when_the_lidar_stops_and_drives_on_its_next_scan():
    rng = np.random.default_rng(5)
    rc = _Racecar()
    with drive_v2.program_imports(rc):
        program = runpy.run_path(str(drive_v2.PROGRAM), run_name="driver")
    program["start"]()

    def drives(new_scan: bool) -> bool:
        """Whether the car drives after an update."""
        if new_scan:
            rc.samples = rng.uniform(50.0, 500.0, 1080).astype(np.float32)
        program["update"]()
        return rc.commands[-1] is not None

    # A new scan every 8 updates, a turn of the head.
    assert all(drives(k % 8 == 0) for k in range(60))
    assert drives(new_scan=True)
    # The LiDAR stops, and racecar_core repeats its last scan.
    for k in range(1, 61):
        age = k * rc.DT
        if age < policy.STALE_SCAN - rc.DT / 2:
            assert drives(new_scan=False), age
        elif age > policy.STALE_SCAN + rc.DT / 2:
            assert not drives(new_scan=False), age
        else:
            drives(new_scan=False)
    assert drives(new_scan=True)


def test_an_action_is_within_its_limits_for_any_observation():
    car = _random_policy(3)
    obs = np.random.default_rng(4).normal(0.0, 100.0, (1000, policy.OBS_DIM))
    action = policy.act(car, obs)
    assert action.shape == (1000, 2)
    assert np.all(np.abs(action) <= 1.0)


def test_the_command_scales_the_throttle_and_steers_to_racecar_cores_sign():
    car = _random_policy(5)
    for throttle, steering, limit in ((1.0, 0.5, 1.0), (-0.5, -1.0, 0.25)):
        speed, angle = policy.command(car, np.array([throttle, steering]), limit)
        assert speed == throttle * car.throttle_scale * limit
        # racecar_core's angle is + to the right; the policy's steering is + left.
        assert angle == -steering


def test_the_steady_speeds_are_the_reference_cars():
    measured = tuple(
        (at.command, at.value) for at in RACECAR_NEO_V2.drive.steady_speed.value
    )
    assert policy.STEADY_SPEED == ((0.0, 0.0), *measured)
    for throttle, speed in policy.STEADY_SPEED:
        assert policy.steady_speed(throttle) == speed
        assert policy.steady_speed(-throttle) == -speed


def test_the_car_program_parses_as_the_cars_python():
    folder = importlib.resources.files("examples.racecar_neo_v2")
    for name in ("policy.py", "driver.py"):
        ast.parse((folder / name).read_text(), name, feature_version=CAR_PYTHON)
