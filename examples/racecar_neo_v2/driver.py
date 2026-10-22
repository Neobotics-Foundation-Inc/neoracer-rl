"""
Drive the RACECAR Neo V2 with a policy trained in its MuJoCo model
(policy.py, v2_policy.npz). It needs racecar_core and numpy only.

Every policy.CONTROL_PERIOD the policy reads the range in eight directions
from the LiDAR's scan, the speed from the encoder and its own last action,
and sets the throttle and the steering. It was trained to drive
counterclockwise around a closed corridor about 1 m wide.

On the car, from this folder:
    python3 driver.py
Hold the right bumper while it drives; release it to stop. For a first run,
lower SPEED_LIMIT and keep a hand on the bumper.
"""

import csv
import os
import sys

# Next to the labs, the library is two directories up.
sys.path.insert(0, "../../library")
import policy
import racecar_core  # ty: ignore[unresolved-import]

# The share of the policy's throttle that the car gets, in (0, 1].
SPEED_LIMIT = 1.0
# "clockwise" drives the other way round: the policy sees the scan and
# steers mirrored, left for right, as the car is close to symmetric.
DIRECTION: policy.Direction = "counterclockwise"
# Drive only while the right bumper is held.
REQUIRE_DEADMAN = True
# A CSV file to record each action in, with policy.LOG_COLUMNS, or None.
LOG: str | None = None

rc = racecar_core.create_racecar()
POLICY = policy.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_policy.npz")
)
# A racecar_core without the encoder's speed gives the steady speed of the
# last throttle instead.
ENCODER = hasattr(rc.physics, "get_encoder_speed")

driver = policy.START
scan = policy.NO_SCAN
clock = 0.0  # s since start


def speed() -> float:
    """The car's speed (m/s)."""
    if ENCODER:
        return float(rc.physics.get_encoder_speed())
    throttle, _ = policy.command(POLICY, driver.action, SPEED_LIMIT)
    return policy.steady_speed(throttle)


def start() -> None:
    global driver, scan, clock
    driver, scan, clock = policy.START, policy.NO_SCAN, 0.0
    if LOG is not None:
        with open(LOG, "w", newline="") as f:
            csv.writer(f).writerow(policy.LOG_COLUMNS)
    # The policy's throttle is the share of the car's full speed.
    rc.drive.set_max_speed(1.0)
    rc.drive.stop()
    print(f">> V2 policy, {DIRECTION}, speed limit {SPEED_LIMIT}.")
    print(">> Hold the RIGHT BUMPER to drive.")


def update() -> None:
    global driver, scan, clock
    dt = rc.get_delta_time()
    clock += dt
    scan = policy.watch(scan, rc.lidar.get_samples(), dt)
    if REQUIRE_DEADMAN and not rc.controller.is_down(rc.controller.Button.RB):
        driver = policy.START
        rc.drive.stop()
        return
    if len(scan.samples_cm) == 0 or policy.stale(scan):
        # No scan yet, or the LiDAR has stopped: the next new scan gets a
        # new action.
        driver = policy.START
        rc.drive.stop()
        return
    car_speed, last, acts = speed(), driver.action, policy.due(driver, dt)
    driver = policy.control(POLICY, driver, dt, scan.samples_cm, car_speed, DIRECTION)
    command = policy.command(POLICY, driver.action, SPEED_LIMIT)
    rc.drive.set_speed_angle(*command)
    if acts and LOG is not None:
        ranges = policy.ranges_from_scan(scan.samples_cm)
        with open(LOG, "a", newline="") as f:
            row = (clock, *ranges, car_speed, *last, *driver.action, *command)
            csv.writer(f).writerow(row)


def update_slow() -> None:
    throttle, steering = driver.action
    print(
        f"[v2] throttle {throttle:+.2f} steering {steering:+.2f} speed {speed():.2f} m/s"
    )


if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
