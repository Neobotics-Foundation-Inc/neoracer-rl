"""
A golden run of the shipped policy: examples/drive_v2.py runs the car's
program, examples/racecar_neo_v2/driver.py, unchanged on the RACECAR Neo
V2's model on loop_corridor, through the racecar_core adapter.
"""

import csv
from typing import get_args

import numpy as np

from examples import drive_v2
from examples.racecar_neo_v2 import policy


def test_the_shipped_policy_laps_both_ways_without_touching_a_wall():
    ways: tuple[tuple[policy.Direction, float], ...] = (
        ("counterclockwise", 1.0),
        ("clockwise", -1.0),
    )
    for direction, sign in ways:
        result = drive_v2.drive(60.0, direction, speed_limit=1.0)
        assert result.wall_contacts == 0, (direction, result)
        assert sign * result.laps > 1.5, (direction, result)


def test_the_log_has_a_row_for_each_action_that_replays_it(tmp_path):
    car = policy.load(str(drive_v2.PROGRAM.parent / "v2_policy.npz"))
    seconds, speed_limit = 5.0, 0.5
    for direction in get_args(policy.Direction.__value__):
        path = tmp_path / f"{direction}.csv"
        drive_v2.drive(seconds, direction, speed_limit, log=str(path))
        with path.open(newline="") as f:
            reader = csv.reader(f)
            assert tuple(next(reader)) == policy.LOG_COLUMNS
            log = np.array([[float(x) for x in row] for row in reader])
        assert abs(len(log) - seconds / policy.CONTROL_PERIOD) <= 2, direction
        beams = len(policy.BEAM_DEGREES)
        for row in log:
            ranges, speed = row[1 : 1 + beams], row[1 + beams]
            last, action = row[2 + beams : 4 + beams], row[4 + beams : 6 + beams]
            np.testing.assert_array_equal(
                policy.decide(car, ranges, speed, last, direction), action
            )
            command = policy.command(car, action, speed_limit)
            np.testing.assert_array_equal(row[6 + beams :], command)
        # Each action is the next one's last action.
        np.testing.assert_array_equal(
            log[1:, 2 + beams : 4 + beams], log[:-1, 4 + beams : 6 + beams]
        )
