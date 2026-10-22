"""
A golden run of the shipped policy: examples/drive_v2.py runs the car's
program, examples/racecar_neo_v2/driver.py, unchanged on the RACECAR Neo
V2's model on loop_corridor, through the racecar_core adapter.
"""

from examples import drive_v2
from examples.racecar_neo_v2 import policy


def test_the_shipped_policy_laps_both_ways_without_touching_a_wall():
    ways: tuple[tuple[policy.Direction, float], ...] = (
        ("counterclockwise", 1.0),
        ("clockwise", -1.0),
    )
    for direction, sign in ways:
        result = drive_v2.drive(40.0, direction, speed_limit=1.0)
        assert result.wall_contacts == 0, (direction, result)
        assert sign * result.laps > 1.2, (direction, result)
