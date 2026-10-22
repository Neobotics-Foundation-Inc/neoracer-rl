"""
Run the car's program, racecar_neo_v2/driver.py, unchanged on the RACECAR
Neo V2's MuJoCo model on loop_corridor, through neoracer-mujoco's
racecar_core adapter: the V2's spinning LiDAR, its hall encoder and the
model's physics step. Count the laps and the times the car began to touch
a wall.

The car starts on the bottom straight, facing the way it drives. Each
option sets the program's constant of the same name.

Usage:
    python3 -m examples.drive_v2 --seconds 60
    python3 -m examples.drive_v2 --seconds 60 --direction clockwise --speed-limit 0.5
"""

import argparse
import contextlib
import importlib.resources
import math
import runpy
import sys
import types
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never, get_args

import mujoco
from neoracer_mujoco import sim
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar, racecar_neo_v2_params

from examples.racecar_neo_v2 import policy
from examples.train_v2 import CAR
from neoracer_rl.envs import NeoRacerEnvConfig, neoracer_env

PROGRAM = Path(str(importlib.resources.files("examples.racecar_neo_v2") / "driver.py"))


@dataclass(frozen=True)
class Drive:
    """What the car did."""

    laps: float  # turns around the loop's center, + counterclockwise
    wall_contacts: int  # the times the car began to touch a wall


@contextlib.contextmanager
def program_imports(rc: MujocoRacecar) -> Generator[None]:
    """In the block, `import racecar_core` gives a module whose
    create_racecar() returns rc, and the program's folder is first on
    sys.path, so it imports its own policy.py. On exit, sys.path and the
    modules of those names are as they were."""
    names = ("racecar_core", "policy")
    path = list(sys.path)
    hidden = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    shim = types.ModuleType("racecar_core")
    shim.__dict__["create_racecar"] = lambda: rc
    sys.modules["racecar_core"] = shim
    sys.path.insert(0, str(PROGRAM.parent))
    try:
        yield
    finally:
        sys.path[:] = path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(hidden)


def drive(seconds: float, direction: policy.Direction, speed_limit: float) -> Drive:
    """Run driver.py for seconds of simulated time, with its DIRECTION and
    SPEED_LIMIT set to direction and speed_limit."""
    config = NeoRacerEnvConfig(car_path=CAR)
    scene = neoracer_env.loop_scene(config, visuals=False)
    model, car = scene.model, scene.car
    data = sim.settle(model)
    match direction:
        case "counterclockwise":
            pass
        case "clockwise":
            # Turn the car to face -X, then let its suspension settle again.
            data.bind_scalar(car.joint).qpos[3:7] = (0.0, 0.0, 0.0, 1.0)
            mujoco.mj_forward(model, data)
            for _ in range(config.spawn.resettle_steps):
                mujoco.mj_step(model, data)
        case _:
            assert_never(direction)
    rc = MujocoRacecar(
        scene,
        data,
        max_steps=round(seconds / model.opt.timestep),
        car=racecar_neo_v2_params(),
    )
    with program_imports(rc):
        program: dict[str, Any] = runpy.run_path(str(PROGRAM), run_name="driver")
    settings = program["update"].__globals__
    settings["DIRECTION"], settings["SPEED_LIMIT"] = direction, speed_limit

    angle = float(neoracer_env.angle_around_center(model, car, data.sensordata))
    turned, contacts, touching = 0.0, 0, False

    def update() -> None:
        nonlocal angle, turned, contacts, touching
        program["update"]()
        now = float(neoracer_env.angle_around_center(model, car, data.sensordata))
        turned += math.remainder(now - angle, 2.0 * math.pi)
        angle = now
        touches = bool(data.bind_scalar(car.truth.wall_contact).data[0] > 0.0)
        if touches and not touching:
            contacts += 1
        touching = touches

    rc.set_start_update(program["start"], update, program["update_slow"])
    rc.go()
    return Drive(laps=turned / (2.0 * math.pi), wall_contacts=contacts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=60.0, help="simulated time")
    parser.add_argument(
        "--direction",
        choices=get_args(policy.Direction.__value__),
        default="counterclockwise",
    )
    parser.add_argument("--speed-limit", type=float, default=1.0)
    args = parser.parse_args()
    result = drive(args.seconds, args.direction, args.speed_limit)
    print(
        f"[drive_v2] {args.seconds:.0f} s {args.direction}: {result.laps:+.2f} laps, "
        f"{result.wall_contacts} wall contacts"
    )


if __name__ == "__main__":
    main()
