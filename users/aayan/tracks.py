"""
Build a drivable scene: a track with the car on it.

The simulator does the work. assets.compose attaches the car to a track spec
and returns a Scene, whose .model is the compiled MjModel this module hands
back. track_generation builds procedural tracks. This module only names the
tracks I use and remembers where the car belongs on each one.

A track name is either one of the shipped tracks (run this file to list them)
or "gen:<seed>" / "gen:<seed>:<difficulty>" for a generated loop. Difficulty
runs from 0, which is wide and gentle, to 3, which is tight and narrow. The
seed makes a generated track reproducible, so a seed worth returning to is
worth writing down.
"""

import os

import mujoco
import numpy as np
from neoracer_mujoco import assets
from neoracer_mujoco.track_generation import (
    Difficulty,
    NoValidTrack,
    Track,
    generate_track,
    settings_for_difficulty,
    to_spec,
)

GEN_PREFIX = "gen:"

# Where the car starts on each shipped track, in metres, as the offset that
# assets.compose applies. loop_corridor is a ring between an inner and an outer
# wall, so a car left at the origin lands inside the walled island in the
# middle. Its bottom straight sits at y = -2.5, halfway between the inner wall
# at y = -2 and the outer wall at y = -3. Every other track starts the car at
# its origin, and NeoRacerEnv uses the same offset for loop_corridor.
SPAWN: dict[str, assets.Position] = {"loop_corridor": (0.0, -2.5, 0.0)}


def shipped() -> list[str]:
    """The names of the track XMLs that come with the simulator."""
    return [os.path.splitext(os.path.basename(p))[0] for p in assets.tracks()]


def parse_generated(name: str) -> tuple[int, Difficulty] | None:
    """Read "gen:7" or "gen:7:3" as a seed and a difficulty. Other names give None."""
    if not name.startswith(GEN_PREFIX):
        return None
    parts = name[len(GEN_PREFIX) :].split(":")
    try:
        seed = int(parts[0])
        level = int(parts[1]) if len(parts) > 1 else 1
    except ValueError:
        raise SystemExit(
            f"cannot read {name!r}. Write gen:<seed> or gen:<seed>:<difficulty>."
        ) from None
    if level not in (0, 1, 2, 3):
        raise SystemExit(f"difficulty runs from 0 to 3, so {level} is out of range.")
    return seed, level


def _generated_track(seed: int, difficulty: Difficulty) -> Track:
    """The Track for a generated loop, or an error if none could be drawn."""
    match generate_track(seed, settings_for_difficulty(difficulty)):
        case Track() as track:
            return track
        case NoValidTrack(attempts=attempts):
            raise SystemExit(
                f"no track for seed {seed} at difficulty {difficulty} "
                f"after {attempts} attempts. Try another seed."
            )


def generated_spec(seed: int, difficulty: Difficulty) -> mujoco.MjSpec:
    """The MuJoCo spec for a procedural loop.

    generate_track returns Track or NoValidTrack, so the failure is a value to
    match on rather than an exception to catch.
    """
    return to_spec(_generated_track(seed, difficulty))


def track_path(name: str) -> str:
    """The XML path of a shipped track, or of a path given as-is."""
    if os.path.exists(name):
        return name
    path = os.path.join(assets.TRACK_DIR, f"{name}.xml")
    if not os.path.exists(path):
        raise SystemExit(
            f"no track called {name!r}. The shipped tracks are "
            f"{', '.join(shipped())}, or use gen:<seed>[:<difficulty>]."
        )
    return path


def load(name: str = "loop_corridor") -> mujoco.MjModel:
    """Compile a track with the car on it. See the module docstring for names."""
    if (generated := parse_generated(name)) is not None:
        return assets.compose(generated_spec(*generated)).model

    path = track_path(name)
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = mujoco.MjSpec.from_file(path)
    return assets.compose(spec, pos=SPAWN.get(stem, (0.0, 0.0, 0.0))).model


# loop_corridor is a ring between an outer wall at x = +-4, y = +-3 and an
# inner wall at x = +-3, y = +-2, with the corners cut at 45 degrees. Halfway
# between the two is an octagon through these corners, which is the line a car
# should follow. The order is the car's forward direction: it starts at
# (0, -2.5) facing +X, so the loop runs anticlockwise.
_LOOP_CORRIDOR_CORNERS = (
    (3.0, -2.5), (3.5, -2.0), (3.5, 2.0), (3.0, 2.5),
    (-3.0, 2.5), (-3.5, 2.0), (-3.5, -2.0), (-3.0, -2.5),
)


def centreline(name: str = "loop_corridor", spacing_m: float = 0.05) -> np.ndarray:
    """Points along the middle of the track, in metres, as an (N, 2) array.

    The points are roughly spacing_m apart and the last joins back to the
    first. A generated track carries its own centre line; loop_corridor's is
    the octagon above. No other shipped track is a closed loop, so there is
    nothing to follow on one.
    """
    if (generated := parse_generated(name)) is not None:
        track = _generated_track(*generated)
        return np.asarray(track.center, dtype=float)
    if name != "loop_corridor":
        raise SystemExit(
            f"no centre line for {name!r}. Use loop_corridor or gen:<seed>, "
            "the two tracks that are closed loops."
        )
    corners = np.array(_LOOP_CORRIDOR_CORNERS, dtype=float)
    points = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        steps = max(2, int(round(float(np.hypot(*(end - start))) / spacing_m)))
        points.append(start + np.outer(np.arange(steps) / steps, end - start))
    return np.concatenate(points)


def describe(model: mujoco.MjModel) -> str:
    """One line about a compiled scene. Useful when a track looks wrong."""
    floors = [
        model.geom(i).name
        for i in range(model.ngeom)
        if int(model.geom(i).type[0]) == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    floor = floors[0] if floors else "none, so the car will fall"
    return f"ngeom={model.ngeom} nu={model.nu} nsensor={model.nsensor} floor={floor}"


if __name__ == "__main__":
    for name in [*shipped(), "gen:1", "gen:7:0", "gen:7:3"]:
        model = load(name)
        data = mujoco.MjData(model)
        for _ in range(600):
            mujoco.mj_step(model, data)
        height = data.body("car").xpos[2]
        print(f"{name:20s} {describe(model):56s} car height after 1 s = {height:+.4f} m")
