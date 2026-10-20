"""
Record a video of a policy driving the loop track.

    python3 users/aayan/record.py runs/sarsa.npz          # a trained table
    python3 users/aayan/record.py --random               # the baseline
    python3 users/aayan/record.py runs/sarsa.npz --chase # chase camera

Writes an mp4 to runs/. Headless -- no window, no mjpython, so it works over
ssh and while something else is training.

The environment offers render_mode "human", a passive viewer, so there is no
rgb_array to ask it for. This reads env.model and env.data, which the env
exposes and which are plain MuJoCo objects, and drives its own offscreen
mujoco.Renderer. Nothing here writes to the env.
"""

import argparse
import pathlib

import imageio.v2 as imageio
import mujoco
import numpy as np

import tabular
from neoracer_rl.envs import NeoRacerEnv

_HERE = pathlib.Path(__file__).resolve().parent
RUNS = _HERE / "runs"

# The compiled model's offscreen framebuffer is 640x480 unless the XML raises
# it, and asking the renderer for more than that is an error rather than a
# resize -- so stay inside it.
WIDTH, HEIGHT = 640, 480


def _camera(model, chase: bool):
    cam = mujoco.MjvCamera()
    if chase:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body("car").id
        cam.distance, cam.elevation, cam.azimuth = 1.6, -25.0, 0.0
    else:
        # Overhead view framed to whatever track this is, so a lap reads as a
        # lap. Generated tracks (track_generation/) run 50 m+ against
        # loop_corridor's 8x6, so a fixed distance would either crop them or
        # shrink the car to nothing. Fit to the actual wall extent instead.
        # Take each scenery geom's EXTENT, not just its centre: a track whose
        # floor is one big plane has a single centre point, which would collapse
        # the span to zero and zoom absurdly far in.
        xs, ys = [], []
        for i in range(model.ngeom):
            name = model.geom(i).name or ""
            if "wall" not in name and "floor" not in name:
                continue
            px, py = float(model.geom_pos[i][0]), float(model.geom_pos[i][1])
            sx, sy = float(model.geom_size[i][0]), float(model.geom_size[i][1])
            xs += [px - sx, px + sx]
            ys += [py - sy, py + sy]
        if xs:
            cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
            span = max(max(xs) - min(xs), max(ys) - min(ys), 4.0)
        else:
            cx, cy, span = 0.0, 0.0, 8.0
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (cx, cy, 0.0)
        cam.distance = span * 1.35
        cam.elevation, cam.azimuth = -89.0, 90.0
    return cam


def record(policy, out: pathlib.Path, seed: int, fps: int, chase: bool,
           max_seconds: float, label: str) -> None:
    env = NeoRacerEnv()
    obs, _ = env.reset(seed=seed)

    cam = _camera(env.model, chase)
    # The env steps physics several times per action (30 Hz control over a
    # ~500 Hz sim), so one frame per env.step already gives real-time video.
    frames = []
    total, steps, done = 0.0, 0, False
    info: dict = {}
    max_steps = int(max_seconds * 30)

    with mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH) as renderer:
        while not done and steps < max_steps:
            if chase:
                cam.azimuth = float(
                    np.degrees(
                        np.arctan2(
                            2 * (env.data.xquat[env.model.body("car").id][0]
                                 * env.data.xquat[env.model.body("car").id][3]),
                            1 - 2 * env.data.xquat[env.model.body("car").id][3] ** 2,
                        )
                    )
                )
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())

            action = policy(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total += float(r)
            steps += 1
            done = bool(terminated or truncated)

    env.close()

    out.parent.mkdir(exist_ok=True)
    # Name the plugin rather than letting imageio choose. Left to itself it
    # picks PyAV here, which writes a 0-byte file and dies inside av on a
    # time_base of None. imageio-ffmpeg is the dependable mp4 writer.
    imageio.mimwrite(str(out), frames, format="FFMPEG", fps=fps, quality=8,
                     macro_block_size=1)
    print(
        f"{label}: {steps} steps ({steps/30:.1f}s sim), return {total:+.2f}, "
        f"lap {float(info.get('lap_frac', 0.0)):.3f}, "
        f"{info.get('termination_reason') or 'truncated'}"
    )
    print(f"  -> {out}  ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", nargs="?", help="a runs/*.npz saved by tabular.py")
    ap.add_argument("--random", action="store_true", help="record a random policy instead")
    ap.add_argument("--out", help="output mp4 path")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chase", action="store_true", help="chase cam instead of overhead")
    ap.add_argument("--max-seconds", type=float, default=70.0)
    a = ap.parse_args()

    if a.random:
        rng = np.random.default_rng(a.seed)
        policy = lambda _obs: rng.uniform(-1, 1, size=2).astype(np.float32)  # noqa: E731
        name = "random"
    else:
        if not a.table:
            ap.error("give a runs/*.npz table, or --random")
        Q = np.load(a.table)["Q"]
        policy = lambda obs: np.array(  # noqa: E731
            tabular.ACTIONS[int(np.argmax(Q[tabular.discretize(obs)]))], dtype=np.float32
        )
        name = pathlib.Path(a.table).stem

    out = pathlib.Path(a.out) if a.out else RUNS / f"{name}{'_chase' if a.chase else ''}.mp4"
    record(policy, out, a.seed, a.fps, a.chase, a.max_seconds, name)
