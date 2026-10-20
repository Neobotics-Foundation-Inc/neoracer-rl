"""
Score the classical controllers, tune their gains, and race them.

    python3 users/aayan/run.py solo                      # score every controller
    python3 users/aayan/run.py solo --episodes 5
    python3 users/aayan/run.py tune pd_centering         # sweep one controller's gains
    python3 users/aayan/run.py duel --agent runs/sarsa.npz --opponent pure_pursuit

solo runs each controller alone in NeoRacerEnv, the same environment the
tabular agents train in, so the numbers go straight into the table in
README.md. tune sweeps the gains of one controller and prints the best few.
duel puts a saved table against a controller on the same track.
"""

import argparse
import itertools
import json

import numpy as np
from neoracer_rl.envs import NeoRacerEnv

import baselines
import duel
import multicar
import store
import tabular
import tracks

CONTROLLERS = ("pd_centering", "wall_follow", "follow_the_gap", "pure_pursuit")


def solo(controller, episodes: int = 5, seed_base: int = 1000, track: str = "loop_corridor"):
    """Run one controller alone and return a row per episode.

    The controller sees the observation NeoRacerEnv gives. A privileged one
    also gets the car's pose, read from the environment's public scene.
    """
    env = NeoRacerEnv()
    needs_pose = getattr(controller, "is_privileged", False)
    dt = env.effective_control_dt
    rows = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        controller.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            pose = multicar.pose_of(env.data, env.scene.car) if needs_pose else None
            throttle, steer = controller.act(obs, dt, pose)
            obs, r, terminated, truncated, info = env.step(
                np.array([throttle, steer], dtype=np.float32)
            )
            total += float(r)
            steps += 1
            done = terminated or truncated
        rows.append(
            {
                "return": total,
                "lap": float(info["lap_frac"]),
                "steps": steps,
                "reason": info["termination_reason"] or "truncated",
            }
        )
    env.close()
    return rows


CONTROL_HZ = 30.0


def summarise(rows) -> dict:
    """Mean over episodes. lap_time_s counts only the episodes that finished a
    lap, so it is nan for a controller that never does."""
    laps = [r["lap"] for r in rows]
    finished = [r["steps"] for r in rows if r["reason"] == "lap_complete"]
    return {
        "return": float(np.mean([r["return"] for r in rows])),
        "lap": float(np.mean(laps)),
        "best_lap": float(np.max(laps)),
        "lap_time_s": float(np.mean(finished)) / CONTROL_HZ if finished else float("nan"),
        "laps_done": len(finished),
        "n": len(rows),
    }


def build(name: str, track: str = "loop_corridor", **gains):
    """A controller by name, with any of its gains replaced."""
    match name:
        case "pd_centering":
            return baselines.PDCentering(baselines.CenteringParams(**gains))
        case "wall_follow":
            return baselines.WallFollow(baselines.WallFollowParams(**gains))
        case "follow_the_gap":
            return baselines.FollowTheGap(baselines.GapParams(**gains))
        case "pure_pursuit":
            return baselines.PurePursuit(
                tracks.centreline(track), baselines.PursuitParams(**gains)
            )
        case other:
            raise SystemExit(f"unknown controller {other!r}. Pick one of {CONTROLLERS}.")


# The gains each controller is swept over, and the values tried for each.
SWEEPS: dict[str, dict[str, tuple]] = {
    "pd_centering": {
        "steer": tuple(
            baselines.PDGains(kp=kp, kd=kd)
            for kp in (0.6, 0.9, 1.2, 1.6, 2.2)
            for kd in (0.0, 0.15, 0.3)
        ),
        "target_ms": (1.0, 1.4, 1.8, 2.2),
    },
    "wall_follow": {
        "steer": tuple(
            baselines.PDGains(kp=kp, kd=kd)
            for kp in (1.6, 2.2, 3.0)
            for kd in (0.15, 0.3)
        ),
        "standoff_m": (0.42, 0.52),
        "target_ms": (1.6, 2.0, 2.4, 2.8),
    },
    "follow_the_gap": {
        "steer_gain": (0.5, 0.85, 1.2),
        "smoothing": (0.35, 0.55, 0.8),
        "target_ms": (1.2, 1.6, 2.0),
    },
    "pure_pursuit": {
        "lookahead_m": (0.5, 0.75, 1.0),
        "lookahead_per_ms": (0.15, 0.28, 0.4),
        "target_ms": (1.6, 2.0, 2.5, 3.0),
    },
}


def tune(name: str, episodes: int = 3, track: str = "loop_corridor", show: int = 6):
    """Sweep one controller's gains and print the best settings found.

    The score is the mean lap fraction, with the mean return breaking ties,
    because finishing a lap matters more than how it was scored along the way.
    """
    grid = SWEEPS[name]
    keys = tuple(grid)
    combinations = list(itertools.product(*(grid[k] for k in keys)))
    print(f"tuning {name}: {len(combinations)} settings, {episodes} episodes each")
    results = []
    for i, values in enumerate(combinations, 1):
        gains = dict(zip(keys, values, strict=True))
        stats = summarise(solo(build(name, track, **gains), episodes, track=track))
        results.append((stats, gains))
        if i % 10 == 0 or i == len(combinations):
            print(f"  {i}/{len(combinations)}")
    # Finishing comes first, then speed, then how far it got. A controller that
    # laps in 11 s beats one that laps in 14 s, and both beat one that crashes.
    def rank(row):
        stats = row[0]
        done = stats["laps_done"] / stats["n"]
        time_s = stats["lap_time_s"] if stats["laps_done"] else float("inf")
        return (-done, time_s, -stats["lap"])

    results.sort(key=rank)
    sweep_path = store.unique(f"tune_{name}", ".jsonl")
    with sweep_path.open("w") as f:
        for stats, gains in results:
            readable = {
                k: (f"kp={v.kp},kd={v.kd}" if isinstance(v, baselines.PDGains) else v)
                for k, v in gains.items()
            }
            f.write(json.dumps({"gains": readable, **stats}) + "\n")
    print(f"\nthe whole sweep is in {sweep_path}")
    print(f"\nbest {show} for {name}:")
    for stats, gains in results[:show]:
        pretty = ", ".join(
            f"{k}={v.kp}/{v.kd}" if isinstance(v, baselines.PDGains) else f"{k}={v}"
            for k, v in gains.items()
        )
        time_s = f"{stats['lap_time_s']:.1f} s" if stats["laps_done"] else "no lap"
        print(
            f"  laps {stats['laps_done']}/{stats['n']}  time {time_s:>7}  "
            f"lap {stats['lap']:.3f}  {pretty}"
        )
    best_stats, best_gains = results[0]
    store.log_result(
        "tune",
        {"controller": name, "track": track, "episodes": episodes,
         "settings_tried": len(results), "sweep": sweep_path.name,
         "best_lap_time_s": best_stats["lap_time_s"],
         "best_laps": f"{best_stats['laps_done']}/{best_stats['n']}"},
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("solo", "tune", "duel"))
    parser.add_argument("controller", nargs="?", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--track", default="loop_corridor")
    parser.add_argument("--agent", default=None, help="a table saved by tabular.py")
    parser.add_argument("--opponent", default="pd_centering")
    parser.add_argument("--abreast", action="store_true")
    args = parser.parse_args()

    match args.mode:
        case "solo":
            names = (args.controller,) if args.controller else CONTROLLERS
            print(f"{'controller':<16}{'mean return':>13}{'mean lap':>11}"
                  f"{'lap time':>11}{'laps':>9}")
            print("-" * 60)
            for name in names:
                stats = summarise(solo(build(name, args.track), args.episodes, track=args.track))
                mark = " *" if build(name, args.track).is_privileged else ""
                time_s = (
                    f"{stats['lap_time_s']:.1f} s" if stats["laps_done"] else "none"
                )
                print(
                    f"{name + mark:<16}{stats['return']:>+13.2f}{stats['lap']:>11.3f}"
                    f"{time_s:>11}{stats['laps_done']:>6}/{stats['n']}"
                )
                store.log_result(
                    "solo",
                    {"controller": name, "track": args.track,
                     "episodes": args.episodes, **{k: round(v, 4)
                     if isinstance(v, float) else v for k, v in stats.items()}},
                )
            print("\n* reads the track's centre line and the car's true pose, "
                  "which the learner never sees")
        case "tune":
            if args.controller is None:
                raise SystemExit(f"say which controller to tune: {CONTROLLERS}")
            tune(args.controller, args.episodes, args.track)
        case "duel":
            opponent = duel.make_opponent(args.opponent, args.track)
            policy = None
            if args.agent is not None:
                table = np.load(args.agent)["Q"]
                policy = lambda obs: np.array(  # noqa: E731
                    tabular.ACTIONS[int(np.argmax(table[tabular.discretize(obs)]))],
                    dtype=np.float32,
                )
            config = duel.DuelConfig(track=args.track, abreast=args.abreast)
            print(f"{args.agent or 'a still car'} against {opponent.name}")
            rows = duel.race(opponent, policy, args.episodes, config)
            for ep, (total, mine, theirs, reason) in enumerate(rows):
                print(
                    f"  episode {ep}: return {total:+7.2f}  agent lap {mine:.3f}  "
                    f"opponent lap {theirs:.3f}  ended {reason or 'truncated'}"
                )
            store.log_result(
                "duel",
                {"agent": args.agent or "still", "opponent": args.opponent,
                 "track": args.track, "abreast": args.abreast,
                 "episodes": len(rows),
                 "agent_lap": round(float(np.mean([r[1] for r in rows])), 4),
                 "opponent_lap": round(float(np.mean([r[2] for r in rows])), 4),
                 "agent_finished": sum(r[3] == "lap_complete" for r in rows)},
            )


if __name__ == "__main__":
    main()
