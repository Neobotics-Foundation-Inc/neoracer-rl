"""
Train a PPO policy to race a car that a controller drives, or evaluate a
saved policy in such races.

The learner starts at the track's start, and the opponent, which a
controller of neoracer_mujoco.control drives, --offset meters ahead of it.
The controller holds about 1.75 m/s, half the learner's top speed, so a
learner that laps fast has to pass it. The learner sees CarObservation, its
own sensors: the opponent shows only in its LiDAR. Its task is the lap task
of task.py, where touching the other car ends the race as a crash does
(RaceEnds, RaceReward). Training is train_ppo.train on races, a race per
--num-envs.

Evaluation races with the policy's mean action until each race's first
episode ends, and prints how each ended, the laps of both cars, and how far
ahead of the opponent the learner finished. A policy that train_ppo trained
alone evaluates here too.

A run can start from a policy that train_ppo trained alone (--init-from),
so that it learns to race from a car that already laps.

Usage:
    python3 -m examples.train_race --total-steps 400000 --seed 1 --out race.npz
    python3 -m examples.train_race --init-from ppo.npz --out race.npz
    python3 -m examples.train_race --eval race.npz --races 20
"""

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from typing import Literal, assert_never

import jax.numpy as jnp
import numpy as np
from neoracer_mujoco.track_generation import shortest_loop_distance

from examples.train_ppo import (
    Exact,
    Fresh,
    Init,
    Outcome,
    PPOParams,
    Resume,
    Trained,
    greedy_action,
    load_policy,
    normalize_obs,
    save_policy,
    train,
)
from neoracer_rl.envs import (
    NeoRacerEnvConfig,
    RaceConfig,
    Racer,
    TrackCentering,
    WallFollowing,
)
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerRace
from neoracer_rl.envs.neoracer_env import substeps_per_action
from neoracer_rl.envs.task import Ends, LapEnds, ProgressReward, Rows, Step


@dataclass(frozen=True)
class RaceEnds:
    """LapEnds, with touch after crash: the car touched another car."""

    lap: LapEnds = LapEnds()

    def __call__(self, step: Step) -> Ends:
        touch = np.zeros(len(step.contact), np.bool_)
        for rival in step.rivals:
            touch = touch | rival.contact
        ends: Ends = {}
        for name, mask in self.lap(step).items():
            ends[name] = mask
            if name == "crash":
                ends["touch"] = touch
        return ends


@dataclass(frozen=True)
class RaceReward:
    """ProgressReward, less touch_penalty at a touch."""

    progress: ProgressReward = ProgressReward()
    touch_penalty: float = 5.0
    """The crash penalty."""

    def __call__(self, step: Step, ends: Ends) -> Rows:
        touched = np.where(ends["touch"], self.touch_penalty, 0.0)
        return self.progress(step, ends) - touched


type Opponent = Literal["track-centering", "wall-following"]


def race_config(opponent: Opponent, offset: float) -> RaceConfig:
    """A learner at the track's start and opponent offset meters ahead, on
    loop_corridor."""
    match opponent:
        case "track-centering":
            driver: TrackCentering | WallFollowing = TrackCentering()
        case "wall-following":
            driver = WallFollowing()
        case _:
            assert_never(opponent)
    env = NeoRacerEnvConfig(terminate=RaceEnds(), reward=RaceReward())
    return RaceConfig(env, (Racer(), Racer(driver, (offset, 0.0))))


@dataclass(frozen=True)
class RaceResult:
    outcome: Outcome
    laps: tuple[float, float]
    """The laps of the learner and of the opponent."""
    lead: float
    """How far ahead of the opponent the learner finished, in laps."""
    seconds: float
    """The time the race took."""


def evaluate_races(
    policy: Trained, races: int, seed: int, config: RaceConfig, nthread: int
) -> list[RaceResult]:
    """The first episode of each of races, from the spawns of
    default_rng(seed), with the policy's mean action."""
    env = NeoRacerRace(Cpu(nthread=min(nthread, races)), races, config)
    model = env.scene.model
    period = substeps_per_action(config.env, model) * float(model.opt.timestep)
    state = env.reset(np.random.default_rng(seed))
    head_start = state.spawn_angle[:, 1] - state.spawn_angle[:, 0]
    gap = shortest_loop_distance(head_start, 2.0 * np.pi) / (2.0 * np.pi)
    results: dict[int, RaceResult] = {}
    steps = 0
    while len(results) < races:
        obs = normalize_obs(
            policy.stats, state.obs.astype(np.float64), policy.params.obs_clip
        )
        action = np.asarray(greedy_action(policy.net, jnp.asarray(obs)))
        state, out = env.step(state, action)
        steps += 1
        for r in np.flatnonzero(out.done):
            if r in results:
                continue
            code = out.reason[r]
            mine, theirs = float(out.laps[r, 0]), float(out.laps[r, 1])
            results[int(r)] = RaceResult(
                out.ends[code - 1] if code else "truncated",
                (mine, theirs),
                mine - theirs - float(gap[r]),
                steps * period,
            )
    env.close()
    return [results[r] for r in range(races)]


def print_races(name: str, results: list[RaceResult], seed: int) -> None:
    print(f"[eval] {name}: {len(results)} races from default_rng({seed})")
    for k, r in enumerate(results):
        print(
            f"  race {k:3d}  {r.seconds:5.1f} s  laps {r.laps[0]:+.3f} vs "
            f"{r.laps[1]:+.3f}  lead {r.lead:+.3f}  outcome {r.outcome}"
        )
    ahead = sum(r.lead > 0 for r in results)
    laps = [r.seconds for r in results if r.outcome == "lap_complete"]
    lap_time = f"{np.mean(laps):.1f} s" if laps else "none"
    print(
        f"  ahead at the end {ahead}/{len(results)}  mean lap {lap_time}  "
        f"outcomes {dict(Counter(r.outcome for r in results))}"
    )


def parse_opponent(text: str) -> Opponent:
    match text:
        case "track-centering" | "wall-following":
            return text
        case _:
            raise argparse.ArgumentTypeError(
                f"opponent must be track-centering or wall-following, not {text!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="env steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8, help="races")
    parser.add_argument(
        "--nthread", type=int, default=os.cpu_count() or 1, help="CPU threads"
    )
    parser.add_argument("--opponent", type=parse_opponent, default="track-centering")
    parser.add_argument(
        "--offset", type=float, default=1.0, help="the opponent's head start (m)"
    )
    parser.add_argument("--out", help="save the trained policy to this .npz file")
    parser.add_argument(
        "--init-from", metavar="PATH", help="train on from a saved policy"
    )
    parser.add_argument("--eval", metavar="PATH", help="evaluate a saved policy")
    parser.add_argument("--races", type=int, default=20, help="evaluation races")
    parser.add_argument("--eval-seed", type=int, default=10_000)
    args = parser.parse_args()
    params = PPOParams()
    config = race_config(args.opponent, args.offset)

    if args.eval is not None:
        net, stats = load_policy(args.eval)
        policy = Trained(net, stats, params)
        results = evaluate_races(
            policy, args.races, args.eval_seed, config, args.nthread
        )
        print_races(f"policy {args.eval}", results, args.eval_seed)
        return

    print(f"[train_race] {params}")
    print(
        f"[train_race] total_steps={args.total_steps} seed={args.seed} "
        f"races={args.num_envs} opponent={args.opponent} offset={args.offset} "
        f"init_from={args.init_from}"
    )
    backend = Cpu(nthread=min(args.nthread, args.num_envs))
    init: Init = (
        Fresh() if args.init_from is None else Resume(*load_policy(args.init_from))
    )
    net, stats = train(
        params,
        args.total_steps,
        args.seed,
        args.num_envs,
        backend,
        config,
        Exact(),
        init,
    )
    if args.out is not None:
        save_policy(args.out, net, stats)
        print(f"[train_race] saved the policy to {args.out}")


if __name__ == "__main__":
    main()
