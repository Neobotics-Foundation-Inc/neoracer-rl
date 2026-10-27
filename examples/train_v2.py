"""
Train the RACECAR Neo V2's driving policy, racecar_neo_v2/policy.py, with PPO
in the car's MuJoCo model, or evaluate a trained one. --tracks loop trains
on loop_corridor, --tracks generated on a pool of generated tracks.

The policy sees what the car's racecar_core shows a program. The V2's model
reads its LiDAR beams as racecar_core reports them: a whole turn of the head
at a time, up to two turns old (neoracer-mujoco's mjcf.beam_timing). Each
episode draws the car's physics from PHYSICS, the LiDAR's scan rate among
them. At each control step, sense() adds what the model does not have:
- A range has noise, and a beam now and then reads no return. Both stay the
  same until the beams show the next scan, as racecar_core repeats a scan.
- The encoder reads the wheels' speed times a factor that each episode draws.
- In some episodes, each action reaches the car a control step late.

The network, the rollout and the training are train_ppo.py's; this file
gives them the car's sensing, RacecarCore. --eval drives each car with
policy.act, the code that runs on the car.

Usage:
    python3 -m examples.train_v2 --total-steps 2500000 --seed 0 --out v2_policy.npz
    python3 -m examples.train_v2 --eval examples/racecar_neo_v2/v2_policy.npz
"""

import argparse
import os
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from neoracer_mujoco import assets
from neoracer_mujoco.physics import PhysicsRanges, Range
from neoracer_mujoco.racecar import racecar_neo_v2_params

from examples.racecar_neo_v2 import policy
from examples.train_ppo import (
    ACTION_DIM,
    ActorCritic,
    EpisodeResult,
    ObsStats,
    PPOParams,
    add_track_arguments,
    ended,
    parse_track_arguments,
    policy_arrays,
    train,
)
from neoracer_rl.envs import NeoRacerEnvConfig, task
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch
from neoracer_rl.envs.race import RaceConfig, Racer

CONFIG = NeoRacerEnvConfig(
    car_path=os.path.join(assets.ASSET_DIR, "racecar_neo_v2.xml"),
    car=racecar_neo_v2_params(),
)
"""The env on the V2, whose model holds its own speed: action[0] times
max_speed is its throttle."""

PHYSICS = PhysicsRanges(
    wheel_friction=Range(0.6, 1.2),
    chassis_mass=Range(0.8, 1.3),
    motor_torque=Range(0.7, 1.4),
    steer_stiffness=Range(0.7, 1.4),
    lidar_rate=Range(0.95, 1.05),
)
"""The physics ranges of training: wide enough for the policy to drive a car
that differs from the model, not measured on the real car."""

type Array = npt.NDArray[np.float64]
type EnvObservations = npt.NDArray[np.float32]
"""Shape (N, task.OBS_DIM)."""


@dataclass(frozen=True)
class SensingParams:
    """What the car shows the policy that the model does not: the noise of
    its LiDAR, its encoder's error and its late actions. Each episode draws
    its own value from each range."""

    range_noise: float = 0.02
    """The standard deviation of a range (m)."""
    dropout: float = 0.02
    """The chance that a beam reads no return."""
    speed_scale: Range = Range(0.95, 1.2)
    """The encoder's speed over the wheels'."""
    late: float = 0.5
    """The share of episodes whose actions arrive a step late."""

    def __post_init__(self) -> None:
        if self.range_noise < 0.0:
            raise ValueError(f"range_noise must be >= 0, not {self.range_noise}")
        if not 0.0 <= self.dropout <= 1.0 or not 0.0 <= self.late <= 1.0:
            raise ValueError("dropout and late must be in [0, 1]")


@dataclass(frozen=True)
class Sensing:
    """What each car shows the policy, between two control steps."""

    scan: Array
    """The model's last scan of true ranges (m), shape (N, beams).
    racecar_core repeats a scan until the next, with its noise and its lost
    beams."""
    error: Array
    """The noise of its ranges (m), shape (N, beams)."""
    lost: npt.NDArray[np.bool_]
    """Its beams with no return, shape (N, beams)."""
    speed_scale: Array
    """Shape (N,)."""
    late: npt.NDArray[np.bool_]
    """The car's actions arrive a step late, shape (N,)."""
    last_action: Array
    """The policy's last action, clipped, shape (N, ACTION_DIM)."""


def true_ranges(obs: EnvObservations) -> Array:
    """The ranges (N, beams) m of the env's LiDAR beams, at BEAM_DEGREES."""
    return policy.clip_ranges(obs[:, : len(policy.BEAM_DEGREES)])


def wheel_speed(obs: EnvObservations) -> Array:
    """The wheels' speed (N,) m/s, from the env's observation."""
    return obs[:, task.OBS_DIM - 1].astype(np.float64)


def start(params: SensingParams, ranges: Array, rng: np.random.Generator) -> Sensing:
    """The sensing of cars at their spawns, whose beams read ranges (N, beams)."""
    n = ranges.shape[0]
    return Sensing(
        scan=ranges,
        error=rng.normal(0.0, params.range_noise, ranges.shape),
        lost=rng.random(ranges.shape) < params.dropout,
        speed_scale=rng.uniform(params.speed_scale.low, params.speed_scale.high, n),
        late=rng.random(n) < params.late,
        last_action=np.zeros((n, ACTION_DIM)),
    )


def restart(
    params: SensingParams,
    s: Sensing,
    rows: npt.NDArray[np.bool_],
    ranges: Array,
    rng: np.random.Generator,
) -> Sensing:
    """s with the cars in rows (N,) started over at their spawns."""
    new = start(params, ranges, rng)

    def pick[T: np.generic](
        fresh: npt.NDArray[T], old: npt.NDArray[T]
    ) -> npt.NDArray[T]:
        return np.where(rows.reshape(-1, *[1] * (old.ndim - 1)), fresh, old)

    return Sensing(
        scan=pick(new.scan, s.scan),
        error=pick(new.error, s.error),
        lost=pick(new.lost, s.lost),
        speed_scale=pick(new.speed_scale, s.speed_scale),
        late=pick(new.late, s.late),
        last_action=pick(new.last_action, s.last_action),
    )


def sense(
    params: SensingParams,
    s: Sensing,
    ranges: Array,
    speed: Array,
    rng: np.random.Generator,
) -> tuple[Sensing, policy.Observation]:
    """The sensing a control step later, and the policy's observations, from
    the ranges (N, beams) m that the cars' beams read then and their wheel
    speeds (N,) m/s."""
    # The beams change when the LiDAR's head ends a turn: a new scan, with
    # new noise and new lost beams.
    new = np.any(ranges != s.scan, axis=1)[:, None]
    error = np.where(new, rng.normal(0.0, params.range_noise, ranges.shape), s.error)
    lost = np.where(new, rng.random(ranges.shape) < params.dropout, s.lost)
    scan = np.clip(
        np.where(lost, policy.MAX_RANGE, ranges + error), 0.0, policy.MAX_RANGE
    )
    s = replace(s, scan=ranges, error=error, lost=lost)
    return s, policy.observation(scan, speed * s.speed_scale, s.last_action)


def delay(s: Sensing, action: Array) -> tuple[Sensing, Array]:
    """The sensing after the policy acted, and the actions (N, ACTION_DIM)
    that reach the cars: on a late car, the policy's last action."""
    clipped = np.clip(action, -1.0, 1.0)
    applied = np.where(s.late[:, None], s.last_action, clipped)
    return replace(s, last_action=clipped), applied


@dataclass(frozen=True)
class RacecarCore:
    """What the car's racecar_core shows the policy, and the actions that
    reach the car, as a train_ppo.Interface."""

    params: SensingParams

    def obs_dim(self, env_obs_dim: int) -> int:
        return policy.OBS_DIM

    def start(self, obs: EnvObservations, rng: np.random.Generator) -> Sensing:
        return start(self.params, true_ranges(obs), rng)

    def observe(
        self, s: Sensing, obs: EnvObservations, rng: np.random.Generator
    ) -> tuple[Sensing, policy.Observation]:
        return sense(self.params, s, true_ranges(obs), wheel_speed(obs), rng)

    def act(self, s: Sensing, action: Array) -> tuple[Sensing, Array]:
        return delay(s, action)

    def restart(
        self,
        s: Sensing,
        rows: npt.NDArray[np.bool_],
        obs: EnvObservations,
        rng: np.random.Generator,
    ) -> Sensing:
        return restart(self.params, s, rows, true_ranges(obs), rng)


# --- Effects: environments, files ------------------------------------------


def save_policy(
    path: str,
    net: ActorCritic,
    stats: ObsStats,
    params: PPOParams,
    config: NeoRacerEnvConfig,
) -> None:
    """The arrays of train_ppo.save_policy, which load_policy reads, and
    what policy.load needs beside them: obs_clip and throttle_scale, the
    config's max_speed."""
    np.savez(
        path,
        allow_pickle=False,
        **policy_arrays(net, stats),
        obs_clip=np.array(params.obs_clip),
        throttle_scale=np.array(config.max_speed),
    )


def evaluate(
    car: policy.Policy,
    sensing_params: SensingParams,
    episodes: int,
    seed: int,
    nthread: int,
    config: NeoRacerEnvConfig,
) -> list[EpisodeResult]:
    """One episode on each of episodes cars at once, each driven by
    policy.act. The spawns, the physics and the sensing come from
    default_rng(seed)."""
    env = NeoRacerBatch(Cpu(nthread=min(nthread, episodes)), episodes, config)
    interface = RacecarCore(sensing_params)
    rng = np.random.default_rng(seed)
    state = env.reset(rng)
    sensing = interface.start(state.obs, rng)
    returns, steps = np.zeros(episodes), 0
    results: dict[int, EpisodeResult] = {}
    while len(results) < episodes:
        sensing, obs = interface.observe(sensing, state.obs, rng)
        sensing, applied = interface.act(sensing, policy.act(car, obs))
        state, out = env.step(state, applied)
        returns += out.reward
        steps += 1
        # Each row's first episode, which started with the others.
        for i, result in ended(out, returns, np.full(episodes, steps)).items():
            results.setdefault(i, result)
    env.close()
    return [results[i] for i in range(episodes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--total-steps", type=int, default=2_500_000, help="env steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--nthread",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads that step the cars",
    )
    parser.add_argument("--out", help="save the trained policy to this .npz file")
    parser.add_argument("--eval", metavar="PATH", help="evaluate a saved policy")
    parser.add_argument("--episodes", type=int, default=32, help="evaluation episodes")
    add_track_arguments(parser, default="loop")
    args = parser.parse_args()
    params, sensing_params = PPOParams(), SensingParams()
    tracks = parse_track_arguments(args)
    config = replace(CONFIG, tracks=tracks, physics=PHYSICS)

    if args.eval is not None:
        car = policy.load(args.eval)
        results = evaluate(
            car,
            sensing_params,
            args.episodes,
            args.seed + 10_000,
            args.nthread,
            # The throttle that action 1 commands, as on the car.
            replace(config, max_speed=car.throttle_scale),
        )
        laps = [
            r.steps * policy.CONTROL_PERIOD
            for r in results
            if r.outcome == "lap_complete"
        ]
        print(
            f"[eval] {args.eval} on {tracks}: {len(results)} episodes, "
            f"outcomes {dict(Counter(r.outcome for r in results))}, "
            f"mean lap_frac {np.mean([r.lap_frac for r in results]):.3f}, "
            f"mean lap time {np.mean(laps) if laps else float('nan'):.1f} s"
        )
        return

    print(f"[train_v2] {params}")
    print(f"[train_v2] {sensing_params}")
    print(
        f"[train_v2] total_steps={args.total_steps} seed={args.seed} "
        f"num_envs={args.num_envs} tracks={tracks} physics={PHYSICS}"
    )
    net, stats = train(
        params,
        args.total_steps,
        args.seed,
        args.num_envs,
        Cpu(nthread=min(args.nthread, args.num_envs)),
        RaceConfig(config, (Racer(),)),
        RacecarCore(sensing_params),
    )
    if args.out is not None:
        save_policy(args.out, net, stats, params, config)
        print(f"[train_v2] saved the policy to {args.out}")


if __name__ == "__main__":
    main()
