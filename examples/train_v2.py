"""
Train the RACECAR Neo V2's driving policy, racecar_neo_v2/policy.py, with PPO
in the car's MuJoCo model on loop_corridor, or evaluate a trained one.

The policy sees what the car's racecar_core shows a program. At each
control step, NeoRacerBatch gives each car's true LiDAR ranges and wheel
speed, and sense() turns them into what the car would show:
- The LiDAR's head turns about 7.4 times a second and measures each
  direction as it passes it, and racecar_core gives the last complete turn.
  So each range is as the head saw it, up to two turns ago. Each episode
  draws the time of a turn, the direction at which a turn begins, and when
  the first turn ends. A range has noise, and a beam now and then reads no
  return.
- The encoder reads the wheels' speed times a factor that each episode draws.
- In some episodes, each action reaches the car a control step late.
Each episode also draws the car's physics from PHYSICS.

The network, the loss, the update and the observation statistics are those
of train_ppo.py; the observation and the rollout are this file's. --eval
drives each car with policy.act, the code that runs on the car.

Usage:
    python3 -m examples.train_v2 --total-steps 2500000 --seed 0 --out v2_policy.npz
    python3 -m examples.train_v2 --eval examples/racecar_neo_v2/v2_policy.npz
"""

import argparse
import os
import time
from collections import Counter
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from neoracer_mujoco import assets, sensors
from neoracer_mujoco.physics import PhysicsRanges, Range
from neoracer_mujoco.racecar_core_adapter import racecar_neo_v2_params
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2

from examples.racecar_neo_v2 import policy
from examples.train_ppo import (
    ACTION_DIM,
    ActorCritic,
    Batch,
    EpisodeResult,
    ObsStats,
    PPOParams,
    UpdateLog,
    format_log,
    gae,
    init_mlp,
    init_opt_state,
    normalize_obs,
    ppo_update,
    sample_action,
    update_obs_stats,
    value,
)
from neoracer_rl.envs import LoopCorridor, NeoRacerEnvConfig, neoracer_env
from neoracer_rl.envs.neoracer_batch import REASONS, BatchState, Cpu, NeoRacerBatch

CAR = os.path.join(assets.ASSET_DIR, "racecar_neo_v2.xml")
TYRE_RADIUS = RACECAR_NEO_V2.wheel.diameter.value / 2.0  # m
SCAN_RATE = racecar_neo_v2_params().lidar.scan_rate  # turns/s
# The physics ranges of training: wide enough for the policy to drive a car
# that differs from the model, not measured on the real car.
PHYSICS = PhysicsRanges(
    wheel_friction=Range(0.6, 1.2),
    chassis_mass=Range(0.8, 1.3),
    motor_torque=Range(0.7, 1.4),
    steer_stiffness=Range(0.7, 1.4),
)

type Array = npt.NDArray[np.float64]
type EnvObservations = npt.NDArray[np.float32]  # (N, neoracer_env.OBS_DIM)


@dataclass(frozen=True)
class SensingParams:
    """How the car shows the policy its LiDAR, its speed and its actions.
    Each episode draws its own value from each range."""

    scan_period: tuple[float, float] = (0.95 / SCAN_RATE, 1.05 / SCAN_RATE)  # s
    history: int = 10  # control steps of true ranges kept
    range_noise: float = 0.02  # m, the standard deviation of a range
    dropout: float = 0.02  # the chance that a beam reads no return
    speed_scale: Range = Range(0.95, 1.2)  # the encoder's speed over the wheels'
    late: float = 0.5  # the share of episodes whose actions arrive a step late

    def __post_init__(self) -> None:
        low, high = self.scan_period
        if not 0.0 < low <= high:
            raise ValueError(f"scan_period needs 0 < low <= high, not {low, high}")
        # A range is at most two turns old.
        if (self.history - 1) * policy.CONTROL_PERIOD < 2.0 * high:
            raise ValueError(f"history {self.history} holds less than two turns")
        if self.range_noise < 0.0:
            raise ValueError(f"range_noise must be >= 0, not {self.range_noise}")
        if not 0.0 <= self.dropout <= 1.0 or not 0.0 <= self.late <= 1.0:
            raise ValueError("dropout and late must be in [0, 1]")


@dataclass(frozen=True)
class Sensing:
    """What each car shows the policy, between two control steps."""

    history: Array  # (N, history, beams) m, the true ranges, newest first
    time: Array  # (N,) s since the car stood at its spawn
    period: Array  # (N,) s, a turn of the LiDAR's head
    first_turn: Array  # (N,) s, when the first turn ends, in [0, period)
    passes: Array  # (N, beams) when in a turn the head passes each beam, in [0, 1)
    speed_scale: Array  # (N,)
    late: npt.NDArray[np.bool_]  # (N,) the car's actions arrive a step late
    last_action: Array  # (N, ACTION_DIM) the policy's last action, clipped
    # racecar_core repeats the scan of a turn until the next turn ends, with
    # its noise and its lost beams.
    turn: Array  # (N,) the number of the last complete turn
    error: Array  # (N, beams) m, the noise of its ranges
    lost: npt.NDArray[np.bool_]  # (N, beams) its beams with no return


def true_ranges(obs: EnvObservations) -> Array:
    """The ranges (N, beams) m of the env's LiDAR beams, at BEAM_DEGREES."""
    return policy.clip_ranges(obs[:, : len(policy.BEAM_DEGREES)])


def wheel_speed(obs: EnvObservations) -> Array:
    """The wheels' speed (N,) m/s. The env gives the encoder's turn rate
    times the NeoRacer's wheel radius."""
    speed = obs[:, neoracer_env.OBS_DIM - 1].astype(np.float64)
    return speed * (TYRE_RADIUS / sensors.WHEEL_RADIUS)


def start(params: SensingParams, ranges: Array, rng: np.random.Generator) -> Sensing:
    """The sensing of cars that stood at their spawns, with ranges (N, beams)."""
    n = ranges.shape[0]
    period = rng.uniform(*params.scan_period, n)
    # The head turns clockwise, and a turn begins at a random direction.
    clockwise = (360.0 - np.array(policy.BEAM_DEGREES)) % 360.0
    begin = rng.uniform(0.0, 360.0, (n, 1))
    first_turn = rng.uniform(0.0, 1.0, n) * period
    return Sensing(
        history=np.repeat(ranges[:, None], params.history, axis=1),
        time=np.zeros(n),
        period=period,
        first_turn=first_turn,
        passes=(clockwise - begin) % 360.0 / 360.0,
        speed_scale=rng.uniform(params.speed_scale.low, params.speed_scale.high, n),
        late=rng.random(n) < params.late,
        last_action=np.zeros((n, ACTION_DIM)),
        turn=np.floor(-first_turn / period),
        error=rng.normal(0.0, params.range_noise, ranges.shape),
        lost=rng.random(ranges.shape) < params.dropout,
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
        history=pick(new.history, s.history),
        time=pick(new.time, s.time),
        period=pick(new.period, s.period),
        first_turn=pick(new.first_turn, s.first_turn),
        passes=pick(new.passes, s.passes),
        speed_scale=pick(new.speed_scale, s.speed_scale),
        late=pick(new.late, s.late),
        last_action=pick(new.last_action, s.last_action),
        turn=pick(new.turn, s.turn),
        error=pick(new.error, s.error),
        lost=pick(new.lost, s.lost),
    )


def sense(
    params: SensingParams,
    s: Sensing,
    ranges: Array,
    speed: Array,
    rng: np.random.Generator,
) -> tuple[Sensing, policy.Observation]:
    """The sensing a control step later, and the policy's observations, from
    the cars' true ranges (N, beams) m and wheel speeds (N,) m/s then."""
    history = np.concatenate([ranges[:, None], s.history[:, :-1]], axis=1)
    now = s.time + policy.CONTROL_PERIOD
    # The end of the last complete turn, and when in it the head passed each beam.
    turn = np.floor((now - s.first_turn) / s.period)
    end = s.first_turn + turn * s.period
    passed = end[:, None] - s.period[:, None] * (1.0 - s.passes)
    lag = np.rint((now[:, None] - passed) / policy.CONTROL_PERIOD).astype(np.intp)
    lag = np.clip(lag, 0, params.history - 1)
    seen = history[np.arange(len(now))[:, None], lag, np.arange(ranges.shape[1])]
    # A turn that has just ended brings new noise and lost beams.
    new = (turn != s.turn)[:, None]
    error = np.where(new, rng.normal(0.0, params.range_noise, seen.shape), s.error)
    lost = np.where(new, rng.random(seen.shape) < params.dropout, s.lost)
    scan = np.clip(
        np.where(lost, policy.MAX_RANGE, seen + error), 0.0, policy.MAX_RANGE
    )
    s = replace(s, history=history, time=now, turn=turn, error=error, lost=lost)
    return s, policy.observation(scan, speed * s.speed_scale, s.last_action)


def delay(s: Sensing, action: Array) -> tuple[Sensing, Array]:
    """The sensing after the policy acted, and the actions (N, ACTION_DIM)
    that reach the cars: on a late car, the policy's last action."""
    clipped = np.clip(action, -1.0, 1.0)
    applied = np.where(s.late[:, None], s.last_action, clipped)
    return replace(s, last_action=clipped), applied


def init_net(key: jax.Array, params: PPOParams) -> ActorCritic:
    """train_ppo.init_actor_critic, for policy.OBS_DIM inputs."""
    actor_key, critic_key = jax.random.split(key)
    return ActorCritic(
        actor=init_mlp(actor_key, (policy.OBS_DIM, *params.hidden, ACTION_DIM), 0.01),
        critic=init_mlp(critic_key, (policy.OBS_DIM, *params.hidden, 1), 1.0),
        log_std=jnp.full(ACTION_DIM, params.init_log_std),
    )


# --- Effects: environments, rollouts, logging, files -----------------------


@dataclass(frozen=True)
class RolloutState:
    """The state that a rollout carries to the next rollout."""

    env: BatchState
    sensing: Sensing
    episode_return: Array  # (N,) the return so far of each car
    episode_steps: npt.NDArray[np.int64]  # (N,) the steps so far of each car
    stats: ObsStats


def collect_rollout(
    env: NeoRacerBatch,
    state: RolloutState,
    net: ActorCritic,
    key: jax.Array,
    params: PPOParams,
    sensing_params: SensingParams,
    rng: np.random.Generator,
) -> tuple[Batch, RolloutState, list[EpisodeResult]]:
    """train_ppo.collect_rollout, with each observation and action passed
    through the car's sensing."""
    t_len, n = params.rollout_steps, env.nbatch
    obs_buf = np.zeros((t_len, n, policy.OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((t_len, n, ACTION_DIM), dtype=np.float32)
    logp_buf, val_buf, rew_buf, done_buf = (
        np.zeros((t_len, n), dtype=np.float32) for _ in range(4)
    )
    env_state, sensing, stats = state.env, state.sensing, state.stats
    ep_ret, ep_steps = state.episode_return.copy(), state.episode_steps.copy()
    finished: list[EpisodeResult] = []
    step_keys = jax.random.split(key, t_len)

    def observe(s: Sensing, obs: EnvObservations) -> tuple[Sensing, Array]:
        return sense(sensing_params, s, true_ranges(obs), wheel_speed(obs), rng)

    for t in range(t_len):
        sensing, raw_obs = observe(sensing, env_state.obs)
        stats = update_obs_stats(stats, raw_obs)
        obs = normalize_obs(stats, raw_obs, params.obs_clip)
        action, logp, val = sample_action(net, jnp.asarray(obs), step_keys[t])
        action_np = np.asarray(action, dtype=np.float64)
        obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action_np, logp, val
        sensing, applied = delay(sensing, action_np)
        env_state, out = env.step(env_state, applied)
        ep_ret += out.reward
        ep_steps += 1
        rew_buf[t] = out.reward
        done_buf[t] = out.done
        if out.truncated.any():
            # The value of the last observation, as the policy would see it.
            _, final = observe(sensing, out.obs)
            final_obs = normalize_obs(stats, final, params.obs_clip)
            final_value = np.asarray(value(net, jnp.asarray(final_obs)))
            rew_buf[t] += np.where(out.truncated, params.gamma * final_value, 0.0)
        for i in np.flatnonzero(out.done):
            code = int(out.reason[i])
            finished.append(
                EpisodeResult(
                    float(ep_ret[i]),
                    float(out.lap_frac[i]),
                    REASONS[code - 1] if code else "truncated",
                    int(ep_steps[i]),
                )
            )
        ep_ret[out.done], ep_steps[out.done] = 0.0, 0
        if out.done.any():
            ranges = true_ranges(env_state.obs)
            sensing = restart(sensing_params, sensing, out.done, ranges, rng)

    _, raw_obs = observe(sensing, env_state.obs)
    last_value = value(net, jnp.asarray(normalize_obs(stats, raw_obs, params.obs_clip)))
    advantage, target = gae(
        jnp.asarray(rew_buf),
        jnp.asarray(val_buf),
        jnp.asarray(done_buf),
        last_value,
        params.gamma,
        params.gae_lambda,
    )
    batch = Batch(
        obs=jnp.asarray(obs_buf).reshape(t_len * n, policy.OBS_DIM),
        action=jnp.asarray(act_buf).reshape(t_len * n, ACTION_DIM),
        log_prob=jnp.asarray(logp_buf).reshape(-1),
        advantage=advantage.reshape(-1),
        target=target.reshape(-1),
    )
    next_state = RolloutState(env_state, sensing, ep_ret, ep_steps, stats)
    return batch, next_state, finished


def train(
    params: PPOParams,
    sensing_params: SensingParams,
    total_steps: int,
    seed: int,
    num_envs: int,
    nthread: int,
    config: NeoRacerEnvConfig,
) -> tuple[ActorCritic, ObsStats]:
    """Train for at least total_steps env steps. Print one log line per update.
    The spawns, the physics and the sensing come from default_rng(seed)."""
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    net = init_net(init_key, params)
    opt_state = init_opt_state(net, params)
    env = NeoRacerBatch(Cpu(nthread=min(nthread, num_envs)), num_envs, config)
    rng = np.random.default_rng(seed)
    env_state = env.reset(rng)
    state = RolloutState(
        env=env_state,
        sensing=start(sensing_params, true_ranges(env_state.obs), rng),
        episode_return=np.zeros(num_envs),
        episode_steps=np.zeros(num_envs, dtype=np.int64),
        stats=ObsStats(np.zeros(policy.OBS_DIM), np.ones(policy.OBS_DIM), 0.0),
    )
    steps_per_update = params.rollout_steps * num_envs
    for update in range(1, -(-total_steps // steps_per_update) + 1):
        start_time = time.perf_counter()
        key, rollout_key, update_key = jax.random.split(key, 3)
        batch, state, finished = collect_rollout(
            env, state, net, rollout_key, params, sensing_params, rng
        )
        state = replace(state, env=env.respawn(state.env, rng))
        net, opt_state, loss = ppo_update(net, opt_state, batch, update_key, params)
        log = UpdateLog(
            update=update,
            env_steps=update * steps_per_update,
            episodes=tuple(finished),
            loss=loss,
            action_std=tuple(float(s) for s in np.exp(np.asarray(net.log_std))),
        )
        rate = steps_per_update / (time.perf_counter() - start_time)
        print(format_log(log, rate), flush=True)
    env.close()
    return net, state.stats


def save_policy(
    path: str,
    net: ActorCritic,
    stats: ObsStats,
    params: PPOParams,
    config: NeoRacerEnvConfig,
) -> None:
    """The arrays of train_ppo.save_policy, which load_policy reads, and
    what policy.load needs beside them: obs_clip and throttle_scale."""
    arrays: dict[str, np.ndarray] = {
        "log_std": np.asarray(net.log_std),
        "obs_mean": stats.mean,
        "obs_var": stats.var,
        "obs_count": np.array(stats.count),
        "obs_clip": np.array(params.obs_clip),
        "throttle_scale": np.array(config.throttle_scale),
    }
    for name, layers in (("actor", net.actor), ("critic", net.critic)):
        for i, layer in enumerate(layers):
            arrays[f"{name}.{i}.weight"] = np.asarray(layer.weight)
            arrays[f"{name}.{i}.bias"] = np.asarray(layer.bias)
    np.savez(path, allow_pickle=False, **arrays)


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
    rng = np.random.default_rng(seed)
    state = env.reset(rng)
    sensing = start(sensing_params, true_ranges(state.obs), rng)
    returns, steps = np.zeros(episodes), 0
    results: dict[int, EpisodeResult] = {}
    while len(results) < episodes:
        ranges, speed = true_ranges(state.obs), wheel_speed(state.obs)
        sensing, obs = sense(sensing_params, sensing, ranges, speed, rng)
        sensing, applied = delay(sensing, policy.act(car, obs))
        state, out = env.step(state, applied)
        returns += out.reward
        steps += 1
        for i in np.flatnonzero(out.done):
            if int(i) not in results:
                code = int(out.reason[i])
                results[int(i)] = EpisodeResult(
                    float(returns[i]),
                    float(out.lap_frac[i]),
                    REASONS[code - 1] if code else "truncated",
                    steps,
                )
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
    args = parser.parse_args()
    params, sensing_params = PPOParams(), SensingParams()
    config = NeoRacerEnvConfig(car_path=CAR, tracks=LoopCorridor(), physics=PHYSICS)

    if args.eval is not None:
        car = policy.load(args.eval)
        results = evaluate(
            car,
            sensing_params,
            args.episodes,
            args.seed + 10_000,
            args.nthread,
            # The throttle that action 1 commands, as on the car.
            replace(config, throttle_scale=car.throttle_scale),
        )
        laps = [
            r.steps * policy.CONTROL_PERIOD
            for r in results
            if r.outcome == "lap_complete"
        ]
        print(
            f"[eval] {args.eval}: {len(results)} episodes, "
            f"outcomes {dict(Counter(r.outcome for r in results))}, "
            f"mean lap_frac {np.mean([r.lap_frac for r in results]):.3f}, "
            f"mean lap time {np.mean(laps) if laps else float('nan'):.1f} s"
        )
        return

    print(f"[train_v2] {params}")
    print(f"[train_v2] {sensing_params}")
    print(
        f"[train_v2] total_steps={args.total_steps} seed={args.seed} "
        f"num_envs={args.num_envs} physics={PHYSICS}"
    )
    net, stats = train(
        params,
        sensing_params,
        args.total_steps,
        args.seed,
        args.num_envs,
        args.nthread,
        config,
    )
    if args.out is not None:
        save_policy(args.out, net, stats, params, config)
        print(f"[train_v2] saved the policy to {args.out}")


if __name__ == "__main__":
    main()
