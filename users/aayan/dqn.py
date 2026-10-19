"""
Deep Q-learning on the NeoRacer, to compare against the tables.

This is the same problem tabular.py solves, with one piece swapped. It keeps
the 15 action grid from tabular.py and the environment's own observation, and
replaces the 100 row table with a network that reads all 20 numbers. So the
comparison isolates the thing that changed: whether a learned function
generalises better than bins drawn by hand.

    python3 users/aayan/dqn.py --steps 200000
    python3 users/aayan/dqn.py --hidden 256 256 --learning-rate 1e-4
    python3 users/aayan/dqn.py --eval runs/dqn_200000steps_s0.npz

What the table throws away, this does not. The discretizer keeps three LiDAR
features out of eight beams and bins each one, so two corners that differ by
less than a bin are the same state. The network sees every beam, the IMU and
the speed, at full resolution.

The pieces are the ones from the 2015 DQN paper: a replay buffer so updates
are not all from one trajectory, a target network held still between syncs so
the regression has a fixed aim, and epsilon decayed from 1 to a floor. Double
Q-learning is on by default: the online network picks the next action and the
target network scores it, which takes out most of the overestimation that a
plain max introduces.

JAX and optax, not PyTorch, because the package needs Python 3.15 and PyTorch
has no wheels for it. The network is built the way examples/train_ppo.py
builds its own, with orthogonal weights and tanh hidden layers.
"""

import argparse
import functools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from neoracer_rl.envs import NeoRacerEnv
from neoracer_rl.envs.neoracer_env import OBS_DIM

import store
import tabular

N_ACTIONS = tabular.N_ACTIONS


@dataclass(frozen=True)
class DQNParams:
    """Everything that decides how the agent learns. All of it is on the CLI."""

    hidden: tuple[int, ...] = (128, 128)  # widths of the hidden layers
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100_000  # transitions kept for replay
    warmup_steps: int = 2_000  # fill the buffer before the first update
    train_every: int = 1  # env steps between updates
    target_sync_every: int = 1_000  # env steps between target network copies
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3  # fraction of training spent decaying epsilon
    double: bool = True  # double Q-learning, see the module docstring
    max_grad_norm: float = 10.0
    obs_clip: float = 10.0  # normalized observation clip, in standard deviations

    def __post_init__(self) -> None:
        if min(self.hidden, default=1) < 1:
            raise ValueError(f"every hidden layer needs a width, got {self.hidden}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], not {self.gamma}")
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError("epsilon must fall from eps_start to eps_end inside [0, 1]")
        if min(self.learning_rate, self.batch_size, self.buffer_size) <= 0:
            raise ValueError("the learning rate, batch and buffer must all be positive")


# --- the network ----------------------------------------------------------
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Dense:
    weight: jax.Array  # shape (inputs, outputs)
    bias: jax.Array  # shape (outputs,)


def init_mlp(key: jax.Array, sizes: Sequence[int], out_scale: float) -> tuple[Dense, ...]:
    """An MLP with orthogonal weights and zero biases, as train_ppo.py builds."""
    keys = jax.random.split(key, len(sizes) - 1)
    gains = [np.sqrt(2.0)] * (len(sizes) - 2) + [out_scale]
    return tuple(
        Dense(
            weight=jax.nn.initializers.orthogonal(gain)(k, (n_in, n_out)),
            bias=jnp.zeros(n_out),
        )
        for k, n_in, n_out, gain in zip(keys, sizes[:-1], sizes[1:], gains, strict=True)
    )


def mlp(layers: tuple[Dense, ...], x: jax.Array) -> jax.Array:
    """tanh hidden layers, linear output."""
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer.weight + layer.bias)
    return x @ layers[-1].weight + layers[-1].bias


@functools.partial(jax.jit, static_argnames=("params",))
def init_q(key: jax.Array, params: DQNParams) -> tuple[Dense, ...]:
    """One network from an observation to a value for each of the 15 actions."""
    return init_mlp(key, (OBS_DIM, *params.hidden, N_ACTIONS), 1.0)


# --- observation normalization --------------------------------------------
@dataclass
class ObsStats:
    """Running mean and variance of the raw observations, as train_ppo keeps."""

    mean: np.ndarray
    var: np.ndarray
    count: float = 1e-4

    @classmethod
    def new(cls) -> "ObsStats":
        return cls(mean=np.zeros(OBS_DIM), var=np.ones(OBS_DIM))

    def update(self, batch: np.ndarray) -> None:
        batch = np.atleast_2d(batch)
        n = batch.shape[0]
        mean, var = batch.mean(axis=0), batch.var(axis=0)
        delta = mean - self.mean
        total = self.count + n
        new_mean = self.mean + delta * n / total
        m_a, m_b = self.var * self.count, var * n
        new_var = (m_a + m_b + delta**2 * self.count * n / total) / total
        self.mean, self.var, self.count = new_mean, new_var, total


def normalize(stats: ObsStats, obs: np.ndarray, clip: float) -> np.ndarray:
    return np.clip((obs - stats.mean) / np.sqrt(stats.var + 1e-8), -clip, clip).astype(
        np.float32
    )


# --- replay ----------------------------------------------------------------
class Replay:
    """A ring of past transitions. Nothing clever: uniform sampling."""

    def __init__(self, capacity: int) -> None:
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.next = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.next
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.done[i] = next_obs, float(done)
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, batch: int):
        idx = rng.integers(0, self.size, size=batch)
        return (
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.done[idx],
        )


# --- learning --------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("params",))
def q_loss(online, target, batch, params: DQNParams):
    """Mean squared temporal difference error over a batch."""
    obs, action, reward, next_obs, done = batch
    if params.double:
        # The online network picks the next action, the target scores it.
        best = jnp.argmax(mlp(online, next_obs), axis=-1)
        next_value = jnp.take_along_axis(
            mlp(target, next_obs), best[:, None], axis=-1
        )[:, 0]
    else:
        next_value = jnp.max(mlp(target, next_obs), axis=-1)
    goal = reward + params.gamma * (1.0 - done) * next_value
    taken = jnp.take_along_axis(mlp(online, obs), action[:, None].astype(jnp.int32), axis=-1)[:, 0]
    return jnp.mean((taken - jax.lax.stop_gradient(goal)) ** 2)


@functools.partial(jax.jit, static_argnames=("params", "optimizer"))
def update(online, target, opt_state, batch, params: DQNParams, optimizer):
    loss, grads = jax.value_and_grad(q_loss)(online, target, batch, params)
    updates, opt_state = optimizer.update(grads, opt_state, online)
    return optax.apply_updates(online, updates), opt_state, loss


def epsilon(step: int, total: int, params: DQNParams) -> float:
    """Linear decay over the first eps_fraction of training, then a floor."""
    decay_steps = max(1, int(params.eps_fraction * total))
    frac = min(1.0, step / decay_steps)
    return params.eps_start + frac * (params.eps_end - params.eps_start)


def greedy_action(online, stats: ObsStats, obs: np.ndarray, params: DQNParams) -> int:
    values = mlp(online, jnp.asarray(normalize(stats, obs, params.obs_clip)[None]))
    return int(jnp.argmax(values[0]))


def train(total_steps: int, params: DQNParams, seed: int, log_every: int = 10_000):
    """Run DQN for total_steps environment steps. Returns the network, the
    observation statistics and one row per finished episode."""
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    online = init_q(init_key, params)
    target = online
    optimizer = optax.chain(
        optax.clip_by_global_norm(params.max_grad_norm),
        optax.adam(params.learning_rate),
    )
    opt_state = optimizer.init(online)

    rng = np.random.default_rng(seed)
    replay = Replay(params.buffer_size)
    stats = ObsStats.new()
    env = NeoRacerEnv()

    obs, _ = env.reset(seed=seed)
    stats.update(obs)
    history, episode_return, episode_steps = [], 0.0, 0
    started = time.time()

    for step in range(1, total_steps + 1):
        eps = epsilon(step, total_steps, params)
        if step < params.warmup_steps or rng.random() < eps:
            action = int(rng.integers(N_ACTIONS))
        else:
            action = greedy_action(online, stats, obs, params)

        next_obs, reward, terminated, truncated, info = env.step(
            np.array(tabular.ACTIONS[action], dtype=np.float32)
        )
        stats.update(next_obs)
        # The target bootstraps past a time limit but not past a real ending.
        replay.add(
            normalize(stats, obs, params.obs_clip),
            action,
            reward,
            normalize(stats, next_obs, params.obs_clip),
            terminated,
        )
        episode_return += float(reward)
        episode_steps += 1
        obs = next_obs

        if terminated or truncated:
            history.append(
                {
                    "step": step,
                    "return": episode_return,
                    "lap": float(info["lap_frac"]),
                    "steps": episode_steps,
                    "reason": info["termination_reason"] or "truncated",
                    "eps": eps,
                }
            )
            obs, _ = env.reset()
            episode_return, episode_steps = 0.0, 0

        if step >= params.warmup_steps and step % params.train_every == 0:
            batch = replay.sample(rng, params.batch_size)
            online, opt_state, _ = update(
                online, target, opt_state, batch, params, optimizer
            )
        if step % params.target_sync_every == 0:
            target = online

        if step % log_every == 0 or step == total_steps:
            recent = history[-20:]
            laps = np.mean([h["lap"] for h in recent]) if recent else 0.0
            ret = np.mean([h["return"] for h in recent]) if recent else 0.0
            print(
                f"    step {step:>7}/{total_steps}  episodes {len(history):>4}  "
                f"return {ret:+7.2f}  lap {laps:.3f}  eps {eps:.2f}  "
                f"{step / max(time.time() - started, 1e-9):.0f} steps/s"
            )

    env.close()
    return online, stats, history


def evaluate(online, stats: ObsStats, params: DQNParams, episodes: int = 5, seed_base: int = 1000):
    """Greedy rollouts, reported the way tabular.py reports them."""
    env = NeoRacerEnv()
    returns, laps, outcomes, lengths = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total, done, steps = 0.0, False, 0
        info: dict = {}
        while not done:
            action = greedy_action(online, stats, obs, params)
            obs, reward, terminated, truncated, info = env.step(
                np.array(tabular.ACTIONS[action], dtype=np.float32)
            )
            total += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(total)
        laps.append(float(info["lap_frac"]))
        lengths.append(steps)
        outcomes.append(info["termination_reason"] or "truncated")
        print(
            f"  episode {ep:2d}: steps={steps:5d}  return={total:+7.2f}  "
            f"outcome={str(outcomes[-1]):15s} lap_frac={laps[-1]:.3f}"
        )
    env.close()
    return float(np.mean(returns)), float(np.mean(laps)), outcomes, lengths


def _flat(net: tuple[Dense, ...]) -> dict:
    """The network as plain arrays, so np.savez can hold it."""
    out = {}
    for i, layer in enumerate(net):
        out[f"w{i}"] = np.asarray(layer.weight)
        out[f"b{i}"] = np.asarray(layer.bias)
    return out


def _unflat(data, depth: int) -> tuple[Dense, ...]:
    return tuple(
        Dense(weight=jnp.asarray(data[f"w{i}"]), bias=jnp.asarray(data[f"b{i}"]))
        for i in range(depth)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    defaults = DQNParams()
    parser.add_argument("--steps", type=int, default=200_000, help="environment steps")
    parser.add_argument("--hidden", type=int, nargs="+", default=list(defaults.hidden),
                        help="widths of the hidden layers, for example --hidden 256 256")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--buffer-size", type=int, default=defaults.buffer_size)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--train-every", type=int, default=defaults.train_every)
    parser.add_argument("--target-sync-every", type=int, default=defaults.target_sync_every)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--eps-start", type=float, default=defaults.eps_start)
    parser.add_argument("--eps-end", type=float, default=defaults.eps_end)
    parser.add_argument("--eps-fraction", type=float, default=defaults.eps_fraction)
    parser.add_argument("--no-double", dest="double", action="store_false",
                        help="plain DQN, without double Q-learning")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--name", default=None, help="name the saved network")
    parser.add_argument("--eval", metavar="RUN.npz", default=None,
                        help="score a saved network, do not train")
    parser.set_defaults(double=defaults.double)
    args = parser.parse_args()

    params = replace(
        defaults,
        hidden=tuple(args.hidden),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup_steps=args.warmup_steps,
        train_every=args.train_every,
        target_sync_every=args.target_sync_every,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_fraction=args.eps_fraction,
        double=args.double,
    )

    if args.eval:
        with np.load(args.eval, allow_pickle=False) as data:
            saved_hidden = tuple(int(x) for x in data["hidden"])
            settings = json.loads(str(data["settings"]))
            params = replace(params, hidden=saved_hidden)
            net = _unflat(data, len(saved_hidden) + 1)
            stats = ObsStats(mean=data["obs_mean"], var=data["obs_var"])
        print("trained with: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
        mean_ret, mean_lap, outcomes, lengths = evaluate(
            net, stats, params, args.eval_episodes
        )
        print(f"\nmean return {mean_ret:+.2f}   mean lap {mean_lap:.3f}   "
              f"laps {sum(o == 'lap_complete' for o in outcomes)}/{len(outcomes)}")
        store.log_result("eval", {"network": str(args.eval), "episodes": args.eval_episodes,
                                  "return": round(mean_ret, 3), "lap": round(mean_lap, 4)})
        return

    print(f"DQN: hidden {params.hidden}, lr {params.learning_rate}, batch "
          f"{params.batch_size}, {'double' if params.double else 'plain'}, "
          f"{args.steps} steps")
    started = time.time()
    net, stats, history = train(args.steps, params, args.seed, args.log_every)
    minutes = (time.time() - started) / 60

    print("\n  greedy evaluation:")
    mean_ret, mean_lap, outcomes, lengths = evaluate(net, stats, params, args.eval_episodes)
    laps_done = sum(o == "lap_complete" for o in outcomes)
    print(f"  => mean return {mean_ret:+.2f}   mean lap {mean_lap:.3f}   "
          f"laps completed {laps_done}/{len(outcomes)}")

    settings = {
        "method": "dqn",
        "steps": args.steps,
        "episodes": len(history),
        "seed": args.seed,
        "hidden": list(params.hidden),
        "learning_rate": params.learning_rate,
        "batch_size": params.batch_size,
        "buffer_size": params.buffer_size,
        "target_sync_every": params.target_sync_every,
        "gamma": params.gamma,
        "double": params.double,
        "minutes": round(minutes, 2),
    }
    path = store.unique(args.name or f"dqn_{args.steps}steps_s{args.seed}")
    # The same stamp store.save_table puts on a table: when it ran and against
    # which package versions, because those move week to week.
    settings = {**settings, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **store.versions()}
    np.savez(
        path,
        hidden=np.array(params.hidden),
        obs_mean=stats.mean,
        obs_var=stats.var,
        history=np.array([[h["return"], h["lap"]] for h in history]),
        settings=json.dumps(settings),
        **_flat(net),
    )
    print(f"  saved -> {path}")
    store.log_result("train", {**settings, "return": round(mean_ret, 3),
                               "lap": round(mean_lap, 4), "laps": laps_done,
                               "n": len(outcomes), "table": path.name})


if __name__ == "__main__":
    main()
