"""
Train a PPO policy on NeoRacerBatch, or evaluate a saved policy on
NeoRacerEnv.

PPO (proximal policy optimization) is a policy-gradient method.
It collects a batch of steps with the current policy, then makes a few
gradient steps on a loss that limits how far the new policy moves from the old one.
This script is a small baseline, not a tuned agent.

The policy is a Gaussian distribution over (speed, steer). An MLP gives the mean,
and a learned vector gives the log standard deviation. env.step() clips each
sampled action to [-1, 1]; the log probability uses the unclipped sample.
A second MLP (the critic) estimates the value of an observation.

Training steps every environment at once with NeoRacerBatch, on the backend
that --backend names: cpu (mujoco.rollout on CPU threads) or warp (MuJoCo
Warp, for a CUDA GPU; it needs the warp extra). Evaluation runs the
Gymnasium env NeoRacerEnv, one episode at a time.

The script normalizes each observation with a running mean and variance,
because the LiDAR value -1 (no hit) and the large IMU ranges otherwise
dominate the input of the MLP.

The pure core is: the MLP, gae(), clipped_surrogate(), ppo_loss(), ppo_update()
and the observation statistics. main() and the functions below "Effects" step
the environments, print the log and read or write the .npz file.

Usage:
    python3 -m examples.train_ppo --total-steps 1000000 --seed 0 --out ppo.npz
    python3 -m examples.train_ppo --backend warp --num-envs 1024 --out ppo.npz
    python3 -m examples.train_ppo --eval ppo.npz --episodes 10
    python3 -m examples.train_ppo --eval-baseline random --episodes 10
"""

import argparse
import functools
import os
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, assert_never, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from neoracer_mujoco.physics import PhysicsRanges, Range

from neoracer_rl.envs import (
    GeneratedTracks,
    LoopCorridor,
    NeoRacerEnv,
    NeoRacerEnvConfig,
)
from neoracer_rl.envs.neoracer_batch import (
    REASONS,
    Backend,
    BatchState,
    Cpu,
    NeoRacerBatch,
    StepResult,
    Warp,
)
from neoracer_rl.envs.neoracer_env import (
    OBS_DIM,
    EnvInfo,
    TerminationReason,
    Tracks,
)

ACTION_DIM = 2
"""(speed, steer), as neoracer_env.py gives them."""

RANDOM_PHYSICS = PhysicsRanges(
    wheel_friction=Range(0.3, 1.2),
    chassis_mass=Range(0.7, 1.5),
    motor_torque=Range(0.5, 1.5),
    steer_stiffness=Range(0.5, 2.0),
)
"""The physics ranges of --random-physics: an example to start from, wide
enough to see an effect, not measured on the real car."""

type ObsBatch = npt.NDArray[np.float64]
"""Shape (batch, obs_dim)."""


@dataclass(frozen=True)
class PPOParams:
    """Hyperparameters of PPO. One update uses num_envs * rollout_steps env steps."""

    rollout_steps: int = 256
    """Env steps per environment between two updates."""
    gamma: float = 0.99
    """The discount per env step (30 Hz), a horizon of about 1/(1-gamma) steps."""
    gae_lambda: float = 0.95
    """GAE's weight between 1-step (0) and Monte Carlo (1)."""
    clip_eps: float = 0.2
    """PPO's clip range of the probability ratio."""
    learning_rate: float = 3e-4
    """Adam's step size."""
    epochs: int = 10
    """Passes over each rollout batch."""
    minibatches: int = 4
    """Minibatches per epoch."""
    value_coef: float = 0.5
    """The weight of the value loss in the total loss."""
    entropy_coef: float = 0.0
    """The weight of the entropy bonus in the total loss."""
    max_grad_norm: float = 0.5
    """The gradient clip, on the global L2 norm."""
    hidden: tuple[int, ...] = (64, 64)
    """The hidden layer widths of both MLPs."""
    init_log_std: float = 0.0
    """The initial log standard deviation of each action. With -0.5 (std
    0.6), the policy did not find the left turn at the first corner in 300k
    steps. With 0.0 (std 1), it completed laps after about 100k steps."""
    obs_clip: float = 10.0
    """The normalized observation's clip, in standard deviations."""

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.rollout_steps < 1 or self.epochs < 1 or self.minibatches < 1:
            raise ValueError("rollout_steps, epochs and minibatches must be >= 1")
        # A batch has rollout_steps * num_envs samples, so this check gives
        # each minibatch at least one sample for any num_envs >= 1.
        if self.minibatches > self.rollout_steps:
            raise ValueError("minibatches must be <= rollout_steps")
        if (
            min(self.clip_eps, self.learning_rate, self.max_grad_norm, self.obs_clip)
            <= 0
        ):
            raise ValueError(
                "clip_eps, learning_rate, max_grad_norm and obs_clip must be > 0"
            )
        if self.value_coef < 0 or self.entropy_coef < 0:
            raise ValueError("value_coef and entropy_coef must be >= 0")


# --- Network -------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Dense:
    weight: jax.Array
    """Shape (inputs, outputs)."""
    bias: jax.Array
    """Shape (outputs,)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ActorCritic:
    actor: tuple[Dense, ...]
    """obs -> action mean."""
    critic: tuple[Dense, ...]
    """obs -> value."""
    log_std: jax.Array
    """The log standard deviation of each action, shape (ACTION_DIM,)."""


def init_mlp(
    key: jax.Array, sizes: Sequence[int], out_scale: float
) -> tuple[Dense, ...]:
    """An MLP with orthogonal weights and zero biases. The last layer uses
    out_scale as the gain, the hidden layers use sqrt(2)."""
    keys = jax.random.split(key, len(sizes) - 1)
    gains = [np.sqrt(2.0)] * (len(sizes) - 2) + [out_scale]
    return tuple(
        Dense(
            weight=jax.nn.initializers.orthogonal(gain)(k, (n_in, n_out)),
            bias=jnp.zeros(n_out),
        )
        for k, n_in, n_out, gain in zip(keys, sizes[:-1], sizes[1:], gains, strict=True)
    )


@functools.partial(jax.jit, static_argnames=("params", "obs_dim"))
def init_actor_critic(
    key: jax.Array, params: PPOParams, obs_dim: int = OBS_DIM
) -> ActorCritic:
    actor_key, critic_key = jax.random.split(key)
    return ActorCritic(
        # A small gain makes the initial action mean close to 0.
        actor=init_mlp(actor_key, (obs_dim, *params.hidden, ACTION_DIM), 0.01),
        critic=init_mlp(critic_key, (obs_dim, *params.hidden, 1), 1.0),
        log_std=jnp.full(ACTION_DIM, params.init_log_std),
    )


def mlp(layers: tuple[Dense, ...], x: jax.Array) -> jax.Array:
    """tanh hidden layers, linear output."""
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer.weight + layer.bias)
    return x @ layers[-1].weight + layers[-1].bias


def value(net: ActorCritic, obs: jax.Array) -> jax.Array:
    """The critic value of each normalized observation. Shape (batch,)."""
    return mlp(net.critic, obs)[..., 0]


def gaussian_log_prob(mean: jax.Array, log_std: jax.Array, x: jax.Array) -> jax.Array:
    """log density of x under a diagonal Gaussian, summed over the last axis."""
    z = (x - mean) / jnp.exp(log_std)
    return jnp.sum(-0.5 * z**2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)


@jax.jit
def sample_action(
    net: ActorCritic, obs: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(unclipped action, its log probability, value) for a batch of normalized
    observations."""
    mean = mlp(net.actor, obs)
    action = mean + jnp.exp(net.log_std) * jax.random.normal(key, mean.shape)
    return action, gaussian_log_prob(mean, net.log_std, action), value(net, obs)


@jax.jit
def greedy_action(net: ActorCritic, obs: jax.Array) -> jax.Array:
    """The mean action of the policy (deterministic)."""
    return mlp(net.actor, obs)


# --- Observation normalization --------------------------------------------


@dataclass(frozen=True)
class ObsStats:
    """Running mean and variance of the raw observations."""

    mean: npt.NDArray[np.float64]
    """Shape (obs_dim,)."""
    var: npt.NDArray[np.float64]
    """The population variance, shape (obs_dim,)."""
    count: float
    """The observations seen; 0 before any."""


def empty_obs_stats(obs_dim: int = OBS_DIM) -> ObsStats:
    return ObsStats(mean=np.zeros(obs_dim), var=np.ones(obs_dim), count=0.0)


def update_obs_stats(stats: ObsStats, batch: ObsBatch) -> ObsStats:
    """Combine the statistics with one batch (parallel variance formula,
    Chan et al. 1979)."""
    n = float(batch.shape[0])
    total = stats.count + n
    delta = batch.mean(axis=0) - stats.mean
    m2 = (
        stats.var * stats.count
        + batch.var(axis=0) * n
        + delta**2 * stats.count * n / total
    )
    return ObsStats(mean=stats.mean + delta * n / total, var=m2 / total, count=total)


def normalize_obs(stats: ObsStats, obs: ObsBatch, clip: float) -> ObsBatch:
    return np.clip((obs - stats.mean) / np.sqrt(stats.var + 1e-8), -clip, clip)


# --- GAE and the PPO loss ---------------------------------------------------


def gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over a rollout of shape (T, N).

    dones[t] is 1 if the episode ended after step t (terminated or truncated).
    Then step t does not use the value of the next observation.
    last_value (shape (N,)) is the value of the observation after step T-1.
    Returns (advantages, returns), both of shape (T, N); returns = advantages + values.
    """
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
    not_done = 1.0 - dones
    deltas = rewards + gamma * next_values * not_done - values

    def backward(carry: jax.Array, x: tuple[jax.Array, jax.Array]):
        delta, nd = x
        adv = delta + gamma * lam * nd * carry
        return adv, adv

    _, advantages = jax.lax.scan(
        backward, jnp.zeros_like(last_value), (deltas, not_done), reverse=True
    )
    return advantages, advantages + values


def clipped_surrogate(
    ratio: jax.Array, advantage: jax.Array, clip_eps: float
) -> jax.Array:
    """The PPO policy loss of each sample: -min(r A, clip(r, 1-eps, 1+eps) A)."""
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return -jnp.minimum(ratio * advantage, clipped * advantage)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Batch:
    """Flat training samples. All fields have the same first axis."""

    obs: jax.Array
    """The normalized observation, shape (B, OBS_DIM)."""
    action: jax.Array
    """The unclipped sampled action, shape (B, ACTION_DIM)."""
    log_prob: jax.Array
    """The log probability of action under the rollout policy."""
    advantage: jax.Array
    target: jax.Array
    """The value target, GAE's return."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LossInfo:
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    """The sum over the action dimensions, in nats."""
    approx_kl: jax.Array
    """An estimate of KL(old || new), in nats."""
    clip_frac: jax.Array
    """The share of samples with |ratio - 1| > clip_eps."""


def ppo_loss(
    net: ActorCritic, batch: Batch, params: PPOParams
) -> tuple[jax.Array, LossInfo]:
    """Total loss = policy loss + value_coef * value loss - entropy_coef * entropy.
    The function normalizes the advantages over the batch."""
    adv = (batch.advantage - batch.advantage.mean()) / (batch.advantage.std() + 1e-8)
    log_prob = gaussian_log_prob(mlp(net.actor, batch.obs), net.log_std, batch.action)
    log_ratio = log_prob - batch.log_prob
    ratio = jnp.exp(log_ratio)
    policy_loss = clipped_surrogate(ratio, adv, params.clip_eps).mean()
    value_loss = 0.5 * jnp.mean((value(net, batch.obs) - batch.target) ** 2)
    entropy = jnp.sum(net.log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
    total = policy_loss + params.value_coef * value_loss - params.entropy_coef * entropy
    return total, LossInfo(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=jnp.mean((ratio - 1.0) - log_ratio),
        clip_frac=jnp.mean(jnp.abs(ratio - 1.0) > params.clip_eps),
    )


def make_optimizer(params: PPOParams) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(params.max_grad_norm),
        optax.adam(params.learning_rate, eps=1e-5),
    )


def init_opt_state(net: ActorCritic, params: PPOParams) -> optax.OptState:
    # optax's type hint for params lists only arrays and containers, not
    # registered dataclasses. optax accepts any JAX pytree.
    return make_optimizer(params).init(cast(optax.Params, net))


@functools.partial(jax.jit, static_argnames=("params",))
def ppo_update(
    net: ActorCritic,
    opt_state: optax.OptState,
    batch: Batch,
    key: jax.Array,
    params: PPOParams,
) -> tuple[ActorCritic, optax.OptState, LossInfo]:
    """params.epochs passes of minibatch Adam steps over a shuffled batch.
    Returns the new network, the new optimizer state and the mean loss info."""
    optimizer = make_optimizer(params)
    size = batch.obs.shape[0]
    mb_size = size // params.minibatches

    def minibatch_step(carry, idx):
        net, opt_state = carry
        mb = jax.tree.map(lambda x: x[idx], batch)
        grads, info = jax.grad(ppo_loss, has_aux=True)(net, mb, params)
        updates, opt_state = optimizer.update(grads, opt_state, net)
        return (optax.apply_updates(net, updates), opt_state), info

    def epoch(carry, epoch_key):
        perm = jax.random.permutation(epoch_key, size)[: params.minibatches * mb_size]
        return jax.lax.scan(minibatch_step, carry, perm.reshape(params.minibatches, -1))

    (net, opt_state), infos = jax.lax.scan(
        epoch, (net, opt_state), jax.random.split(key, params.epochs)
    )
    return net, opt_state, jax.tree.map(jnp.mean, infos)


# --- Effects: environments, rollouts, logging, files -----------------------

type Outcome = TerminationReason | Literal["truncated"]


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    lap_frac: float
    """Laps at the last step, negative if the car drove backward."""
    outcome: Outcome
    steps: int
    """Env steps, at 30 Hz."""


def ended(
    out: StepResult, returns: npt.NDArray[np.float64], steps: npt.NDArray[np.int64]
) -> dict[int, EpisodeResult]:
    """The result of each row whose episode ended in the step that gave out,
    by row: its return and its steps so far, and how it ended."""
    return {
        int(i): EpisodeResult(
            float(returns[i]),
            float(out.lap_frac[i]),
            REASONS[out.reason[i] - 1] if out.reason[i] else "truncated",
            int(steps[i]),
        )
        for i in np.flatnonzero(out.done)
    }


type EnvObsBatch = npt.NDArray[np.float32]
"""The env's observations, shape (N, OBS_DIM)."""
type ActionBatch = npt.NDArray[np.float64]
"""Shape (N, ACTION_DIM)."""
type Rows = npt.NDArray[np.bool_]
"""Shape (N,)."""


class Interface[S](Protocol):
    """What stands between the policy and the envs: what the policy sees of
    the envs' observations, and the actions that reach the envs. S is what
    it carries from one step to the next; rng draws its randomness."""

    @property
    def obs_dim(self) -> int:
        """The size of an observation that the policy sees."""
        ...

    def start(self, obs: EnvObsBatch, rng: np.random.Generator) -> S:
        """For envs at their spawns, where they observe obs."""
        ...

    def observe(
        self, s: S, obs: EnvObsBatch, rng: np.random.Generator
    ) -> tuple[S, ObsBatch]: ...

    def act(self, s: S, action: ActionBatch) -> tuple[S, ActionBatch]: ...

    def restart(
        self, s: S, rows: Rows, obs: EnvObsBatch, rng: np.random.Generator
    ) -> S:
        """s with the envs in rows started over at their spawns, where they
        observe obs."""
        ...


@dataclass(frozen=True)
class Exact:
    """The policy sees the envs' observations, and its actions reach the envs."""

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    def start(self, obs: EnvObsBatch, rng: np.random.Generator) -> None:
        return None

    def observe(
        self, s: None, obs: EnvObsBatch, rng: np.random.Generator
    ) -> tuple[None, ObsBatch]:
        return s, obs.astype(np.float64)

    def act(self, s: None, action: ActionBatch) -> tuple[None, ActionBatch]:
        return s, action

    def restart(
        self, s: None, rows: Rows, obs: EnvObsBatch, rng: np.random.Generator
    ) -> None:
        return s


@dataclass(frozen=True)
class RolloutState[S]:
    """The state that a rollout carries to the next rollout."""

    env: BatchState
    """The episode of each env."""
    episode_return: npt.NDArray[np.float64]
    """The return so far of each env, shape (N,)."""
    episode_steps: npt.NDArray[np.int64]
    """The steps so far of each env, shape (N,)."""
    stats: ObsStats
    carry: S
    """What the interface carries to the next step."""


def _outcome(info: EnvInfo) -> Outcome:
    """The outcome of an episode, from the info of its last step. The env sets a
    termination reason exactly when terminated is True."""
    reason = info["termination_reason"]
    return "truncated" if reason is None else reason


def collect_rollout[S](
    env: NeoRacerBatch,
    state: RolloutState[S],
    net: ActorCritic,
    key: jax.Array,
    params: PPOParams,
    interface: Interface[S],
    rng: np.random.Generator,
) -> tuple[Batch, RolloutState[S], list[EpisodeResult]]:
    """Step every env params.rollout_steps times with the sampled policy,
    which sees the envs and acts on them through interface. An env whose
    episode ends starts over in the same step.

    On truncation, the reward of the last step gets gamma * V(final observation),
    so that the time limit does not look like a terminal state to GAE.
    """
    t_len, n, obs_dim = params.rollout_steps, env.nbatch, interface.obs_dim
    obs_buf = np.zeros((t_len, n, obs_dim), dtype=np.float32)
    act_buf = np.zeros((t_len, n, ACTION_DIM), dtype=np.float32)
    logp_buf = np.zeros((t_len, n), dtype=np.float32)
    val_buf = np.zeros((t_len, n), dtype=np.float32)
    rew_buf = np.zeros((t_len, n), dtype=np.float32)
    done_buf = np.zeros((t_len, n), dtype=np.float32)
    env_state, ep_ret, ep_steps, stats, carry = (
        state.env,
        state.episode_return.copy(),
        state.episode_steps.copy(),
        state.stats,
        state.carry,
    )
    finished: list[EpisodeResult] = []
    step_keys = jax.random.split(key, t_len)

    for t in range(t_len):
        carry, raw_obs = interface.observe(carry, env_state.obs, rng)
        stats = update_obs_stats(stats, raw_obs)
        obs = normalize_obs(stats, raw_obs, params.obs_clip)
        action, logp, val = sample_action(net, jnp.asarray(obs), step_keys[t])
        action_np = np.asarray(action, dtype=np.float64)
        obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action_np, logp, val
        carry, applied = interface.act(carry, action_np)
        env_state, out = env.step(env_state, applied)
        ep_ret += out.reward
        ep_steps += 1
        rew_buf[t] = out.reward
        done_buf[t] = out.done
        if out.truncated.any():
            # The value of the last observation, as the policy would see it.
            _, final_obs = interface.observe(carry, out.obs, rng)
            final = normalize_obs(stats, final_obs, params.obs_clip)
            final_value = np.asarray(value(net, jnp.asarray(final)))
            rew_buf[t] += np.where(out.truncated, params.gamma * final_value, 0.0)
        finished.extend(ended(out, ep_ret, ep_steps).values())
        ep_ret[out.done], ep_steps[out.done] = 0.0, 0
        if out.done.any():
            carry = interface.restart(carry, out.done, env_state.obs, rng)

    _, raw_obs = interface.observe(carry, env_state.obs, rng)
    last_obs = normalize_obs(stats, raw_obs, params.obs_clip)
    last_value = value(net, jnp.asarray(last_obs))
    advantage, target = gae(
        jnp.asarray(rew_buf),
        jnp.asarray(val_buf),
        jnp.asarray(done_buf),
        last_value,
        params.gamma,
        params.gae_lambda,
    )
    batch = Batch(
        obs=jnp.asarray(obs_buf).reshape(t_len * n, obs_dim),
        action=jnp.asarray(act_buf).reshape(t_len * n, ACTION_DIM),
        log_prob=jnp.asarray(logp_buf).reshape(-1),
        advantage=advantage.reshape(-1),
        target=target.reshape(-1),
    )
    next_state = RolloutState(env_state, ep_ret, ep_steps, stats, carry)
    return batch, next_state, finished


@dataclass(frozen=True)
class UpdateLog:
    update: int
    env_steps: int
    episodes: tuple[EpisodeResult, ...]
    """The episodes that ended in this rollout."""
    loss: LossInfo
    action_std: tuple[float, ...]
    """The standard deviation of each action."""


def format_log(log: UpdateLog, steps_per_s: float) -> str:
    eps = log.episodes
    ret = np.mean([e.episode_return for e in eps]) if eps else float("nan")
    lap = np.mean([e.lap_frac for e in eps]) if eps else float("nan")
    reasons = Counter(e.outcome for e in eps)
    return (
        f"update {log.update:4d}  steps {log.env_steps:8d}  {steps_per_s:5.0f} steps/s  "
        f"episodes {len(eps):3d}  return {ret:+7.2f}  lap_frac {lap:+.3f}  "
        f"laps {reasons['lap_complete']:2d} crash {reasons['crash']:2d} "
        f"stuck {reasons['stuck']:2d} rollover {reasons['rollover']:2d} "
        f"trunc {reasons['truncated']:2d}  "
        f"pi {float(log.loss.policy_loss):+.4f} v {float(log.loss.value_loss):.4f} "
        f"kl {float(log.loss.approx_kl):.4f} clip {float(log.loss.clip_frac):.2f} "
        f"std {log.action_std[0]:.2f},{log.action_std[1]:.2f}"
    )


def train[S](
    params: PPOParams,
    total_steps: int,
    seed: int,
    num_envs: int,
    backend: Backend,
    config: NeoRacerEnvConfig,
    interface: Interface[S],
) -> tuple[ActorCritic, ObsStats]:
    """Train for at least total_steps env steps. Print one log line per update.
    The spawns, the physics and the interface's randomness come from
    default_rng(seed); each update draws new spawns and physics."""
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    net = init_actor_critic(init_key, params, interface.obs_dim)
    opt_state = init_opt_state(net, params)
    env = NeoRacerBatch(backend, num_envs, config)
    rng = np.random.default_rng(seed)
    env_state = env.reset(rng)
    state = RolloutState(
        env=env_state,
        episode_return=np.zeros(num_envs),
        episode_steps=np.zeros(num_envs, dtype=np.int64),
        stats=empty_obs_stats(interface.obs_dim),
        carry=interface.start(env_state.obs, rng),
    )
    steps_per_update = params.rollout_steps * num_envs
    updates = -(-total_steps // steps_per_update)  # ceiling division
    for update in range(1, updates + 1):
        start = time.perf_counter()
        key, rollout_key, update_key = jax.random.split(key, 3)
        batch, state, finished = collect_rollout(
            env, state, net, rollout_key, params, interface, rng
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
        print(
            format_log(log, steps_per_update / (time.perf_counter() - start)),
            flush=True,
        )
    env.close()
    return net, state.stats


def save_policy(path: str, net: ActorCritic, stats: ObsStats) -> None:
    np.savez(path, allow_pickle=False, **policy_arrays(net, stats))


def policy_arrays(net: ActorCritic, stats: ObsStats) -> dict[str, np.ndarray]:
    """The arrays of net and stats, by the names that load_policy reads."""
    arrays: dict[str, np.ndarray] = {
        "log_std": np.asarray(net.log_std),
        "obs_mean": stats.mean,
        "obs_var": stats.var,
        "obs_count": np.array(stats.count),
    }
    for name, layers in (("actor", net.actor), ("critic", net.critic)):
        for i, layer in enumerate(layers):
            arrays[f"{name}.{i}.weight"] = np.asarray(layer.weight)
            arrays[f"{name}.{i}.bias"] = np.asarray(layer.bias)
    return arrays


def load_policy(path: str) -> tuple[ActorCritic, ObsStats]:
    """Parse a file from save_policy(). Raises KeyError if an array is missing."""
    with np.load(path) as f:

        def layers(name: str) -> tuple[Dense, ...]:
            count = sum(
                1 for k in f.files if k.startswith(f"{name}.") and k.endswith(".weight")
            )
            if count == 0:
                raise KeyError(f"{path} has no {name} layers")
            return tuple(
                Dense(
                    jnp.asarray(f[f"{name}.{i}.weight"]),
                    jnp.asarray(f[f"{name}.{i}.bias"]),
                )
                for i in range(count)
            )

        net = ActorCritic(layers("actor"), layers("critic"), jnp.asarray(f["log_std"]))
        stats = ObsStats(f["obs_mean"], f["obs_var"], float(f["obs_count"]))
    return net, stats


# --- Evaluation -------------------------------------------------------------


@dataclass(frozen=True)
class Trained:
    net: ActorCritic
    stats: ObsStats
    params: PPOParams
    """Its obs_clip must be the training value."""


@dataclass(frozen=True)
class RandomActions: ...


@dataclass(frozen=True)
class ZeroAction: ...


type EvalPolicy = Trained | RandomActions | ZeroAction
type Baseline = Literal["random", "zero"]


def evaluate(
    policy: EvalPolicy,
    episodes: int,
    first_seed: int,
    rng: np.random.Generator,
    config: NeoRacerEnvConfig,
) -> list[EpisodeResult]:
    """Run episodes of NeoRacerEnv(config) with env seeds first_seed,
    first_seed + 1, ... A trained policy acts with its mean action
    (deterministic). RandomActions draws uniform actions from rng."""
    env = NeoRacerEnv(config)
    results = []
    for k in range(episodes):
        obs, _ = env.reset(seed=first_seed + k)
        ep_ret, steps = 0.0, 0
        while True:
            match policy:
                case Trained(net=net, stats=stats, params=p):
                    norm = normalize_obs(
                        stats, obs[None].astype(np.float64), p.obs_clip
                    )
                    mean = np.asarray(greedy_action(net, jnp.asarray(norm)))[0]
                    action = mean
                case RandomActions():
                    action = rng.uniform(-1.0, 1.0, ACTION_DIM)
                case ZeroAction():
                    action = np.zeros(ACTION_DIM)
                case _:
                    assert_never(policy)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            steps += 1
            if terminated or truncated:
                results.append(
                    EpisodeResult(
                        ep_ret,
                        info["lap_frac"],
                        _outcome(info),
                        steps,
                    )
                )
                break
    env.close()
    return results


def print_eval(name: str, results: list[EpisodeResult], first_seed: int) -> None:
    print(f"[eval] {name}: {len(results)} episodes, env seeds {first_seed}..")
    for k, r in enumerate(results):
        print(
            f"  seed {first_seed + k:6d}  return {r.episode_return:+8.3f}  "
            f"lap_frac {r.lap_frac:+.3f}  steps {r.steps:4d}  outcome {r.outcome}"
        )
    print(
        f"  mean return {np.mean([r.episode_return for r in results]):+.3f}  "
        f"mean lap_frac {np.mean([r.lap_frac for r in results]):+.3f}  "
        f"outcomes {dict(Counter(r.outcome for r in results))}"
    )


def parse_backend(text: str) -> Literal["cpu", "warp"]:
    match text:
        case "cpu" | "warp":
            return text
        case _:
            raise argparse.ArgumentTypeError(
                f"backend must be cpu or warp, not {text!r}"
            )


def parse_tracks(text: str) -> Literal["loop", "generated"]:
    match text:
        case "loop" | "generated":
            return text
        case _:
            raise argparse.ArgumentTypeError(
                f"tracks must be loop or generated, not {text!r}"
            )


def add_track_arguments(
    parser: argparse.ArgumentParser, default: Literal["loop", "generated"]
) -> None:
    """The options that parse_track_arguments reads."""
    parser.add_argument(
        "--tracks",
        type=parse_tracks,
        default=default,
        help="loop (loop_corridor) or generated (a pool of generated tracks)",
    )
    parser.add_argument("--pool", type=int, default=16, help="generated tracks")
    parser.add_argument("--difficulty", type=int, choices=range(4), default=1)
    parser.add_argument(
        "--first-track-seed",
        type=int,
        default=0,
        help="the pool is the first --pool seeds from this one that give a track",
    )


def parse_track_arguments(args: argparse.Namespace) -> Tracks:
    """The tracks that the options of add_track_arguments name."""
    kind: Literal["loop", "generated"] = args.tracks
    match kind:
        case "loop":
            return LoopCorridor()
        case "generated":
            return GeneratedTracks(args.difficulty, args.pool, args.first_track_seed)
        case _:
            assert_never(kind)


def parse_baseline(text: str) -> Baseline:
    match text:
        case "random" | "zero":
            return text
        case _:
            raise argparse.ArgumentTypeError(
                f"baseline must be random or zero, not {text!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="env steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--backend", type=parse_backend, default="cpu", help="cpu or warp (GPU)"
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads that step the envs, with --backend cpu",
    )
    parser.add_argument("--out", help="save the trained policy to this .npz file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", metavar="PATH", help="evaluate a saved policy")
    mode.add_argument("--eval-baseline", type=parse_baseline, help="random or zero")
    parser.add_argument("--episodes", type=int, default=10, help="evaluation episodes")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=10_000,
        help="env seed of the first evaluation episode",
    )
    parser.add_argument(
        "--random-physics",
        action="store_true",
        help="train and evaluate with the car physics drawn from RANDOM_PHYSICS",
    )
    add_track_arguments(parser, default="loop")
    args = parser.parse_args()
    params = PPOParams()
    tracks = parse_track_arguments(args)
    config = NeoRacerEnvConfig(
        tracks=tracks,
        physics=RANDOM_PHYSICS if args.random_physics else PhysicsRanges(),
    )
    rng = np.random.default_rng(args.seed)  # actions of the random baseline

    if args.eval is not None:
        net, stats = load_policy(args.eval)
        policy: EvalPolicy = Trained(net, stats, params)
        results = evaluate(policy, args.episodes, args.eval_seed, rng, config)
        print_eval(f"policy {args.eval}", results, args.eval_seed)
        return
    if args.eval_baseline is not None:
        baseline: Baseline = args.eval_baseline
        match baseline:
            case "random":
                policy = RandomActions()
            case "zero":
                policy = ZeroAction()
            case _:
                assert_never(baseline)
        results = evaluate(policy, args.episodes, args.eval_seed, rng, config)
        print_eval(baseline, results, args.eval_seed)
        return

    print(f"[train_ppo] {params}")
    print(
        f"[train_ppo] total_steps={args.total_steps} seed={args.seed} "
        f"num_envs={args.num_envs} backend={args.backend} "
        f"random_physics={args.random_physics} tracks={tracks}"
    )
    kind: Literal["cpu", "warp"] = args.backend
    match kind:
        case "cpu":
            backend: Backend = Cpu(nthread=min(args.nthread, args.num_envs))
        case "warp":
            backend = Warp()
        case _:
            assert_never(kind)
    net, stats = train(
        params, args.total_steps, args.seed, args.num_envs, backend, config, Exact()
    )
    if args.out is not None:
        save_policy(args.out, net, stats)
        print(f"[train_ppo] saved the policy to {args.out}")


if __name__ == "__main__":
    main()
