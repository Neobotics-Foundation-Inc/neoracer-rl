"""
Laws for the PPO baseline (examples/train_ppo.py): GAE, the clipped loss,
the observation statistics, and a short training run on NeoRacerBatch.
"""

import argparse
from dataclasses import replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cases import case

from examples.train_ppo import (
    ACTION_DIM,
    Batch,
    Dense,
    Exact,
    ObsStats,
    PPOParams,
    RolloutState,
    add_track_arguments,
    clipped_surrogate,
    collect_rollout,
    empty_obs_stats,
    gae,
    init_actor_critic,
    init_opt_state,
    load_policy,
    parse_track_arguments,
    ppo_loss,
    ppo_update,
    save_policy,
    update_obs_stats,
)
from neoracer_rl.envs import (
    GeneratedTracks,
    LoopCorridor,
    NeoRacerEnvConfig,
    RewardConfig,
)
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch
from neoracer_rl.envs.neoracer_env import OBS_DIM, Tracks


def _reward_to_go(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """Undiscounted sum of the rewards to the end of each episode, per column."""
    out = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + running * (1.0 - dones[t])
        out[t] = running
    return out


def test_gae_with_unit_gamma_lambda_and_zero_values_is_reward_to_go():
    for seed in range(5):
        with case(seed):
            rng = np.random.default_rng(seed)
            rewards = rng.normal(size=(12, 3))
            dones = (rng.uniform(size=(12, 3)) < 0.2).astype(np.float64)
            zeros = jnp.zeros((12, 3))
            adv, ret = gae(
                jnp.asarray(rewards), zeros, jnp.asarray(dones), jnp.zeros(3), 1.0, 1.0
            )
            expected = _reward_to_go(rewards, dones)
            np.testing.assert_allclose(adv, expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(ret, expected, rtol=1e-5, atol=1e-5)


def test_gae_does_not_leak_across_a_done_step():
    for seed in range(5):
        with case(seed):
            rng = np.random.default_rng(seed)
            rewards = jnp.asarray(rng.normal(size=(10, 2)))
            values = jnp.asarray(rng.normal(size=(10, 2)))
            last_value = jnp.asarray(rng.normal(size=2))
            dones = np.zeros((10, 2))
            dones[4] = 1.0  # an episode ends after step 4
            adv, _ = gae(rewards, values, jnp.asarray(dones), last_value, 0.97, 0.9)
            # Changes after the done step do not change the advantages up to it.
            adv2, _ = gae(
                rewards.at[5:].add(100.0),
                values.at[5:].add(-50.0),
                jnp.asarray(dones),
                last_value + 7.0,
                0.97,
                0.9,
            )
            np.testing.assert_allclose(adv[:5], adv2[:5], rtol=1e-6, atol=1e-6)
            # At the done step, the advantage is the reward minus the value.
            np.testing.assert_allclose(
                adv[4], rewards[4] - values[4], rtol=1e-6, atol=1e-6
            )


def test_clipped_surrogate_is_zero_for_zero_advantage():
    ratio = jnp.linspace(0.1, 3.0, 50)
    np.testing.assert_array_equal(clipped_surrogate(ratio, jnp.zeros(50), 0.2), 0.0)


def test_clipped_surrogate_has_no_gradient_past_the_clip_in_the_improving_direction():
    eps = 0.2
    for advantage in (-2.0, -0.5, 0.5, 2.0):
        with case(advantage):

            def loss(r, a=advantage):
                return clipped_surrogate(r, jnp.asarray(a), eps)

            # A ratio that already moved past the clip in the direction that
            # the advantage rewards gets no more gradient.
            past = 1.5 if advantage > 0 else 0.5
            assert float(jax.grad(loss)(past)) == 0.0
            np.testing.assert_allclose(
                loss(jnp.asarray(past)),
                -(1.0 + np.sign(advantage) * eps) * advantage,
                rtol=1e-6,
            )
            # Inside the clip range, the loss is -ratio * advantage.
            np.testing.assert_allclose(jax.grad(loss)(1.05), -advantage, rtol=1e-6)


def test_policy_loss_is_zero_when_all_advantages_are_zero():
    params = PPOParams()
    net = init_actor_critic(jax.random.key(0), params)
    rng = np.random.default_rng(0)
    batch = Batch(
        obs=jnp.asarray(rng.normal(size=(32, OBS_DIM))),
        action=jnp.asarray(rng.normal(size=(32, 2))),
        log_prob=jnp.asarray(rng.normal(size=32)),
        advantage=jnp.zeros(32),
        target=jnp.zeros(32),
    )
    _, info = ppo_loss(net, batch, params)
    assert float(info.policy_loss) == 0.0


def test_ppo_params_reject_values_outside_their_domain():
    # Each value has its field's type, which no checker can see here.
    outside: list[tuple[str, Any]] = [
        ("gamma", 0.0),
        ("gae_lambda", 1.5),
        ("rollout_steps", 0),
        ("minibatches", 300),  # more than rollout_steps (256)
        ("clip_eps", 0.0),
        ("learning_rate", -1e-4),
        ("max_grad_norm", 0.0),
        ("obs_clip", -1.0),
        ("value_coef", -0.5),
        ("entropy_coef", -0.01),
    ]
    for field, bad in outside:
        with case(field), pytest.raises(ValueError, match=field):
            PPOParams(**{field: bad})


def test_running_obs_stats_match_numpy_over_all_batches():
    for seed in range(3):
        with case(seed):
            rng = np.random.default_rng(seed)
            batches = [
                rng.normal(
                    loc=rng.uniform(-50, 50, OBS_DIM),
                    scale=rng.uniform(0.01, 100, OBS_DIM),
                    size=(int(rng.integers(1, 40)), OBS_DIM),
                )
                for _ in range(6)
            ]
            stats = empty_obs_stats()
            for b in batches:
                stats = update_obs_stats(stats, b)
            data = np.concatenate(batches)
            assert stats.count == data.shape[0]
            np.testing.assert_allclose(
                stats.mean, data.mean(axis=0), rtol=1e-9, atol=1e-9
            )
            np.testing.assert_allclose(
                stats.var, data.var(axis=0), rtol=1e-9, atol=1e-9
            )


def test_short_training_run_changes_the_params_and_keeps_them_finite(tmp_path):
    params = PPOParams(rollout_steps=128, epochs=2, minibatches=2, hidden=(16,))
    env = NeoRacerBatch(Cpu(nthread=0), nbatch=2)
    rng = np.random.default_rng(0)
    state = RolloutState(
        env.reset(rng),
        np.zeros(2),
        np.zeros(2, dtype=np.int64),
        empty_obs_stats(),
        None,
    )
    net = init_actor_critic(jax.random.key(1), params)
    opt_state = init_opt_state(net, params)
    start = net
    # One update from 128 * 2 = 256 env steps.
    batch, state, _ = collect_rollout(
        env, state, net, jax.random.key(10), params, Exact(), rng
    )
    env.close()
    net, opt_state, info = ppo_update(net, opt_state, batch, jax.random.key(20), params)
    assert all(np.isfinite(float(x)) for x in jax.tree.leaves(info))
    leaves, start_leaves = jax.tree.leaves(net), jax.tree.leaves(start)
    assert all(np.all(np.isfinite(x)) for x in leaves)
    assert any(
        not np.array_equal(a, b) for a, b in zip(leaves, start_leaves, strict=True)
    )
    assert state.stats.count == 128 * 2

    # A saved policy loads back to the same arrays.
    save_policy(str(tmp_path / "p.npz"), net, state.stats)
    loaded, stats = load_policy(str(tmp_path / "p.npz"))
    for a, b in zip(jax.tree.leaves(loaded), leaves, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(stats.mean, state.stats.mean)


def test_truncation_bootstraps_from_the_last_observation_of_the_episode():
    """The env gives no reward here, and the critic returns the speed in the
    observation. An episode truncated after step t then gets the target
    gamma * (speed after step t): the value of the last observation of the
    episode, not of the first one of the next, and GAE stops there."""
    nsteps, episode, speed = 6, 3, OBS_DIM - 1  # speed: observation index
    params = PPOParams(rollout_steps=nsteps, minibatches=1, hidden=(), obs_clip=1e9)
    reward_free = RewardConfig(progress_weight=0.0)
    config = NeoRacerEnvConfig(max_episode_steps=episode, reward=reward_free)
    env = NeoRacerBatch(Cpu(nthread=0), nbatch=2, config=config)
    pick = np.zeros((OBS_DIM, 1))
    pick[speed] = 1.0
    net = init_actor_critic(jax.random.key(0), params)
    net = replace(net, critic=(Dense(jnp.asarray(pick), jnp.zeros(1)),))
    # With this many observations seen, normalize_obs is the identity.
    stats = ObsStats(np.zeros(OBS_DIM), np.ones(OBS_DIM), 1e15)
    rng = np.random.default_rng(0)
    start = env.reset(rng)
    state = RolloutState(start, np.zeros(2), np.zeros(2, dtype=np.int64), stats, None)
    batch, _, finished = collect_rollout(
        env, state, net, jax.random.key(1), params, Exact(), rng
    )
    assert [e.outcome for e in finished] == ["truncated"] * 4

    # Replay the actions to see the observation after each step.
    actions = np.asarray(batch.action).reshape(nsteps, 2, ACTION_DIM)
    target = np.asarray(batch.target).reshape(nsteps, 2)
    replay = env.reset(np.random.default_rng(0))
    for t in range(nsteps):
        replay, out = env.step(replay, actions[t])
        if (t + 1) % episode == 0:
            assert out.truncated.all()
            last = out.obs[:, speed]  # m/s
            assert np.all(np.abs(last - replay.obs[:, speed]) > 1e-3)
            np.testing.assert_allclose(target[t], params.gamma * last, rtol=1e-4)
    env.close()


def test_the_track_options_name_the_tracks():
    generated = ("--tracks", "generated")
    cases: tuple[tuple[Literal["loop", "generated"], tuple[str, ...], Tracks], ...] = (
        ("loop", (), LoopCorridor()),
        ("generated", (), GeneratedTracks()),
        ("generated", ("--tracks", "loop"), LoopCorridor()),
        ("loop", generated, GeneratedTracks()),
        (
            "loop",
            (*generated, "--pool", "4", "--difficulty", "3", "--first-track-seed", "9"),
            GeneratedTracks(difficulty=3, count=4, first_seed=9),
        ),
    )
    for default, argv, tracks in cases:
        parser = argparse.ArgumentParser()
        add_track_arguments(parser, default=default)
        assert parse_track_arguments(parser.parse_args(argv)) == tracks, argv
