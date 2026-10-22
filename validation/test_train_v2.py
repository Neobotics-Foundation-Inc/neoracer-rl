"""
Laws for the V2's trainer, examples/train_v2.py: the LiDAR as racecar_core
gives it, the encoder and the late actions, a saved policy as the car loads
it, and a short training run on the V2's model.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.racecar_neo_v2 import policy
from examples.train_ppo import (
    ObsStats,
    PPOParams,
    greedy_action,
    init_opt_state,
    load_policy,
    normalize_obs,
    ppo_update,
)
from examples.train_v2 import (
    CAR,
    PHYSICS,
    RolloutState,
    SensingParams,
    collect_rollout,
    delay,
    init_net,
    restart,
    save_policy,
    sense,
    start,
    true_ranges,
)
from neoracer_rl.envs import NeoRacerEnvConfig
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch

BEAMS = len(policy.BEAM_DEGREES)
EXACT = SensingParams(range_noise=0.0, dropout=0.0)  # the ranges as the head saw them


def test_racecar_core_gives_a_whole_turn_at_once_at_most_two_turns_old():
    cars, steps, cp = 16, 300, policy.CONTROL_PERIOD

    # The true range of beam b at step k is 0.01 k + 0.001 b, so a range
    # tells when the head saw it.
    def true(k: int) -> np.ndarray:
        return np.tile(0.01 * k + 0.001 * np.arange(BEAMS), (cars, 1))

    rng = np.random.default_rng(0)
    s = start(EXACT, true(0), rng)
    seen_before = np.zeros((cars, BEAMS), dtype=np.intp)
    changes: list[list[int]] = [[] for _ in range(cars)]
    for k in range(1, steps):
        s, obs = sense(EXACT, s, true(k), np.zeros(cars), rng)
        seen = np.rint((obs[:, :BEAMS] - 0.001 * np.arange(BEAMS)) / 0.01)
        age = (k - seen) * cp  # s
        assert np.all(age >= 0.0)
        assert np.all(age <= 2.0 * s.period[:, None] + cp / 2)
        # In the first two turns, some beams still read the car standing at
        # its spawn. Then every beam changes when a turn ends, and none between.
        changed = (seen != seen_before)[k * cp > 2.0 * s.period]
        assert np.all(changed.all(axis=1) | ~changed.any(axis=1))
        for i in np.flatnonzero((seen != seen_before).all(axis=1)):
            changes[i].append(k)
        seen_before = seen
    for i in range(cars):
        gaps = np.diff(changes[i][1:]) * cp
        assert np.all(np.abs(gaps - s.period[i]) <= cp), (i, gaps, s.period[i])


def test_the_policy_sees_the_scaled_wheel_speed_and_its_own_last_action():
    rng = np.random.default_rng(1)
    ranges = rng.uniform(0.2, 4.0, (8, BEAMS))
    s = start(EXACT, ranges, rng)
    s, _ = delay(s, rng.uniform(-1.5, 1.5, (8, 2)))
    speed = rng.uniform(0.0, 1.2, 8)
    s, obs = sense(EXACT, s, ranges, speed, rng)
    np.testing.assert_array_equal(obs[:, :BEAMS], ranges)  # a still scene
    np.testing.assert_allclose(obs[:, BEAMS], speed * s.speed_scale)
    np.testing.assert_array_equal(obs[:, BEAMS + 1 :], s.last_action)
    assert np.all(np.abs(s.last_action) <= 1.0)


def test_a_late_car_gets_the_last_action_and_an_on_time_car_the_new_one():
    rng = np.random.default_rng(2)
    s = start(SensingParams(late=0.5), rng.uniform(0.2, 4.0, (64, BEAMS)), rng)
    assert s.late.any() and not s.late.all()
    first, second = rng.uniform(-1.0, 1.0, (2, 64, 2))
    s, _ = delay(s, first)
    s, applied = delay(s, second)
    np.testing.assert_array_equal(applied[s.late], first[s.late])
    np.testing.assert_array_equal(applied[~s.late], second[~s.late])
    np.testing.assert_array_equal(s.last_action, second)


def test_a_restart_changes_only_the_cars_that_start_over():
    rng = np.random.default_rng(3)
    params = SensingParams()
    s = start(params, rng.uniform(0.2, 4.0, (10, BEAMS)), rng)
    for _ in range(5):
        s, _ = sense(params, s, rng.uniform(0.2, 4.0, (10, BEAMS)), np.ones(10), rng)
    rows = np.arange(10) % 3 == 0
    ranges = rng.uniform(0.2, 4.0, (10, BEAMS))
    r = restart(params, s, rows, ranges, rng)
    for name in s.__dataclass_fields__:
        before, after = getattr(s, name), getattr(r, name)
        np.testing.assert_array_equal(after[~rows], before[~rows], err_msg=name)
    np.testing.assert_array_equal(r.time[rows], 0.0)
    np.testing.assert_array_equal(r.history[rows], np.repeat(ranges[rows, None], 10, 1))


def test_sensing_params_reject_values_outside_their_domain():
    with pytest.raises(ValueError, match="scan_period"):
        SensingParams(scan_period=(0.2, 0.1))
    with pytest.raises(ValueError, match="history"):
        SensingParams(history=5)  # less than two turns
    with pytest.raises(ValueError, match="range_noise"):
        SensingParams(range_noise=-0.01)
    with pytest.raises(ValueError, match="dropout"):
        SensingParams(dropout=1.5)
    with pytest.raises(ValueError, match="late"):
        SensingParams(late=-0.1)


def test_the_car_acts_as_the_trained_networks_mean_action(tmp_path):
    params = PPOParams()
    config = NeoRacerEnvConfig(car_path=CAR, throttle_scale=0.7)
    net = init_net(jax.random.key(0), params)
    # A network whose output layer is not the near-zero one of the start.
    net = replace(net, actor=jax.tree.map(lambda x: 100.0 * x, net.actor))
    rng = np.random.default_rng(4)
    stats = ObsStats(
        rng.normal(size=policy.OBS_DIM), rng.uniform(0.1, 3.0, policy.OBS_DIM), 100.0
    )
    path = str(tmp_path / "v2.npz")
    save_policy(path, net, stats, params, config)
    car = policy.load(path)
    assert car.throttle_scale == 0.7 and car.obs_clip == params.obs_clip
    obs = rng.normal(0.0, 3.0, (200, policy.OBS_DIM))
    mean = greedy_action(net, jnp.asarray(normalize_obs(stats, obs, params.obs_clip)))
    expected = np.clip(np.asarray(mean), -1.0, 1.0)
    assert 0.1 < np.mean(np.abs(expected) == 1.0) < 0.9  # some actions clip
    # JAX computes in float32, the car in float64.
    np.testing.assert_allclose(policy.act(car, obs), expected, atol=1e-4)
    # train_ppo reads the same file.
    loaded, _ = load_policy(path)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(net), strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_short_training_run_on_the_v2_changes_the_params_and_keeps_them_finite():
    params = PPOParams(rollout_steps=64, epochs=2, minibatches=2, hidden=(16,))
    sensing = SensingParams()
    config = NeoRacerEnvConfig(car_path=CAR, physics=PHYSICS)
    env = NeoRacerBatch(Cpu(nthread=0), nbatch=2, config=config)
    rng = np.random.default_rng(5)
    env_state = env.reset(rng)
    state = RolloutState(
        env_state,
        start(sensing, true_ranges(env_state.obs), rng),
        np.zeros(2),
        np.zeros(2, dtype=np.int64),
        ObsStats(np.zeros(policy.OBS_DIM), np.ones(policy.OBS_DIM), 0.0),
    )
    net = init_net(jax.random.key(1), params)
    opt_state = init_opt_state(net, params)
    batch, state, _ = collect_rollout(
        env, state, net, jax.random.key(2), params, sensing, rng
    )
    env.close()
    new, _, info = ppo_update(net, opt_state, batch, jax.random.key(3), params)
    assert all(np.isfinite(float(x)) for x in jax.tree.leaves(info))
    leaves = jax.tree.leaves(new)
    assert all(np.all(np.isfinite(x)) for x in leaves)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(leaves, jax.tree.leaves(net), strict=True)
    )
    assert state.stats.count == 64 * 2
