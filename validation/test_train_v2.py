"""
Laws for the V2's trainer, examples/train_v2.py: the model's LiDAR as
racecar_core gives it, its noise, the encoder and the late actions, a saved policy as the car loads
it, and a short training run on the V2's model.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from neoracer_mujoco.racecar_core_adapter import racecar_neo_v2_params

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
EXACT = SensingParams(range_noise=0.0, dropout=0.0)  # the model's ranges as they are


def test_the_v2s_beams_show_a_new_scan_once_a_turn_of_the_head():
    """Through NeoRacerBatch, the V2 model's beams change all at once, about
    7.4 times a second: the batch keeps the beams' history from step to step,
    and a beam reads the last complete turn."""
    seconds = 3.0
    config = NeoRacerEnvConfig(car_path=CAR)
    env = NeoRacerBatch(Cpu(nthread=0), nbatch=2, config=config)
    state = env.reset(np.random.default_rng(7))
    ranges = true_ranges(state.obs)
    scans = np.zeros(2, dtype=np.int64)
    for _ in range(round(seconds / policy.CONTROL_PERIOD)):
        state, _ = env.step(state, np.tile([0.5, 0.2], (2, 1)))
        now = true_ranges(state.obs)
        changed = now != ranges
        # A turn shows within a physics step on every beam, so almost always
        # in one control step.
        scans += changed.sum(axis=1) >= len(policy.BEAM_DEGREES) - 1
        ranges = now
    env.close()
    scan_rate = racecar_neo_v2_params().lidar.scan_rate  # turns/s
    np.testing.assert_allclose(scans, seconds * scan_rate, atol=2)


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


def test_a_scan_keeps_its_noise_and_lost_beams_until_the_beams_change():
    cars = 16
    rng = np.random.default_rng(6)
    params = SensingParams(dropout=0.2)
    ranges = rng.uniform(0.5, 4.0, (cars, BEAMS))
    s = start(params, ranges, rng)
    s, before = sense(params, s, ranges, np.zeros(cars), rng)
    for k in range(100):
        # The beams show a new scan every 4 control steps.
        if k % 4 == 0:
            ranges = rng.uniform(0.5, 4.0, (cars, BEAMS))
        s, obs = sense(params, s, ranges, np.zeros(cars), rng)
        same = np.all(obs[:, :BEAMS] == before[:, :BEAMS], axis=1)
        np.testing.assert_array_equal(same, k % 4 != 0)
        before = obs


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
    np.testing.assert_array_equal(r.scan[rows], ranges[rows])


def test_sensing_params_reject_values_outside_their_domain():
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
