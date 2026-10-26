"""
Tests for neoracer_rl.envs.NeoRacerEnv: the Gymnasium environment
interface. These tests do not train a policy. validation/test_train_ppo.py
tests the PPO baseline.

Random actions rarely complete a lap or roll the car over. So
validation/test_task.py states the laws of the task's functions over many
inputs, and the env tests here check that step() uses them.
"""

import os
import pickle
import re
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from cases import case
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from neoracer_mujoco import assets, export, sim
from neoracer_mujoco.physics import PhysicsRanges, Range
from neoracer_mujoco.racecar import neoracer_params
from neoracer_mujoco.track_generation import Track
from neoracer_mujoco.track_generation.mjcf import to_start_frame
from neoracer_spec.neoracer import NEORACER

from neoracer_rl.envs import GeneratedTracks, LapEnds, NeoRacerEnv, NeoRacerEnvConfig
from neoracer_rl.envs.neoracer_env import arenas, lap_angle

LAP = 2.0 * np.pi
"""One full lap of progress (rad)."""


@pytest.fixture
def env():
    e = NeoRacerEnv()
    yield e
    e.close()


# --- construction / spaces --------------------------------------------------


def test_env_rejects_a_car_without_a_free_joint(tmp_path):
    """__init__ parses the car's free joint once. A car without one fails
    there with a clear error, not later in reset()."""
    xml = re.sub(r"<freejoint\s*/>", "", export.car_xml(NEORACER)).replace(
        'file="meshes/', f'file="{os.path.join(assets.ASSET_DIR, "meshes")}/'
    )
    car = tmp_path / "no_freejoint.xml"
    car.write_text(xml)
    with pytest.raises(ValueError, match="free joint"):
        NeoRacerEnv(NeoRacerEnvConfig(car_path=str(car)))


def test_action_space_is_valid_box(env):
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


def test_observation_space_is_valid_box(env):
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (20,)
    assert env.observation_space.dtype == np.float32
    assert np.all(np.isfinite(env.observation_space.low))
    assert np.all(np.isfinite(env.observation_space.high))


def test_gymnasium_check_env_passes(env):
    """Gymnasium's env checker checks the API, the dtypes and the spaces,
    that reset and step return what the spaces allow, and that a seed gives
    the same reset and the same step, so no test here repeats them."""
    check_env(env, skip_render_check=True)


# --- reset() / step() basic contract ----------------------------------------


def test_full_bounded_episode_runs_without_crashing(env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(env.config.max_episode_steps):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, reward, terminated, truncated, _info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    else:
        pytest.fail("episode neither terminated nor truncated within max_episode_steps")


# --- determinism / seeding ---------------------------------------------------


def test_different_seeds_produce_different_spawns():
    env = NeoRacerEnv()
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)
    assert not np.array_equal(obs_a, obs_b)
    env.close()


def _global_random_state() -> bytes:
    """The state of the global np.random generator, as bytes that compare
    with ==."""
    return pickle.dumps(np.random.get_state())


def test_reset_and_step_do_not_use_global_numpy_random(env):
    """The env takes its random numbers from env.np_random only. So reset()
    and step() do not change the state of the global np.random generator."""
    before = _global_random_state()
    env.reset(seed=7)
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
    env.reset()
    assert _global_random_state() == before


# --- action handling ----------------------------------------------------------


def test_out_of_range_actions_are_clamped_not_rejected():
    """step() clamps the action to [-1, 1]. So action [5, -5] writes the
    same ctrl as action [1, -1], and does not raise an error."""
    env_a, env_b = NeoRacerEnv(), NeoRacerEnv()
    env_a.reset(seed=3)
    env_b.reset(seed=3)
    env_a.step(np.array([5.0, -5.0], dtype=np.float32))
    env_b.step(np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(env_a.data.ctrl, env_b.data.ctrl)
    env_a.close()
    env_b.close()


def test_without_a_speed_loop_full_speed_gives_max_speed_of_the_throttle():
    """On a car without a speed loop, action[0] = 1 is the throttle max_speed:
    on the NeoRacer's motor, that share of its voltage limit, the battery's."""
    for scale in (0.1, 0.5, 1.0):
        with case(scale):
            car = replace(neoracer_params(), speed_loop=None)
            config = NeoRacerEnvConfig(max_speed=scale, car=car)
            env = NeoRacerEnv(config)
            env.reset(seed=0)
            env.step(np.array([1.0, 0.0], dtype=np.float32))
            limit = env.model.bind_scalar(env.scene.car.drive).ctrlrange[1]  # V
            ctrl = env.data.bind_scalar(env.scene.car.drive).ctrl[0]
            env.close()
            assert ctrl == pytest.approx(scale * limit)


def test_the_car_holds_the_speed_that_the_action_asks_for():
    """action[0] asks the NeoRacer's speed loop for that share of max_speed
    times the loop's max speed. Straight ahead from the start of
    loop_corridor, the car's speed is within 0.1 m/s of it after 1.5 s."""
    loop = NeoRacerEnvConfig().car.speed_loop
    assert loop is not None
    for speed, max_speed in ((1.0, 0.1), (0.5, 0.2), (0.4, 0.5)):
        with case((speed, max_speed)):
            env = NeoRacerEnv(NeoRacerEnvConfig(max_speed=max_speed))
            env.reset(seed=0)
            for _ in range(round(1.5 * env.config.control_hz)):
                _obs, _r, terminated, _t, info = env.step(
                    np.array([speed, 0.0], dtype=np.float32)
                )
                assert not terminated
            env.close()
            target = speed * max_speed * loop.max_speed
            assert info["speed_ms"] == pytest.approx(target, abs=0.1)


def test_max_speed_outside_0_1_is_rejected():
    for scale in (0.0, -0.1, 1.5):
        with case(scale), pytest.raises(ValueError, match="max_speed"):
            NeoRacerEnvConfig(max_speed=scale)


def test_wall_crash_observations_stay_in_space():
    """Every observation up to a wall crash is inside observation_space. The
    actions are full speed, which is full throttle at the full battery
    voltage, and random steering from default_rng(13)."""
    env = NeoRacerEnv()
    rng = np.random.default_rng(13)
    env.reset(seed=13)
    reason = None
    for _ in range(env.config.max_episode_steps):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        action[0] = 1.0
        obs, _reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            reason = info["termination_reason"]
            break
    env.close()
    assert reason == "crash"


def test_repeated_random_actions_produce_no_nans(env):
    env.reset(seed=0)
    rng = np.random.default_rng(123)
    for _ in range(300):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, reward, terminated, truncated, _info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset(seed=0)


# --- the reward in the env ------------------------------------------------------


def _episode_return(speed: float, seed: int) -> tuple[float, str | None]:
    """The return and the end of an episode with a constant action: the
    speed, and no steering."""
    env = NeoRacerEnv()
    env.reset(seed=seed)
    total, reason = 0.0, None
    for _ in range(env.config.max_episode_steps):
        _obs, r, terminated, truncated, info = env.step(
            np.array([speed, 0.0], dtype=np.float32)
        )
        total += r
        if terminated or truncated:
            reason = info["termination_reason"]
            break
    env.close()
    return total, reason


def test_trying_the_first_corner_beats_not_moving():
    """A car that drives straight at the first corner and hits the wall
    returns more than a car that does not move. Otherwise "do not move"
    would be a local optimum that a learner has to escape."""
    for seed in range(3):
        with case(seed):
            still, still_end = _episode_return(0.0, seed)
            tried, tried_end = _episode_return(1.0, seed)
            assert (still_end, tried_end) == ("stuck", "crash")
            assert tried > still


# --- termination and truncation in the env ------------------------------------


def test_rollover_terminates(env):
    """Tip the car onto its side. The next step ends the episode with the
    reason "rollover" and a negative reward."""
    env.reset(seed=0)
    # A 90 degree rotation about the body X axis turns the car +Z axis to
    # world -Y. So upright_cos becomes 0, below upright_rollover_cos.
    half = np.pi / 4.0  # rad, half of the rotation angle
    env.data.bind_scalar(env.scene.car.joint).qpos[3:7] = [
        np.cos(half),
        np.sin(half),
        0.0,
        0.0,
    ]
    mujoco.mj_forward(env.model, env.data)
    assert sim.car_upright_cos(env.data, env.scene.car) < LapEnds().upright_rollover_cos

    _obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "rollover"
    assert reward < 0


def test_stuck_terminates(env):
    """With zero speed, the car makes no progress. So the stuck check
    ends the episode after progress_window_s."""
    env.reset(seed=0)
    cfg = env.config
    window_steps = round(cfg.progress_window_s * cfg.control_hz)
    reason = None
    for _ in range(window_steps + 20):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            reason = info["termination_reason"]
            break
    assert reason == "stuck"


def test_lap_frac_is_signed_progress_and_negative_in_reverse(env):
    """info["lap_frac"] is progress_rad / 2 pi at every step. If the car
    drives backward, both values are negative."""
    env.reset(seed=0)
    for _ in range(60):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([-0.1, 0.0], dtype=np.float32)
        )
        assert info["lap_frac"] == info["progress_rad"] / LAP
        assert not (terminated or truncated)
    assert info["lap_frac"] < 0.0


def test_timeout_truncates():
    config = NeoRacerEnvConfig(max_episode_steps=5)
    env = NeoRacerEnv(config=config)
    env.reset(seed=0)
    results = [
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        for _ in range(config.max_episode_steps)
    ]
    _obs, _reward, terminated, truncated, info = results[-1]
    assert truncated
    assert not terminated
    assert info["termination_reason"] is None
    assert not any(r[3] for r in results[:-1]), "truncated before the last step"
    env.close()


# --- state isolation -----------------------------------------------------------


def test_model_state_does_not_leak_between_episodes(env):
    """reset() and step() write only env.data. They do not change env.model."""
    body_pos_before = env.model.body_pos.copy()
    geom_size_before = env.model.geom_size.copy()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
    env.reset(seed=1)
    np.testing.assert_array_equal(env.model.body_pos, body_pos_before)
    np.testing.assert_array_equal(env.model.geom_size, geom_size_before)


def test_reset_draws_the_physics_of_the_episode():
    """Each reset draws a Physics from config.physics and steps with it: the
    wheel friction of the model is the XML's times the drawn factor, the
    same seed draws the same physics, and the nominal model stays as it was."""
    ranges = PhysicsRanges(wheel_friction=Range(0.3, 0.9))
    env = NeoRacerEnv(NeoRacerEnvConfig(physics=ranges))
    wheel = env.scene.car.wheels[0]
    nominal = float(env.scene.model.bind_scalar(wheel).friction[0])
    drawn = []
    for seed in (0, 1, 0):
        env.reset(seed=seed)
        factor = env.physics.wheel_friction
        assert 0.3 <= factor <= 0.9
        friction = env.model.bind_scalar(wheel).friction[0]
        assert friction == pytest.approx(nominal * factor)
        drawn.append(env.physics)
    assert drawn[0] == drawn[2] != drawn[1]
    assert env.scene.model.bind_scalar(wheel).friction[0] == nominal
    env.close()


def test_lap_angle_follows_the_centerline_of_a_generated_track():
    """With the car moved along the centerline of a generated track, the lap
    angle is 2 pi times the share of the lap behind it."""
    (arena,) = arenas(NeoRacerEnvConfig(tracks=GeneratedTracks(count=1)), False)
    track, scene = arena.layout, arena.scene
    assert isinstance(track, Track)
    data = mujoco.MjData(scene.model)
    center = to_start_frame(track, track.center)
    for i in range(0, len(center), 97):
        data.bind_scalar(scene.car.joint).qpos[:2] = center[i]
        mujoco.mj_forward(scene.model, data)
        expected = 2.0 * np.pi * track.distance_along[i] / track.total_length
        assert lap_angle(arena, data.sensordata) == pytest.approx(expected)


def test_a_pool_gives_episodes_on_several_tracks():
    """reset draws the track of each episode from the pool, and the car
    starts at the beginning of the lap on each."""
    env = NeoRacerEnv(NeoRacerEnvConfig(tracks=GeneratedTracks(count=3)))
    seen = set()
    for seed in range(12):
        obs, _info = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        seen.add(id(env.scene))
        angle = env._progress.angle[0]
        assert min(angle, 2.0 * np.pi - angle) < 0.1  # rad, at the start
    assert len(seen) == 3
    env.close()


def test_too_few_seeds_for_a_pool_is_an_error():
    with pytest.raises(ValueError, match="pool"):
        GeneratedTracks(count=0)
