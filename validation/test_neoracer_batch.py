"""
Laws for neoracer_rl.envs.neoracer_batch: each row of NeoRacerBatch is a
NeoRacerEnv episode, and a row on the Cpu backend steps like NeoRacerEnv,
bit for bit. validation/test_task.py has the laws of the task's functions,
which both use.
"""

from dataclasses import replace

import numpy as np
import pytest
from cases import case
from neoracer_mujoco.physics import Physics, PhysicsRanges, Range

from neoracer_rl.envs import GeneratedTracks, NeoRacerEnv, NeoRacerEnvConfig
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch, StepResult, Warp


def _reason(out: StepResult, row: int) -> str | None:
    """The end of row's episode in out, or None while it goes on."""
    code = int(out.reason[row])
    return out.ends[code - 1] if code else None


RANDOM_PHYSICS = PhysicsRanges(
    wheel_friction=Range(0.4, 1.2),
    chassis_mass=Range(0.7, 1.5),
    motor_torque=Range(0.6, 1.4),
    steer_stiffness=Range(0.5, 1.5),
)


def test_a_cpu_row_steps_like_neoracer_env():
    """Rows 0 and 1 of a batch reset with default_rng(7) start where
    NeoRacerEnv starts after reset(seed=7) and after a second reset, on the
    same track and with the same physics. With the same actions, they give
    the same observations, rewards and episode ends, bit for bit, until the
    first episode ends."""
    configs = {
        "nominal": NeoRacerEnvConfig(),
        "random-physics": NeoRacerEnvConfig(physics=RANDOM_PHYSICS),
        "generated-tracks": NeoRacerEnvConfig(
            tracks=GeneratedTracks(count=3), physics=RANDOM_PHYSICS
        ),
    }
    for name, config in configs.items():
        with case(name):
            _steps_like_neoracer_env(config)


def _steps_like_neoracer_env(config: NeoRacerEnvConfig) -> None:
    sim_batch = NeoRacerBatch(Cpu(nthread=2), nbatch=2, config=config)
    state = sim_batch.reset(np.random.default_rng(7))
    envs = [NeoRacerEnv(config), NeoRacerEnv(config)]
    first = envs[0].reset(seed=7)[0]
    envs[1].reset(seed=7)
    second = envs[1].reset()[0]
    np.testing.assert_array_equal(state.obs, np.stack([first, second]))
    assert (envs[0].physics == Physics()) == (config.physics == PhysicsRanges())
    rng = np.random.default_rng(1)
    ended = False
    while not ended:
        action = rng.uniform((0.0, -1.0), (1.0, 1.0), (2, 2)).astype(np.float32)
        state, out = sim_batch.step(state, action)
        for i, env in enumerate(envs):
            obs, r, terminated, truncated, info = env.step(action[i])
            np.testing.assert_array_equal(out.obs[i], obs)
            assert out.reward[i] == r
            assert _reason(out, i) == info["termination_reason"]
            assert out.truncated[i] == truncated
            ended |= terminated or truncated
    sim_batch.close()
    for env in envs:
        env.close()


def test_a_row_whose_episode_ends_starts_over_from_its_spawn():
    """Row 0 drives into the wall at full speed and row 1 stands still.
    When row 0 crashes, it is back at its spawn in the same step, and row 1
    goes on."""
    sim_batch = NeoRacerBatch(Cpu(nthread=0), nbatch=2)
    start = sim_batch.reset(np.random.default_rng(0))
    action = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    state, out = sim_batch.step(start, action)
    while not out.done[0]:
        state, out = sim_batch.step(state, action)
    sim_batch.close()
    assert _reason(out, 0) == "crash" and not out.done[1]
    np.testing.assert_array_equal(state.obs[0], start.obs[0])
    assert state.progress.steps[0] == 0 and state.progress.total[0] == 0.0
    assert state.progress.steps[1] > 0
    np.testing.assert_array_equal(state.obs[1], out.obs[1])


def test_respawn_changes_only_where_rows_start_over():
    """respawn draws new spawns and leaves the running episodes alone. A row
    whose episode ends after it starts over from its new spawn."""
    sim_batch = NeoRacerBatch(Cpu(nthread=0), nbatch=2)
    rng = np.random.default_rng(0)
    action = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    before, _ = sim_batch.step(sim_batch.reset(rng), action)
    state = sim_batch.respawn(before, rng)
    np.testing.assert_array_equal(state.sim, before.sim)
    np.testing.assert_array_equal(state.obs, before.obs)
    assert state.progress is before.progress
    assert not np.array_equal(state.spawn_obs, before.spawn_obs)
    state, out = sim_batch.step(state, action)
    while not out.done[0]:
        state, out = sim_batch.step(state, action)
    sim_batch.close()
    assert not out.done[1]
    np.testing.assert_array_equal(state.obs[0], state.spawn_obs[0])
    assert state.progress.angle[0] == state.spawn_angle[0]


def test_the_warp_backend_follows_the_cpu_backend():
    """On MuJoCo Warp, a short drive from the same spawns stays within 5 cm
    of the CPU backend in every LiDAR range. Warp has its own collision and
    solver code, so the rows are close, not equal. This test runs only where
    neoracer-mujoco's jax extra is installed."""
    for module in ("jax", "mujoco.mjx", "mujoco_warp"):
        pytest.importorskip(module)
    observations = []
    for backend in (Cpu(nthread=0), Warp()):
        sim_batch = NeoRacerBatch(backend, nbatch=2)
        state = sim_batch.reset(np.random.default_rng(3))
        action = np.array([[0.6, 0.2], [0.6, -0.2]], dtype=np.float32)
        for _ in range(10):
            state, out = sim_batch.step(state, action)
        observations.append(out.obs)
        sim_batch.close()
    np.testing.assert_allclose(
        observations[1][:, :8], observations[0][:, :8], atol=0.05
    )


def test_warp_refuses_a_pool_of_tracks():
    """MuJoCo Warp steps one track per batch."""
    for module in ("jax", "mujoco.mjx", "mujoco_warp"):
        pytest.importorskip(module)
    config = NeoRacerEnvConfig(tracks=GeneratedTracks(count=2))
    with pytest.raises(ValueError, match="one track per batch"):
        NeoRacerBatch(Warp(), nbatch=2, config=config)


def test_a_row_that_starts_over_switches_track_physics_and_progress_together():
    """On a pool with random physics, row 0 ends its episode after a respawn
    drew it another track. It then starts over on the new track, with the
    model, track index, first observation and lap angle of its spawn, and
    steps like NeoRacerEnv from that spawn, bit for bit, once NeoRacerEnv's
    speed loop ticks where the batch's loops do: a row that starts over keeps
    the batch's ticks. Row 1 goes on."""
    config = NeoRacerEnvConfig(tracks=GeneratedTracks(count=3), physics=RANDOM_PHYSICS)
    sim_batch = NeoRacerBatch(Cpu(nthread=0), nbatch=2, config=config)
    state = sim_batch.reset(np.random.default_rng(0))
    # A respawn from default_rng(seed) draws row 0 the spawn of reset(seed).
    seed = next(
        k
        for k in range(1, 50)
        if sim_batch.respawn(state, np.random.default_rng(k)).spawn_track[0]
        != state.track[0]
    )
    state = sim_batch.respawn(state, np.random.default_rng(seed))
    first_track = int(state.track[1])
    action = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    state, out = sim_batch.step(state, action)
    while not out.done[0]:
        state, out = sim_batch.step(state, action)
    assert not out.done[1] and state.track[1] == first_track
    assert state.track[0] == state.spawn_track[0]
    assert state.sim.models[0] is state.spawn.models[0]
    np.testing.assert_array_equal(state.obs[0], state.spawn_obs[0])
    assert state.progress.angle[0] == state.spawn_angle[0]

    env = NeoRacerEnv(config)
    obs, _ = env.reset(seed=seed)
    np.testing.assert_array_equal(state.obs[0], obs)
    assert env.loop is not None and state.loop is not None
    ticks = replace(
        env.loop.measure, span=state.loop.measure.span, due=state.loop.measure.due
    )
    env.loop = replace(env.loop, measure=ticks)
    rng = np.random.default_rng(2)
    for _ in range(30):
        row = rng.uniform((0.0, -1.0), (0.5, 1.0)).astype(np.float32)
        state, out = sim_batch.step(state, np.stack([row, action[1]]))
        obs, reward, terminated, truncated, _ = env.step(row)
        np.testing.assert_array_equal(out.obs[0], obs)
        assert out.reward[0] == reward
        if terminated or truncated:
            break
    sim_batch.close()
    env.close()
