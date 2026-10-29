"""
Laws for races (neoracer_rl.envs.race, stepped by neoracer_batch.NeoRacerRace):
the agents of Each and Joint are the same learners, a Policy car drives as a
Learner given its actions, each car's Step sees the others, a learner's end
ends its race, the controllers drive, no car of a grid starts against a wall
or another car, and a car turned by pi drives back around the lap.
validation/test_neoracer_batch.py has the laws of NeoRacerBatch, a race of
one Learner.
"""

import os
from dataclasses import replace

import numpy as np
import pytest
from cases import case
from neoracer_mujoco import assets, batch, sensors
from neoracer_mujoco.racecar import racecar_neo_v2_params
from neoracer_mujoco.sim import SpawnRange

from neoracer_rl.envs import GeneratedTracks, NeoRacerEnvConfig, Step
from neoracer_rl.envs.neoracer_batch import BatchState, Cpu, NeoRacerRace, Warp
from neoracer_rl.envs.race import (
    Joint,
    Learner,
    Policy,
    RaceConfig,
    Racer,
    TrackCentering,
    WallFollowing,
)
from neoracer_rl.envs.task import CarObservation, Observations

AHEAD = Racer(Learner(), (1.0, 0.0))
"""A learner 1 m ahead of the track's start."""

GRID = tuple(
    Racer(offset=(-(k // 2) * 1.2, 0.275 if k % 2 else -0.275)) for k in range(4)
)
"""Four learners on a grid at the track's start: rows of two, 0.55 m apart,
1.2 m between rows."""

STILL = np.array([0.0, 0.0], np.float32)
FULL = np.array([1.0, 0.0], np.float32)


def _race(nrace: int, config: RaceConfig) -> tuple[NeoRacerRace, BatchState]:
    env = NeoRacerRace(Cpu(nthread=0), nrace, config)
    return env, env.reset(np.random.default_rng(0))


def test_a_joint_agent_is_its_learners_side_by_side():
    """From the same spawns, with the same actions, a Joint agent observes
    what its learners observe as Each's agents, side by side, earns the sum
    of their rewards, and its race ends where theirs does."""
    each = RaceConfig(cars=(Racer(), AHEAD))
    joint = replace(each, agents=Joint())
    (e, es), (j, js) = _race(3, each), _race(3, joint)
    assert (e.nbatch, e.action_dim, j.nbatch, j.action_dim) == (6, 2, 3, 4)
    np.testing.assert_array_equal(js.obs, es.obs.reshape(3, -1))
    rng = np.random.default_rng(1)
    for _ in range(60):
        action = rng.uniform((0.0, -1.0), (1.0, 1.0), (6, 2)).astype(np.float32)
        es, eo = e.step(es, action)
        js, jo = j.step(js, action.reshape(3, 4))
        np.testing.assert_array_equal(jo.obs, eo.obs.reshape(3, -1))
        np.testing.assert_array_equal(jo.reward, eo.reward.reshape(3, 2).sum(axis=1))
        np.testing.assert_array_equal(jo.done, eo.done.reshape(3, 2).any(axis=1))
        np.testing.assert_array_equal(jo.laps, eo.laps)
    e.close()
    j.close()


def _steer_from(obs: Observations) -> Observations:
    """An action of the observations: half speed, steering toward the side
    whose beams at 45 and 315 degrees read farther."""
    steer = np.clip(obs[:, 1] - obs[:, 7], -1.0, 1.0)
    return np.stack([np.full(len(obs), 0.5), steer], axis=1).astype(np.float32)


def _act(step: Step) -> Observations:
    return _steer_from(CarObservation()(step))


def test_a_policy_car_drives_as_a_learner_given_its_actions():
    """A car that a Policy drives steps as a Learner, bit for bit, when the
    learner is given the actions that the Policy would give from its
    observations."""
    with_policy = RaceConfig(cars=(Racer(), Racer(Policy(_act), (1.0, 0.0))))
    two_learners = RaceConfig(cars=(Racer(), AHEAD))
    (p, ps), (ll, ls) = _race(2, with_policy), _race(2, two_learners)
    np.testing.assert_array_equal(ps.obs, ls.obs[0::2])
    rng = np.random.default_rng(2)
    for _ in range(90):
        first = rng.uniform((0.0, -1.0), (1.0, 1.0), (2, 2)).astype(np.float32)
        ahead = _steer_from(ls.obs[1::2])
        ps, po = p.step(ps, first)
        ls, lo = ll.step(ls, np.stack([first, ahead], axis=1).reshape(4, 2))
        np.testing.assert_array_equal(po.obs, lo.obs[0::2])
        np.testing.assert_array_equal(po.reward, lo.reward[0::2])
        np.testing.assert_array_equal(po.laps, lo.laps)
    p.close()
    ll.close()


def _ahead(step: Step) -> Observations:
    """How far ahead the first rival is, and whether it touched the car."""
    rival = step.rivals[0]
    return np.stack([rival.ahead, rival.contact], axis=1).astype(np.float32)


def test_each_car_sees_how_far_ahead_the_other_is():
    """Of two learners, each sees the other as far ahead as the other sees
    it behind: 1 m at the start, as progress around the loop, and then the
    start's gap plus the other's progress less its own."""
    config = RaceConfig(env=NeoRacerEnvConfig(observe=_ahead), cars=(Racer(), AHEAD))
    env, state = _race(2, config)
    start = state.obs[0::2, 0]
    assert (start > 0).all()
    np.testing.assert_array_equal(state.obs[1::2, 0], -start)
    rng = np.random.default_rng(3)
    for _ in range(60):
        action = rng.uniform((0.0, -0.3), (1.0, 0.3), (4, 2)).astype(np.float32)
        state, out = env.step(state, action)
        if out.done.any():
            break
        np.testing.assert_allclose(out.obs[1::2, 0], -out.obs[0::2, 0], atol=1e-6)
        gained = 2.0 * np.pi * (out.laps[:, 1] - out.laps[:, 0])
        np.testing.assert_allclose(out.obs[0::2, 0], start + gained, atol=1e-5)
    env.close()


def test_two_cars_that_touch_both_see_it():
    """A learner drives at full speed into a learner standing 1 m ahead of
    it. Each one's rival contact is the other's in every step, and it holds
    in some."""
    config = RaceConfig(env=NeoRacerEnvConfig(observe=_ahead), cars=(Racer(), AHEAD))
    env, state = _race(1, config)
    touched = []
    for _ in range(45):
        state, out = env.step(state, np.stack([FULL, STILL]))
        assert out.obs[0, 1] == out.obs[1, 1]
        touched.append(bool(out.obs[0, 1]))
        if out.done.any():
            break
    env.close()
    assert any(touched)


def test_a_learners_end_ends_its_race():
    """A learner drives into the wall while the other stands still. In the
    step of its crash, the other learner is truncated, and the race starts
    over: both at their spawns, the opponent too. The other race goes on."""
    config = RaceConfig(cars=(Racer(), AHEAD, Racer(TrackCentering(), (-1.0, 0.0))))
    env, start = _race(2, config)
    crash = np.array([1.0, 1.0], np.float32)
    action = np.stack([crash, STILL, STILL, STILL])
    state, out = env.step(start, action)
    while not out.done[0]:
        state, out = env.step(state, action)
    env.close()
    assert out.ends[out.reason[0] - 1] == "crash"
    assert out.truncated[1] and not out.done[2:].any()
    np.testing.assert_array_equal(state.obs[:2], start.obs[:2])
    for k, progress in enumerate(state.progress):
        with case(k):
            assert progress.steps[0] == 0 and progress.total[0] == 0.0
            assert progress.steps[1] > 0
    np.testing.assert_array_equal(state.plan.target[0], start.plan.target[0])
    assert state.plan.controllers[2][0] == start.plan.controllers[2][0]


def test_a_controller_drives_its_car_around_the_loop():
    """A controller, 1 m ahead of a learner that stands still, drives its
    car forward around the loop: a twentieth of a lap in 3.5 s, while the
    learner stays where it is."""
    for driver in (TrackCentering(), WallFollowing()):
        with case(driver):
            config = RaceConfig(cars=(Racer(), Racer(driver, (1.0, 0.0))))
            env, state = _race(2, config)
            for _ in range(105):
                state, out = env.step(state, np.stack([STILL, STILL]))
                assert not out.done.any()
            env.close()
            assert (out.laps[:, 1] > 0.05).all()
            assert (np.abs(out.laps[:, 0]) < 1e-3).all()


def _both_lidars(step: Step) -> Observations:
    """The LiDAR of CarObservation, and the LiDAR of the car's readings."""
    own = [
        sensors.lidar_ranges(batch.readings(step.model, step.car, step.sensordata[r]))
        for r in range(len(step.sensordata))
    ]
    return np.concatenate([CarObservation()(step)[:, :8], np.stack(own)], axis=1)


def test_a_learner_of_a_race_observes_its_own_cars_lidar():
    """Each learner of a race observes the LiDAR of its own car, as
    batch.readings gives it, the beam that the car blocks at no return."""
    config = RaceConfig(
        env=NeoRacerEnvConfig(observe=_both_lidars), cars=(Racer(), AHEAD)
    )
    env, state = _race(2, config)
    blocked = sensors.LIDAR_BEAM_ORDER.index("lidar_180")
    for _ in range(10):
        state, out = env.step(state, np.tile(FULL, (4, 1)))
        np.testing.assert_array_equal(out.obs[:, :8], out.obs[:, 8:].astype(np.float32))
        assert (out.obs[:, blocked] == sensors.NO_RETURN).all()
    env.close()


def _ends_of_the_car(step: Step) -> dict[str, np.ndarray]:
    """An end named after the step's car, so each car's has another name."""
    return {step.car.body.name: np.zeros(len(step.contact), np.bool_)}


def test_a_race_refuses_ends_that_differ_from_car_to_car():
    """A terminate that names the ends of two learners differently raises
    ValueError at the first step, since a row's end code would mean another
    end in each."""
    config = RaceConfig(
        env=NeoRacerEnvConfig(terminate=_ends_of_the_car), cars=(Racer(), AHEAD)
    )
    env, state = _race(1, config)
    with pytest.raises(ValueError, match="same ends"):
        env.step(state, np.stack([STILL, STILL]))
    env.close()


def test_a_race_needs_a_learner_and_a_controller_needs_a_speed_loop():
    """A race of no Learner, and a controller on a car whose model holds its
    own speed, raise ValueError."""
    v2 = NeoRacerEnvConfig(
        car_path=os.path.join(assets.ASSET_DIR, "racecar_neo_v2.xml"),
        car=racecar_neo_v2_params(),
    )
    with pytest.raises(ValueError, match="needs a Learner"):
        RaceConfig(cars=(Racer(TrackCentering()),))
    with pytest.raises(ValueError, match="no speed loop"):
        RaceConfig(env=v2)


def test_a_race_on_warp_follows_the_race_on_the_cpu():
    """On MuJoCo Warp, a short race of two learners and a TrackCentering
    car, from the same spawns, stays within 5 cm of the CPU race in every
    learner's LiDAR range. This test runs only where neoracer-mujoco's jax
    extra is installed."""
    for module in ("jax", "mujoco.mjx", "mujoco_warp"):
        pytest.importorskip(module)
    config = RaceConfig(cars=(Racer(), AHEAD, Racer(TrackCentering(), (-1.0, 0.0))))
    observations = []
    for backend in (Cpu(nthread=0), Warp()):
        env = NeoRacerRace(backend, 2, config)
        state = env.reset(np.random.default_rng(3))
        action = np.tile([0.6, 0.2], (4, 1)).astype(np.float32)
        for _ in range(10):
            state, out = env.step(state, action)
        observations.append(out.obs)
        env.close()
    np.testing.assert_allclose(
        observations[1][:, :8], observations[0][:, :8], atol=0.05
    )


def _touches(step: Step) -> Observations:
    """Whether the car touched a wall, and each rival, in the step."""
    touched = [step.contact, *(rival.contact for rival in step.rivals)]
    return np.stack(touched, axis=1).astype(np.float32)


def test_no_car_of_a_grid_starts_against_a_wall_or_another_car():
    """On generated tracks, gentle and tight, no car of a grid of four that
    stands still touches a wall in the first step after a spawn, nor
    another car: the grid's cars stand farther apart than a spawn moves
    them."""
    spawn = SpawnRange(lateral=0.02, yaw=0.1)
    for difficulty in (1, 3):
        tracks = GeneratedTracks(difficulty, count=6)
        env_config = NeoRacerEnvConfig(tracks=tracks, spawn=spawn, observe=_touches)
        env = NeoRacerRace(Cpu(nthread=0), 16, RaceConfig(env_config, GRID))
        for seed in range(3):
            with case((difficulty, seed)):
                state = env.reset(np.random.default_rng(seed))
                state, out = env.step(state, np.zeros((env.nbatch, 2), np.float32))
                assert not out.done.any()
                assert not out.obs.any()
        env.close()


def _progress(step: Step) -> Observations:
    """How far around the track the car has gone in the episode."""
    return step.progress.total[:, None].astype(np.float32)


def test_a_car_turned_by_pi_drives_back_around_the_lap():
    """A learner whose heading is pi faces back along the track: at full
    throttle it goes back around the lap, while the learner beside it with
    no heading goes forward."""
    turned = Racer(offset=(0.0, 0.275), heading=np.pi)
    racers = (Racer(offset=(0.0, -0.275)), turned)
    tracks, spawn = GeneratedTracks(count=2), SpawnRange(lateral=0.02, yaw=0.1)
    env_config = NeoRacerEnvConfig(tracks=tracks, spawn=spawn, observe=_progress)
    config = RaceConfig(env_config, racers)
    env, state = _race(4, config)
    for _ in range(20):
        state, out = env.step(state, np.tile(FULL, (env.nbatch, 1)))
        assert not out.done.any()
    assert (out.obs[0::2, 0] > 0.05).all()
    assert (out.obs[1::2, 0] < -0.05).all()
    env.close()
