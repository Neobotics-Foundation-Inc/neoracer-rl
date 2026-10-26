"""
Laws for neoracer_rl.envs.task: the progress of a batch of rows, first_end,
which picks each row's end, the ends and the reward of the lap task, and an
env's use of its observe, terminate and reward.
"""

import numpy as np
import pytest
from cases import case
from gymnasium import spaces
from neoracer_mujoco import batch, sim
from neoracer_mujoco.scene import Scene

from neoracer_rl.envs import (
    Ends,
    LapEnds,
    NeoRacerEnv,
    NeoRacerEnvConfig,
    ProgressReward,
    Step,
)
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch
from neoracer_rl.envs.neoracer_env import arenas
from neoracer_rl.envs.task import (
    Mask,
    Observations,
    ProgressRows,
    Rows,
    advance_rows,
    first_end,
    restart_rows,
    start_rows,
)

LAP = 2.0 * np.pi
"""One full lap of progress (rad)."""


def _wrap(angle: Rows) -> Rows:
    """angle (rad) in [-pi, pi), the range of the lap angle on loop_corridor."""
    return np.remainder(angle + np.pi, LAP) - np.pi


# --- progress -------------------------------------------------------------------


def test_a_step_is_in_minus_pi_to_pi_and_a_whole_turn_from_the_difference():
    """For any two angles, the step's progress is in [-pi, pi), and differs
    from the plain difference only by a full turn."""
    rng = np.random.default_rng(0)
    old, new = rng.uniform(-np.pi, np.pi, (2, 2000))
    p = advance_rows(start_rows(old, 5), new)
    assert np.all((-np.pi <= p.delta) & (p.delta < np.pi))
    turns = (p.delta - (new - old)) / LAP
    np.testing.assert_allclose(turns, np.round(turns), atol=1e-9)
    np.testing.assert_array_equal(p.total, p.delta)
    np.testing.assert_array_equal(p.angle, new)


def test_half_a_lap_either_way_is_minus_pi():
    half = np.pi / 2
    old = np.array([0.0, np.pi, -half, half])
    p = advance_rows(start_rows(old, 5), np.array([np.pi, 0.0, half, -half]))
    np.testing.assert_array_equal(p.delta, -np.pi)


def test_the_total_is_the_true_progress():
    """If the car moves less than half a turn per step, the total is the true
    change of angle, although each angle wraps."""
    rng = np.random.default_rng(1)
    steps = rng.uniform(-0.9 * np.pi, 0.9 * np.pi, (40, 50))  # rad per step and row
    start = rng.uniform(-np.pi, np.pi, 50)
    p = start_rows(start, 7)
    for true_angle in start + np.cumsum(steps, axis=0):
        p = advance_rows(p, _wrap(true_angle))
    np.testing.assert_allclose(p.total, steps.sum(axis=0), atol=1e-9)


def test_recent_is_the_progress_over_the_window():
    """recent is inf until a row has had window_len - 1 steps. Then it is the
    total minus the total window_len - 1 steps before."""
    rng = np.random.default_rng(2)
    window_len, rows = 6, 30
    p = start_rows(np.zeros(rows), window_len)
    totals = [p.total]
    for step in range(1, 16):
        p = advance_rows(p, _wrap(p.angle + rng.normal(0.0, 0.1, rows)))
        totals.append(p.total)
        with case(step):
            if step < window_len - 1:
                assert np.all(p.recent == np.inf)
            else:
                before = totals[step - window_len + 1]
                np.testing.assert_array_equal(p.recent, p.total - before)


def test_restart_starts_over_only_the_rows_that_are_done():
    rng = np.random.default_rng(3)
    p = start_rows(np.zeros(6), 4)
    for _ in range(5):
        p = advance_rows(p, _wrap(p.angle + rng.normal(0.0, 0.3, 6)))
    done = np.array([True, False, True, False, False, True])
    angle = rng.uniform(-np.pi, np.pi, 6)
    q = restart_rows(p, done, angle)
    start = start_rows(angle, 4)
    for field in ("angle", "total", "delta", "steps", "window"):
        with case(field):
            new, old, fresh = (getattr(x, field) for x in (q, p, start))
            np.testing.assert_array_equal(new[done], fresh[done])
            np.testing.assert_array_equal(new[~done], old[~done])


# --- ends and rewards -------------------------------------------------------------


def test_first_end_picks_the_first_end_each_row_meets():
    """A row's code is k + 1 for the first of the ends, in order, that it
    meets, and 0 if it meets none. The ends that first_end gives are true
    only for that one."""
    rng = np.random.default_rng(4)
    names = ("a", "b", "c")
    ends = {name: rng.random(500) < 0.3 for name in names}
    codes, first = first_end(ends, 500)
    for i in range(500):
        met = [k for k, name in enumerate(names, 1) if ends[name][i]]
        assert codes[i] == (met[0] if met else 0)
        assert [bool(first[n][i]) for n in names] == [k == codes[i] for k in (1, 2, 3)]
    assert first_end({}, 4)[0].tolist() == [0, 0, 0, 0]


@pytest.fixture(scope="module")
def scene() -> Scene:
    return arenas(NeoRacerEnvConfig(), visuals=False)[0].scene


def _step(
    scene: Scene,
    upright_cos: Rows,
    contact: Mask,
    total: Rows,
    stuck: Mask,
    delta: Rows | None = None,
) -> Step:
    """A Step of the car at rest at its spawn, but with these tilts, contacts
    and progress: stuck rows made no progress over their window, the others
    one radian."""
    rows = len(total)
    sensordata = np.tile(sim.settle(scene.model).sensordata, (rows, 1))
    zaxis = batch.read(scene.model, sensordata, scene.car.truth.zaxis)
    zaxis[:, 2] = upright_cos  # zaxis is a view into sensordata
    progress = ProgressRows(
        angle=np.zeros(rows),
        total=total,
        delta=np.zeros(rows) if delta is None else delta,
        steps=np.ones(rows, np.int64),
        window=np.stack([total - np.where(stuck, 0.0, 1.0), total], axis=1),
    )
    action = np.zeros((rows, 2), np.float32)
    return Step(scene.model, scene.car, sensordata, contact, progress, action)


def test_lap_ends_come_in_their_order(scene: Scene):
    """Rollover comes first, then a crash, then stuck, then a lap. A lap is
    complete if and only if the total progress is at least +2 pi, so progress
    backward never completes one."""
    cfg = LapEnds()
    rng = np.random.default_rng(5)
    totals = [*rng.uniform(-2.0 * LAP, 2.0 * LAP, 200), LAP, -LAP, 0.0]
    tilts = (-1.0, cfg.upright_rollover_cos - 1e-9, 0.9, 1.0)
    grid = np.array(
        [(u, c, s, t) for u in tilts for c in (0, 1) for s in (0, 1) for t in totals]
    )
    upright_cos, contact, stuck, total = grid.T
    step = _step(scene, upright_cos, contact > 0, total, stuck > 0)
    codes, _ = first_end(cfg(step), len(grid))
    names = ["", *cfg(step)]
    expected = np.where(
        upright_cos < cfg.upright_rollover_cos,
        "rollover",
        np.where(
            contact > 0,
            "crash",
            np.where(stuck > 0, "stuck", np.where(total >= LAP, "lap_complete", "")),
        ),
    )
    assert [names[c] for c in codes] == expected.tolist()


def test_progress_reward_adds_the_amount_of_the_end(scene: Scene):
    """Without an end, the reward is progress_weight * delta. An end adds
    lap_bonus or subtracts its penalty, and an end it does not know adds
    nothing."""
    reward = ProgressReward()
    rng = np.random.default_rng(6)
    rows = 200
    delta = rng.uniform(-np.pi, np.pi, rows)
    ones, zeros = np.ones(rows), np.zeros(rows)
    step = _step(scene, ones, zeros > 0, zeros, zeros > 0, delta)
    amounts = {
        "rollover": -reward.rollover_penalty,
        "crash": -reward.crash_penalty,
        "stuck": -reward.stuck_penalty,
        "lap_complete": reward.lap_bonus,
    }
    base = reward(step, {name: zeros > 0 for name in amounts})
    np.testing.assert_array_equal(base, reward.progress_weight * delta)
    for name, amount in amounts.items():
        with case(name):
            ends = {other: np.full(rows, other == name) for other in amounts}
            np.testing.assert_allclose(reward(step, ends), base + amount)
    np.testing.assert_array_equal(reward(step, {"unknown": ones > 0}), base)


# --- an env's own functions --------------------------------------------------------


def _steps_so_far(step: Step) -> Observations:
    return step.progress.steps[:, None].astype(np.float32)


def _after_three(step: Step) -> Ends:
    return {"far": step.progress.steps >= 3}


def _one_then_seven(step: Step, ends: Ends) -> Rows:
    return np.where(ends["far"], 7.0, 1.0)


def test_the_env_and_the_batch_give_what_their_functions_give():
    """With its own observe, terminate and reward, an env gives their
    observations, ends and rewards: here the steps so far, an end after 3
    steps, and a reward of 1 a step and 7 at the end. NeoRacerEnv's
    observation space is unbounded, of the observation's size."""
    config = NeoRacerEnvConfig(
        observe=_steps_so_far, terminate=_after_three, reward=_one_then_seven
    )
    still = np.zeros(2, np.float32)
    env = NeoRacerEnv(config)
    space = env.observation_space
    assert isinstance(space, spaces.Box) and space.shape == (1,)
    assert np.all(space.low == -np.inf) and np.all(space.high == np.inf)
    obs, _ = env.reset(seed=0)
    seen = [(obs.tolist(), None, False, None)]
    for _ in range(3):
        obs, r, terminated, _, info = env.step(still)
        seen.append((obs.tolist(), r, terminated, info["termination_reason"]))
    env.close()
    assert seen == [
        ([0.0], None, False, None),
        ([1.0], 1.0, False, None),
        ([2.0], 1.0, False, None),
        ([3.0], 7.0, True, "far"),
    ]

    rows = NeoRacerBatch(Cpu(nthread=0), nbatch=2, config=config)
    state = rows.reset(np.random.default_rng(0))
    rewards = []
    for _ in range(3):
        state, out = rows.step(state, np.zeros((2, 2), np.float32))
        rewards.append(out.reward.tolist())
    rows.close()
    assert rewards == [[1.0, 1.0], [1.0, 1.0], [7.0, 7.0]]
    assert out.ends == ("far",) and out.reason.tolist() == [1, 1]
    assert out.obs.tolist() == [[3.0], [3.0]] and state.obs.tolist() == [[0.0], [0.0]]
