"""
A batch of NeoRacer environments that step together, on a backend that you
choose:

- Cpu: neoracer_mujoco's CpuBatch, which steps the rows with mujoco.rollout
  on a pool of CPU threads. Optimized for CPU.
- Warp: neoracer_mujoco's WarpBatch, which steps the rows with MJX on MuJoCo
  Warp. Optimized for a CUDA GPU. It needs neoracer-mujoco's jax extra.

Each row is one NeoRacerEnv episode: the same track, actions, observations,
rewards and episode ends (see neoracer_env.py). The env reads all of them
from the sensordata that the backend returns, with the functions that
NeoRacerEnv uses, so a row on the Cpu backend steps like NeoRacerEnv, bit
for bit, from the same spawn.

The env holds no episode state. reset returns a BatchState, and step takes
one and returns the next:

    env = NeoRacerBatch(Cpu(nthread=8), nbatch=64)
    rng = np.random.default_rng(0)
    state = env.reset(rng)
    state, out = env.step(state, actions)  # actions: (64, 2) in [-1, 1]

A row whose episode ends starts over in the same step: out.obs is the last
observation of the episode, and state.obs the first of the next one.

Limit: a row starts over from the spawn and the physics that the last reset
or respawn drew for it, so its episodes share them until the next respawn. NeoRacerEnv
draws a new spawn for each episode. Call respawn every few hundred steps,
for example once per PPO rollout, to see as many spawns.
"""

import copy
from dataclasses import dataclass, replace
from typing import Any, assert_never

import mujoco
import numpy as np
import numpy.typing as npt
from neoracer_mujoco import batch, contract, physics, sim

from .neoracer_env import (
    NeoRacerEnvConfig,
    TerminationReason,
    angle_around_center,
    apply_spawn_offset,
    loop_scene,
    observation,
    reward,
    stuck_window_len,
    substeps_per_action,
)

type Observations = npt.NDArray[np.float32]  # shape (nbatch, OBS_DIM)
type Rows = npt.NDArray[np.float64]  # shape (nbatch,)
type Mask = npt.NDArray[np.bool_]  # shape (nbatch,)

# The episode end of a row as a code: 0 while the episode goes on, else
# REASONS[code - 1].
REASONS: tuple[TerminationReason, ...] = ("rollover", "crash", "stuck", "lap_complete")


@dataclass(frozen=True)
class Cpu:
    """Step on the CPU with mujoco.rollout."""

    nthread: int  # threads; 0 steps every row on the calling thread


@dataclass(frozen=True)
class Warp:
    """Step with MJX on MuJoCo Warp, on a CUDA GPU if JAX sees one."""


type Backend = Cpu | Warp


@dataclass(frozen=True)
class ProgressRows:
    """Progress (see neoracer_env.Progress) of every row."""

    angle: Rows  # rad; angle of the car around the loop center, at the last step
    total: Rows  # rad; cumulative progress in this episode
    delta: Rows  # rad; progress in the last step
    steps: npt.NDArray[np.int64]  # steps in this episode, shape (nbatch,)
    # rad; the totals of the last window_len steps, as a ring: column
    # k % window_len holds the total after step k of the episode.
    window: npt.NDArray[np.float64]  # shape (nbatch, window_len)


def start_rows(angle: Rows, window_len: int) -> ProgressRows:
    """start_progress for every row."""
    zeros = np.zeros_like(angle)
    return ProgressRows(
        angle=angle,
        total=zeros,
        delta=zeros,
        steps=np.zeros(angle.shape, np.int64),
        window=np.zeros((*angle.shape, window_len)),
    )


def advance_rows(p: ProgressRows, angle: Rows) -> ProgressRows:
    """advance for every row."""
    delta = angle - p.angle
    delta = np.where(
        delta > np.pi,
        delta - 2.0 * np.pi,
        np.where(delta < -np.pi, delta + 2.0 * np.pi, delta),
    )
    total = p.total + delta
    steps = p.steps + 1
    window = p.window.copy()
    window[np.arange(len(steps)), steps % window.shape[1]] = total
    return ProgressRows(angle, total, delta, steps, window)


def stuck_rows(p: ProgressRows, min_progress: float) -> Mask:
    """is_stuck for every row."""
    window_len = p.window.shape[1]
    oldest = p.window[np.arange(len(p.steps)), (p.steps + 1) % window_len]
    full = p.steps >= window_len - 1
    return full & (np.abs(p.total - oldest) < min_progress)


def restart_rows(p: ProgressRows, done: Mask, angle: Rows) -> ProgressRows:
    """p, with the rows where done back at the start, at angle (rad)."""
    return ProgressRows(
        angle=np.where(done, angle, p.angle),
        total=np.where(done, 0.0, p.total),
        delta=np.where(done, 0.0, p.delta),
        steps=np.where(done, 0, p.steps),
        window=np.where(done[:, None], 0.0, p.window),
    )


@dataclass(frozen=True)
class BatchState:
    """The state of every row. sim and spawn are the backend's batch state:
    an array of mjSTATE_FULLPHYSICS rows on Cpu, a batched mjx.Data on Warp."""

    sim: Any  # the physics state of each row
    spawn: Any  # the state each row starts over from
    spawn_obs: Observations  # the first observation from spawn
    spawn_angle: Rows  # rad; the car's angle around the loop center at spawn
    obs: Observations  # the observation of the current state
    progress: ProgressRows


@dataclass(frozen=True)
class StepResult:
    """What one step gives, before any row starts over."""

    obs: Observations  # the observation after the step
    reward: Rows
    reason: npt.NDArray[np.int8]  # episode end code, see REASONS
    truncated: Mask  # the episode reached max_episode_steps
    lap_frac: Rows  # laps; total progress / 2 pi, negative if the car drove backward

    @property
    def done(self) -> Mask:
        return (self.reason > 0) | self.truncated


def termination_codes(
    config: NeoRacerEnvConfig,
    upright_cos: Rows,
    contact: Mask,
    stuck: Mask,
    total: Rows,
) -> npt.NDArray[np.int8]:
    """termination for every row, as codes (see REASONS)."""
    cfg = config.termination
    ends = (
        upright_cos < cfg.upright_rollover_cos,
        contact,
        stuck,
        total >= 2.0 * np.pi,
    )
    return np.select(ends, list(range(1, len(REASONS) + 1)), 0).astype(np.int8)


def rewards(
    config: NeoRacerEnvConfig, delta: Rows, codes: npt.NDArray[np.int8]
) -> Rows:
    """reward for every row."""
    terminal = np.array([reward(config.reward, 0.0, r) for r in (None, *REASONS)])
    return config.reward.progress_weight * delta + terminal[codes]


class NeoRacerBatch:
    """nbatch NeoRacerEnv episodes on loop_corridor.xml, stepped together on
    backend. Actions, observations and rewards are numpy arrays on either
    backend. Call close() when done."""

    def __init__(
        self,
        backend: Backend,
        nbatch: int,
        config: NeoRacerEnvConfig = NeoRacerEnvConfig(),
    ) -> None:
        self.config = config
        self.nbatch = nbatch
        self.scene = loop_scene(config, visuals=False)
        model = self.scene.model
        self._sim: batch.Batch[Any, Any, Any]
        match backend:
            case Cpu(nthread=nthread):
                self._sim = batch.CpuBatch(self.scene, nbatch, nthread)
            case Warp():
                # Imported here: the jax extra of neoracer-mujoco is optional.
                from neoracer_mujoco import batch_warp

                self._sim = batch_warp.WarpBatch(self.scene, nbatch)
            case _:
                assert_never(backend)
        self._nstep = substeps_per_action(config, model)
        limits = contract.actuator_limits(model, self.scene.car)
        self._max_torque = config.throttle_scale * limits.torque_nm  # N·m
        self._max_steer = limits.steer_rad  # rad
        self._window_len = stuck_window_len(config)
        # The car at rest on the ground, before the spawn offset.
        data = mujoco.MjData(model)
        sim.apply_ctrl_and_step(model, data, 0.0, sim.SETTLE_STEPS)
        self._settled = data

    def _spawns(self, rng: np.random.Generator) -> tuple[Any, Observations, Rows]:
        """A new spawn for each row, drawn from rng: the backend state, the
        first observation and the angle (rad) around the loop center."""
        model, car = self.scene.model, self.scene.car
        datas = [copy.copy(self._settled) for _ in range(self.nbatch)]
        rows = []
        for data in datas:  # in the order of NeoRacerEnv.reset
            apply_spawn_offset(model, data, car, self.config.spawn, rng)
            rows.append(physics.sample(self.config.physics, rng))
        sensordata = np.stack([d.sensordata for d in datas])
        return (
            self._sim.initial(datas, rows),
            observation(model, car, sensordata),
            angle_around_center(model, car, sensordata),
        )

    def respawn(self, state: BatchState, rng: np.random.Generator) -> BatchState:
        """state with a new spawn drawn from rng for each row. A row starts
        over from it when its episode ends."""
        spawn, spawn_obs, spawn_angle = self._spawns(rng)
        return replace(state, spawn=spawn, spawn_obs=spawn_obs, spawn_angle=spawn_angle)

    def reset(self, rng: np.random.Generator) -> BatchState:
        """Every row at the start of an episode, from a spawn drawn from rng."""
        spawn, spawn_obs, spawn_angle = self._spawns(rng)
        return BatchState(
            sim=spawn,
            spawn=spawn,
            spawn_obs=spawn_obs,
            spawn_angle=spawn_angle,
            obs=spawn_obs,
            progress=start_rows(spawn_angle, self._window_len),
        )

    def step(
        self, state: BatchState, action: npt.ArrayLike
    ) -> tuple[BatchState, StepResult]:
        """Apply action (nbatch, 2), throttle and steer in [-1, 1] as in
        NeoRacerEnv, for one env step. Rows whose episode ends start over
        from their spawn."""
        model, car, cfg = self.scene.model, self.scene.car, self.config
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = batch.command(
            model,
            car,
            action[:, 0].astype(np.float64) * self._max_torque,
            action[:, 1].astype(np.float64) * self._max_steer,
        )
        physics, sensordata = self._sim.step(state.sim, ctrl, self._nstep)
        # On Warp, reduce on the GPU and copy only the last step to the host.
        last = np.asarray(sensordata[:, -1])
        found = batch.read(model, sensordata, car.truth.wall_contact)[..., 0]
        contact = np.asarray((found > 0).any(axis=1))

        progress = advance_rows(state.progress, angle_around_center(model, car, last))
        stuck = stuck_rows(progress, cfg.termination.stuck_progress_rad)
        upright_cos = batch.read(model, last, car.truth.zaxis)[:, 2]
        codes = termination_codes(cfg, upright_cos, contact, stuck, progress.total)
        out = StepResult(
            obs=observation(model, car, last),
            reward=rewards(cfg, progress.delta, codes),
            reason=codes,
            truncated=(codes == 0) & (progress.steps >= cfg.max_episode_steps),
            lap_frac=progress.total / (2.0 * np.pi),
        )

        done = out.done
        if done.any():
            physics = self._sim.reset(physics, done, state.spawn)
        return replace(
            state,
            sim=physics,
            obs=np.where(done[:, None], state.spawn_obs, out.obs),
            progress=restart_rows(progress, done, state.spawn_angle),
        ), out

    def close(self) -> None:
        if isinstance(self._sim, batch.CpuBatch):
            self._sim.close()
