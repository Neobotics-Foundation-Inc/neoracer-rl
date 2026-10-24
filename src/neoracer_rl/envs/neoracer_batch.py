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

Limit: the rows' speed loops tick together, as batch.drive runs them, and a
row that starts over keeps the batch's ticks, while NeoRacerEnv's reset
starts its ticks over. So a row steps like NeoRacerEnv, bit for bit, in its
first episode, and its later episodes see the ticks at another phase.
"""

import copy
from dataclasses import dataclass, replace
from typing import Any, assert_never

import numpy as np
import numpy.typing as npt
from neoracer_mujoco import batch, contract, physics, sim, speed_loop
from neoracer_mujoco.track_generation import shortest_loop_distance

from .neoracer_env import (
    NeoRacerEnvConfig,
    TerminationReason,
    arenas,
    lap_angle,
    observation,
    reward,
    stuck_window_len,
    substeps_per_action,
)

type Observations = npt.NDArray[np.float32]  # shape (nbatch, OBS_DIM)
type Rows = npt.NDArray[np.float64]  # shape (nbatch,)
type Mask = npt.NDArray[np.bool_]  # shape (nbatch,)
type Ints = npt.NDArray[np.int64]  # shape (nbatch,)

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
    delta = shortest_loop_distance(angle - p.angle, 2.0 * np.pi)
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
    spawn_angle: Rows  # rad; lap_angle at spawn
    spawn_track: Ints  # the index of each row's track at spawn
    obs: Observations  # the observation of the current state
    track: Ints  # the index of each row's track, in the env's pool
    progress: ProgressRows
    loop: speed_loop.SpeedLoopState | None  # every row's speed loop, if the car has one
    spawn_drive: Rows  # rad; the drivetrain's angle at spawn


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
    """nbatch NeoRacerEnv episodes, stepped together on backend. Actions,
    observations and rewards are numpy arrays on either backend. Call close()
    when done. Raise ValueError for a pool of generated tracks on Warp, which
    steps one track per batch."""

    def __init__(
        self,
        backend: Backend,
        nbatch: int,
        config: NeoRacerEnvConfig = NeoRacerEnvConfig(),
    ) -> None:
        self.config = config
        self.nbatch = nbatch
        self._arenas = arenas(config, visuals=False)
        self.scene = self._arenas[0].scene
        model = self.scene.model
        self._sim: batch.Batch[Any, Any, Any]
        match backend:
            case Cpu(nthread=nthread):
                self._sim = batch.CpuBatch(self.scene, nbatch, nthread)
            case Warp():
                if len(self._arenas) > 1:
                    raise ValueError("Warp steps one track per batch, not a pool")
                # Imported here: the jax extra of neoracer-mujoco is optional.
                from neoracer_mujoco import batch_warp

                self._sim = batch_warp.WarpBatch(self.scene, nbatch)
            case _:
                assert_never(backend)
        self._nstep = substeps_per_action(config, model)
        limits = contract.actuator_limits(model, self.scene.car)
        self._max_steer = limits.steer_rad  # rad
        self._window_len = stuck_window_len(config)
        # The car at rest on the ground of each track, before the offset.
        self._settled = []
        for arena in self._arenas:
            self._settled.append(sim.settle(arena.scene.model))

    def _lap_angles(self, track: Ints, sensordata: npt.NDArray) -> Rows:
        """lap_angle of each row on its own track, shape (nbatch,)."""
        angle = np.empty(len(track))
        for i, arena in enumerate(self._arenas):
            rows = track == i
            if rows.any():
                angle[rows] = lap_angle(arena, sensordata[rows])
        return angle

    def _spawns(
        self, rng: np.random.Generator
    ) -> tuple[Any, Observations, Rows, Ints, Rows]:
        """A new spawn, track and physics for each row, drawn from rng in the
        order of NeoRacerEnv.reset: the backend state, the first observation,
        the lap_angle, the track index and the drivetrain's angle of each
        row."""
        model, car = self.scene.model, self.scene.car
        tracks = np.zeros(self.nbatch, np.int64)
        datas, scenes, rows = [], [], []
        for i in range(self.nbatch):
            if len(self._arenas) > 1:  # one track takes no random draw
                tracks[i] = rng.integers(len(self._arenas))
            scene = self._arenas[tracks[i]].scene
            data = copy.copy(self._settled[tracks[i]])
            sim.perturb_spawn(scene.model, data, scene.car, rng, self.config.spawn)
            datas.append(data)
            scenes.append(scene)
            rows.append(physics.sample(self.config.physics, rng))
        # Every track has the same sizes, so the sensor layout of one fits all.
        sensordata = np.stack([d.sensordata for d in datas])
        return (
            self._sim.initial(datas, scenes, rows),
            observation(model, car, sensordata),
            self._lap_angles(tracks, sensordata),
            tracks,
            batch.drive_angle(model, car, np.stack([d.qpos for d in datas])),
        )

    def respawn(self, state: BatchState, rng: np.random.Generator) -> BatchState:
        """state with a new spawn, track and physics drawn from rng for each
        row. A row starts over from them when its episode ends."""
        spawn, spawn_obs, spawn_angle, spawn_track, spawn_drive = self._spawns(rng)
        return replace(
            state,
            spawn=spawn,
            spawn_obs=spawn_obs,
            spawn_angle=spawn_angle,
            spawn_track=spawn_track,
            spawn_drive=spawn_drive,
        )

    def reset(self, rng: np.random.Generator) -> BatchState:
        """Every row at the start of an episode, from a spawn, a track and a
        physics drawn from rng."""
        spawn, spawn_obs, spawn_angle, spawn_track, spawn_drive = self._spawns(rng)
        params = self.config.car.speed_loop
        return BatchState(
            sim=spawn,
            spawn=spawn,
            spawn_obs=spawn_obs,
            spawn_angle=spawn_angle,
            spawn_track=spawn_track,
            obs=spawn_obs,
            track=spawn_track,
            progress=start_rows(spawn_angle, self._window_len),
            loop=None if params is None else speed_loop.start(params, spawn_drive),
            spawn_drive=spawn_drive,
        )

    def step(
        self, state: BatchState, action: npt.ArrayLike
    ) -> tuple[BatchState, StepResult]:
        """Apply action (nbatch, 2), speed and steer in [-1, 1] as in
        NeoRacerEnv, for one env step. Rows whose episode ends start over
        from their spawn."""
        model, car, cfg = self.scene.model, self.scene.car, self.config
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        speed = action[:, 0].astype(np.float64)
        steer = action[:, 1].astype(np.float64) * self._max_steer
        params, loop = cfg.car.speed_loop, state.loop
        if params is None or loop is None:
            ctrl = batch.command(model, car, speed * cfg.max_speed, steer)
            physics, sensordata = self._sim.step(state.sim, ctrl, self._nstep)
        else:
            commanded = np.asarray(speed_loop.target(params, speed, cfg.max_speed))
            physics, sensordata, loop = batch.drive(
                self._sim,
                model,
                car,
                params,
                state.sim,
                loop,
                commanded,
                steer,
                self._nstep,
            )
        # On Warp, reduce on the GPU and copy only the last step to the host.
        last = np.asarray(sensordata[:, -1])
        found = batch.read(model, sensordata, car.truth.wall_contact)[..., 0]
        contact = np.asarray((found > 0).any(axis=1))

        progress = advance_rows(state.progress, self._lap_angles(state.track, last))
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
            if loop is not None:
                loop = speed_loop.restart(loop, done, state.spawn_drive)
        return replace(
            state,
            sim=physics,
            loop=loop,
            obs=np.where(done[:, None], state.spawn_obs, out.obs),
            track=np.where(done, state.spawn_track, state.track),
            progress=restart_rows(progress, done, state.spawn_angle),
        ), out

    def close(self) -> None:
        if isinstance(self._sim, batch.CpuBatch):
            self._sim.close()
