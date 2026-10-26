"""
A batch of NeoRacer environments that step together, on a backend that you
choose:

- Cpu: neoracer_mujoco's CpuBatch, which steps the rows with mujoco.rollout
  on a pool of CPU threads. Optimized for CPU.
- Warp: neoracer_mujoco's WarpBatch, which steps the rows with MJX on MuJoCo
  Warp. Optimized for a CUDA GPU. It needs neoracer-mujoco's jax extra.

Each row is one NeoRacerEnv episode: the same track, actions, observations,
rewards and episode ends (see neoracer_env.py). The env gives each step's
data to the config's observe, terminate and reward (task.py), as NeoRacerEnv
does with one row, so a row on the Cpu backend steps like NeoRacerEnv, bit
for bit, from the same spawn.

Quantities are in SI units unless a name says otherwise.

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

from .neoracer_env import (
    NeoRacerEnvConfig,
    arenas,
    lap_angle,
    progress_window_len,
    substeps_per_action,
)
from .task import (
    Ints,
    Mask,
    Observations,
    ProgressRows,
    Rows,
    Step,
    advance_rows,
    first_end,
    first_step,
    restart_rows,
    start_rows,
)


@dataclass(frozen=True)
class Cpu:
    """Step on the CPU with mujoco.rollout."""

    nthread: int
    """0 steps every row on the calling thread."""


@dataclass(frozen=True)
class Warp:
    """Step with MJX on MuJoCo Warp, on a CUDA GPU if JAX sees one."""


type Backend = Cpu | Warp


@dataclass(frozen=True)
class BatchState:
    """The state of every row. sim and spawn are the backend's batch state:
    an array of mjSTATE_FULLPHYSICS rows on Cpu, a batched mjx.Data on Warp."""

    sim: Any
    """The physics state of each row."""
    spawn: Any
    """The state each row starts over from."""
    spawn_obs: Observations
    """The first observation from spawn."""
    spawn_angle: Rows
    """lap_angle at spawn."""
    spawn_track: Ints
    """The index of each row's track at spawn."""
    obs: Observations
    """The observation of the current state."""
    track: Ints
    """The index of each row's track, in the env's pool."""
    progress: ProgressRows
    loop: speed_loop.SpeedLoopState | None
    """Every row's speed loop, if the car has one."""
    spawn_drive: Rows
    """The drivetrain's angle at spawn."""


@dataclass(frozen=True)
class StepResult:
    """What one step gives, before any row starts over."""

    obs: Observations
    """The observation after the step."""
    reward: Rows
    reason: Ints
    """The end of each row's episode: 0 while it goes on, else k + 1 for
    ends[k]."""
    ends: tuple[str, ...]
    """The names of config.terminate's ends, in its order."""
    truncated: Mask
    """The episode reached max_episode_steps."""
    lap_frac: Rows
    """The total progress / 2 pi in laps, negative if the car drove backward."""

    @property
    def done(self) -> Mask:
        return (self.reason > 0) | self.truncated


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
        self._max_steer = limits.steer_rad
        self._window_len = progress_window_len(config)
        self._settled = [sim.settle(arena.scene.model) for arena in self._arenas]
        """The car at rest on the ground of each track, before the offset."""

    def _observe(self, step: Step) -> Observations:
        return np.asarray(self.config.observe(step), np.float32)

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
        angle = self._lap_angles(tracks, sensordata)
        start = first_step(model, car, sensordata, start_rows(angle, self._window_len))
        return (
            self._sim.initial(datas, scenes, rows),
            self._observe(start),
            angle,
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
        step = Step(model, car, last, contact, progress, action)
        codes, ends = first_end(cfg.terminate(step), self.nbatch)
        out = StepResult(
            obs=self._observe(step),
            reward=cfg.reward(step, ends),
            reason=codes,
            ends=tuple(ends),
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
