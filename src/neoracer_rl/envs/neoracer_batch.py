"""
A batch of NeoRacer races (race.py), or of NeoRacerEnv episodes, that step
together, on a backend that you choose:

- Cpu: neoracer_mujoco's CpuBatch, which steps the rows with mujoco.rollout
  on a pool of CPU threads. Optimized for CPU.
- Warp: neoracer_mujoco's WarpBatch, which steps the rows with MJX on MuJoCo
  Warp. Optimized for a CUDA GPU. It needs neoracer-mujoco's jax extra.

NeoRacerRace steps nrace races, and NeoRacerBatch nbatch NeoRacerEnv
episodes: a race of one Learner each. Each of those is one NeoRacerEnv
episode: the same track, actions, observations, rewards and episode ends
(see neoracer_env.py). The env gives each step's data to the config's
observe, terminate and reward (task.py), as NeoRacerEnv does with one row,
so a row of NeoRacerBatch on the Cpu backend steps like NeoRacerEnv, bit for
bit, from the same spawn.

Quantities are in SI units unless a name says otherwise.

The env holds no episode state. reset returns a BatchState, and step takes
one and returns the next:

    env = NeoRacerBatch(Cpu(nthread=8), nbatch=64)
    rng = np.random.default_rng(0)
    state = env.reset(rng)
    state, out = env.step(state, actions)  # actions: (64, 2) in [-1, 1]

A race whose episode ends starts over in the same step: out.obs is the last
observation of the episode, and state.obs the first of the next one.

Limit: a race starts over from the spawn and the physics that the last reset
or respawn drew for it, so its episodes share them until the next respawn.
NeoRacerEnv draws a new spawn for each episode. Call respawn every few
hundred steps, for example once per PPO rollout, to see as many spawns.

Limit: the cars' speed loops tick together, as batch.drive_cars runs them,
and a race that starts over keeps the batch's ticks, while NeoRacerEnv's
reset starts its ticks over. So a row steps like NeoRacerEnv, bit for bit,
in its first episode, and its later episodes see the ticks at another phase.

Limit: a controller of race.py runs on the host, once per race and step.
"""

import copy
import functools
from dataclasses import dataclass, replace
from typing import Any, assert_never

import mujoco
import numpy as np
import numpy.typing as npt
from neoracer_mujoco import batch, contract, physics, sim, speed_loop
from neoracer_mujoco.control import PDState, track_centering, wall_following

from .neoracer_env import (
    NeoRacerEnvConfig,
    arenas_of,
    lap_angle,
    progress_window_len,
    substeps_per_action,
)
from .race import (
    Each,
    Joint,
    Learner,
    Policy,
    RaceConfig,
    Racer,
    TrackCentering,
    WallFollowing,
)
from .task import (
    Ints,
    Mask,
    Observations,
    ProgressRows,
    Rival,
    Rows,
    SensorData,
    Step,
    advance_rows,
    first_end,
    lead,
    restart_rows,
    start_rows,
)

type Cars = npt.NDArray[np.float64]
"""Shape (nrace, ncars): a value of each car of each race."""


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
class Plan:
    """What the cars that a controller or a Policy drives hold over the next
    step, in every race."""

    action: npt.NDArray[np.float32]
    """Each Policy car's action, shape (nrace, ncars, 2), as a Learner's:
    zeros in the other cars' columns."""
    target: Cars
    """Each controller car's speed (m/s): zeros in the other cars' columns."""
    steer: Cars
    """Each controller car's steering angle (rad), as target."""
    controllers: tuple[tuple[PDState, ...], ...]
    """The state of each car's controller in each race: () for a car that no
    controller drives."""


@dataclass(frozen=True)
class BatchState:
    """The state of every race. sim and spawn are the backend's batch state:
    an array of mjSTATE_FULLPHYSICS rows on Cpu, a batched mjx.Data on Warp.
    obs and spawn_obs have a row per agent (race.Agents)."""

    sim: Any
    """The physics state of each race."""
    spawn: Any
    """The state each race starts over from."""
    spawn_obs: Observations
    """The first observation from spawn."""
    spawn_angle: Cars
    """lap_angle at spawn."""
    spawn_track: Ints
    """The index of each race's track at spawn."""
    spawn_drive: Cars
    """The drivetrain's angle at spawn."""
    spawn_plan: Plan
    """The plan at spawn."""
    obs: Observations
    """The observation of the current state."""
    track: Ints
    """The index of each race's track, in the env's pool."""
    progress: tuple[ProgressRows, ...]
    """Each car's progress, in the order of the race's cars."""
    loop: speed_loop.SpeedLoopState | None
    """Every car's speed loop, of shape (nrace, ncars), if the car has one."""
    plan: Plan


@dataclass(frozen=True)
class StepResult:
    """What one step gives, before any race starts over, a row per agent."""

    obs: Observations
    """The observation after the step."""
    reward: Rows
    reason: Ints
    """The end of each agent's episode: 0 while it goes on, else k + 1 for
    ends[k]. A Joint agent's is its first learner's that met an end."""
    ends: tuple[str, ...]
    """The names of config.terminate's ends, in its order."""
    truncated: Mask
    """The race ended, by another learner's end or at max_episode_steps,
    and the agent met no end of its own."""
    lap_frac: Rows
    """The total progress / 2 pi in laps, negative if the car drove backward:
    a Joint agent's is the mean of its learners'."""
    laps: Cars
    """The total progress / 2 pi of every car of each race."""

    @property
    def done(self) -> Mask:
        return (self.reason > 0) | self.truncated


class NeoRacerRace:
    """nrace races of config, stepped together on backend, a row of the
    env's arrays per agent: nbatch rows, of action_dim numbers in an action
    (race.Agents). Actions, observations and rewards are numpy arrays on
    either backend. Call close() when done. Raise ValueError for a pool of
    generated tracks on Warp, which steps one track per batch."""

    def __init__(
        self, backend: Backend, nrace: int, config: RaceConfig = RaceConfig()
    ) -> None:
        self.config = config
        self.nrace = nrace
        env, cars = config.env, config.cars
        self._learners = list(config.learners)
        self._controlled = [
            k
            for k, racer in enumerate(cars)
            if isinstance(racer.driver, TrackCentering | WallFollowing)
        ]
        match config.agents:
            case Each():
                self.nbatch, self.action_dim = nrace * len(self._learners), 2
            case Joint():
                self.nbatch, self.action_dim = nrace, 2 * len(self._learners)
            case other:
                assert_never(other)
        self._arenas = arenas_of(env, [racer.start for racer in cars], visuals=False)
        self.scene = self._arenas[0].scene
        model = self.scene.model
        self._sim: batch.Batch[Any, Any, Any]
        match backend:
            case Cpu(nthread=nthread):
                self._sim = batch.CpuBatch(self.scene, nrace, nthread)
            case Warp():
                if len(self._arenas) > 1:
                    raise ValueError("Warp steps one track per batch, not a pool")
                # Imported here: the jax extra of neoracer-mujoco is optional.
                from neoracer_mujoco import batch_warp

                self._sim = batch_warp.WarpBatch(self.scene, nrace)
            case _:
                assert_never(backend)
        self._nstep = substeps_per_action(env, model)
        self._dt = self._nstep * float(model.opt.timestep)
        """The control period of a controller: a whole number of physics
        steps."""
        self._max_steer = contract.actuator_limits(model, self.scene.car).steer_rad
        self._window_len = progress_window_len(env)
        self._settled = [sim.settle(arena.scene.model) for arena in self._arenas]
        """The cars at rest on the ground of each track, before the offsets."""
        self._fresh = tuple(
            (PDState(),) * nrace if k in self._controlled else ()
            for k in range(len(cars))
        )
        """Every controller's state at the start of an episode."""

    def _observe(self, step: Step) -> Observations:
        return np.asarray(self.config.env.observe(step), np.float32)

    def _agents(self, rows: Mask) -> Mask:
        """The agent rows of the races where rows, a mask (nrace,)."""
        match self.config.agents:
            case Each():
                return np.repeat(rows, len(self._learners))
            case Joint():
                return rows
            case other:
                assert_never(other)

    def _lap_angles(self, track: Ints, sensordata: SensorData) -> Cars:
        """lap_angle of each car of each race on its own track."""
        angle = np.empty((len(track), len(self.scene.cars)))
        for i, arena in enumerate(self._arenas):
            rows = track == i
            if rows.any():
                for k, car in enumerate(arena.scene.cars):
                    angle[rows, k] = lap_angle(arena, car, sensordata[rows])
        return angle

    def _touched(self, sensordata: Any, sensor: mujoco.MjsSensor) -> Mask:
        """Where the count of sensor, a contact sensor, was above 0 in a
        step of each race of sensordata (nrace, nstep, nsensordata)."""
        # On Warp, reduce on the GPU and copy only the mask to the host.
        found = batch.read(self.scene.model, sensordata, sensor)[..., 0]
        return np.asarray((found > 0).any(axis=1))

    def _steps(
        self,
        sensordata: SensorData,
        walls: npt.NDArray[np.bool_],
        touch: npt.NDArray[np.bool_],
        progress: tuple[ProgressRows, ...],
        actions: npt.NDArray[np.float32],
    ) -> dict[int, Step]:
        """The Step of each car that a Learner or a Policy drives, by the
        car's index. walls (nrace, ncars) holds each car's wall contact, and
        touch (nrace, ncars, ncars) the contacts of each two cars."""
        model, cars = self.scene.model, self.scene.cars
        steps = {}
        for k in range(len(cars)):
            if k in self._controlled:
                continue
            rivals = tuple(
                Rival(
                    cars[j], progress[j], lead(progress[k], progress[j]), touch[:, k, j]
                )
                for j in range(len(cars))
                if j != k
            )
            steps[k] = Step(
                model,
                cars[k],
                sensordata,
                walls[:, k],
                progress[k],
                actions[:, k],
                rivals,
            )
        return steps

    def _plan(
        self,
        sensordata: SensorData,
        steps: dict[int, Step],
        controllers: tuple[tuple[PDState, ...], ...],
    ) -> Plan:
        """What the controllers, from their states controllers, and the
        Policies ask for after a step that left sensordata (nrace,
        nsensordata) and steps."""
        model, cars = self.scene.model, self.scene.cars
        action = np.zeros((self.nrace, len(cars), 2), np.float32)
        target, steer = (
            np.zeros((self.nrace, len(cars))),
            np.zeros((self.nrace, len(cars))),
        )
        states: list[tuple[PDState, ...]] = []
        for k, racer in enumerate(self.config.cars):
            match racer.driver:
                case Learner():
                    control = None
                case Policy(act=act):
                    action[:, k] = np.clip(np.asarray(act(steps[k]), np.float32), -1, 1)
                    control = None
                case TrackCentering(config=c):
                    control = functools.partial(track_centering.step, c)
                case WallFollowing(config=c):
                    control = functools.partial(wall_following.step, c)
                case other:
                    assert_never(other)
            if control is None:
                states.append(())
                continue
            moves = [
                control(state, batch.readings(model, cars[k], sensordata[r]), self._dt)
                for r, state in enumerate(controllers[k])
            ]
            states.append(tuple(state for state, _ in moves))
            target[:, k] = [move.target for _, move in moves]
            steer[:, k] = [move.steer for _, move in moves]
        return Plan(action, target, steer, tuple(states))

    def _spawns(
        self, rng: np.random.Generator
    ) -> tuple[Any, Observations, Cars, Ints, Cars, Plan]:
        """A new spawn, track and physics for each race, drawn from rng in
        the order of NeoRacerEnv.reset, car after car: the backend state, the
        first observation, the lap_angle, the track index, the drivetrain's
        angle and the plan."""
        model, cars, env = self.scene.model, self.scene.cars, self.config.env
        tracks = np.zeros(self.nrace, np.int64)
        datas, scenes, rows = [], [], []
        for i in range(self.nrace):
            if len(self._arenas) > 1:  # one track takes no random draw
                tracks[i] = rng.integers(len(self._arenas))
            scene = self._arenas[tracks[i]].scene
            data = copy.copy(self._settled[tracks[i]])
            for car in scene.cars:
                sim.perturb_spawn(scene.model, data, car, rng, env.spawn)
            datas.append(data)
            scenes.append(scene)
            rows.append(tuple(physics.sample(env.physics, rng) for _ in scene.cars))
        # Every track has the same sizes, so the sensor layout of one fits all.
        sensordata = np.stack([d.sensordata for d in datas])
        angle = self._lap_angles(tracks, sensordata)
        progress = tuple(start_rows(a, self._window_len) for a in columns(angle))
        steps = self._steps(
            sensordata,
            np.zeros((self.nrace, len(cars)), np.bool_),
            np.zeros((self.nrace, len(cars), len(cars)), np.bool_),
            progress,
            np.zeros((self.nrace, len(cars), 2), np.float32),
        )
        obs = [self._observe(steps[k]) for k in self._learners]
        qpos = np.stack([d.qpos for d in datas])
        return (
            self._sim.initial(datas, scenes, rows),
            np.stack(obs, axis=1).reshape(self.nbatch, -1),
            angle,
            tracks,
            np.stack([batch.drive_angle(model, car, qpos) for car in cars], axis=1),
            self._plan(sensordata, steps, self._fresh),
        )

    def respawn(self, state: BatchState, rng: np.random.Generator) -> BatchState:
        """state with a new spawn, track and physics drawn from rng for each
        race. A race starts over from them when its episode ends."""
        spawn, obs, angle, track, drive, plan = self._spawns(rng)
        return replace(
            state,
            spawn=spawn,
            spawn_obs=obs,
            spawn_angle=angle,
            spawn_track=track,
            spawn_drive=drive,
            spawn_plan=plan,
        )

    def reset(self, rng: np.random.Generator) -> BatchState:
        """Every race at the start of an episode, from a spawn, a track and a
        physics drawn from rng."""
        spawn, obs, angle, track, drive, plan = self._spawns(rng)
        params = self.config.env.car.speed_loop
        return BatchState(
            sim=spawn,
            spawn=spawn,
            spawn_obs=obs,
            spawn_angle=angle,
            spawn_track=track,
            spawn_drive=drive,
            spawn_plan=plan,
            obs=obs,
            track=track,
            progress=tuple(start_rows(a, self._window_len) for a in columns(angle)),
            loop=None if params is None else speed_loop.start(params, drive),
            plan=plan,
        )

    def step(
        self, state: BatchState, action: npt.ArrayLike
    ) -> tuple[BatchState, StepResult]:
        """Apply action (nbatch, action_dim): each learner's speed and steer
        in [-1, 1], as in NeoRacerEnv, for one env step. Races whose episode
        ends start over from their spawn."""
        model, cars, env = self.scene.model, self.scene.cars, self.config.env
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        actions = state.plan.action.copy()
        actions[:, self._learners] = action.reshape(self.nrace, len(self._learners), 2)
        speed = actions[..., 0].astype(np.float64)
        steer = actions[..., 1].astype(np.float64) * self._max_steer
        params, loop = env.car.speed_loop, state.loop
        if params is None or loop is None:
            ctrl = batch.command_cars(model, cars, speed * env.max_speed, steer)
            sim_state, sensordata = self._sim.step(state.sim, ctrl, self._nstep)
        else:
            commanded = np.asarray(speed_loop.target(params, speed, env.max_speed))
            commanded[:, self._controlled] = state.plan.target[:, self._controlled]
            steer[:, self._controlled] = state.plan.steer[:, self._controlled]
            sim_state, sensordata, loop = batch.drive_cars(
                self._sim,
                model,
                cars,
                params,
                state.sim,
                loop,
                commanded,
                steer,
                self._nstep,
            )
        last = np.asarray(sensordata[:, -1])
        walls = np.stack(
            [self._touched(sensordata, car.truth.wall_contact) for car in cars], axis=1
        )
        touch = np.zeros((self.nrace, len(cars), len(cars)), np.bool_)
        for (i, j), sensor in self.scene.car_contacts.items():
            touch[:, i, j] = touch[:, j, i] = self._touched(sensordata, sensor)

        angle = self._lap_angles(state.track, last)
        progress = tuple(
            advance_rows(p, a)
            for p, a in zip(state.progress, columns(angle), strict=True)
        )
        steps = self._steps(last, walls, touch, progress, actions)
        learners = [steps[k] for k in self._learners]
        results = [first_end(env.terminate(s), self.nrace) for s in learners]
        names = tuple(results[0][1])
        if any(tuple(ends) != names for _, ends in results):
            raise ValueError("terminate must give every car the same ends, in order")
        codes = [code for code, _ in results]
        rewards = [env.reward(s, ends) for s, (_, ends) in zip(learners, results)]
        done = np.any([code > 0 for code in codes], axis=0) | (
            progress[0].steps >= env.max_episode_steps
        )
        out = self._result(
            [self._observe(s) for s in learners],
            rewards,
            codes,
            names,
            done,
            progress,
        )

        plan = self._plan(last, steps, state.plan.controllers)
        if done.any():
            sim_state = self._sim.reset(sim_state, done, state.spawn)
            if loop is not None:
                loop = speed_loop.restart(loop, done, state.spawn_drive)
            plan = restart_plan(plan, done, state.spawn_plan)
        return replace(
            state,
            sim=sim_state,
            loop=loop,
            obs=np.where(self._agents(done)[:, None], state.spawn_obs, out.obs),
            track=np.where(done, state.spawn_track, state.track),
            progress=tuple(
                restart_rows(p, done, a)
                for p, a in zip(progress, columns(state.spawn_angle), strict=True)
            ),
            plan=plan,
        ), out

    def _result(
        self,
        obs: list[Observations],
        rewards: list[Rows],
        codes: list[Ints],
        ends: tuple[str, ...],
        done: Mask,
        progress: tuple[ProgressRows, ...],
    ) -> StepResult:
        """The StepResult of each agent, from each learner's observation,
        reward and end code, and the races that are done."""
        laps = np.stack([p.total for p in progress], axis=1) / (2.0 * np.pi)
        match self.config.agents:
            case Each():
                reason = np.stack(codes, axis=1).reshape(-1)
                reward = np.stack(rewards, axis=1).reshape(-1)
                lap_frac = laps[:, self._learners].reshape(-1)
            case Joint():
                reason = np.zeros(self.nrace, np.int64)
                for code in codes:
                    reason = np.where(reason == 0, code, reason)
                reward = np.sum(rewards, axis=0)
                lap_frac = laps[:, self._learners].mean(axis=1)
            case other:
                assert_never(other)
        return StepResult(
            obs=np.stack(obs, axis=1).reshape(self.nbatch, -1),
            reward=reward,
            reason=reason,
            ends=ends,
            truncated=self._agents(done) & (reason == 0),
            lap_frac=lap_frac,
            laps=laps,
        )

    def close(self) -> None:
        if isinstance(self._sim, batch.CpuBatch):
            self._sim.close()


def columns(cars: Cars) -> list[Rows]:
    """The value of each car in every race: shape (nrace,) each, in the
    order of the race's cars."""
    return [cars[:, k] for k in range(cars.shape[1])]


def restart_plan(plan: Plan, done: Mask, spawn: Plan) -> Plan:
    """plan, with the races where done at the plan of spawn."""
    return Plan(
        action=np.where(done[:, None, None], spawn.action, plan.action),
        target=np.where(done[:, None], spawn.target, plan.target),
        steer=np.where(done[:, None], spawn.steer, plan.steer),
        controllers=tuple(
            tuple(s if d else p for p, s, d in zip(ps, ss, done, strict=True))
            if ps
            else ()
            for ps, ss in zip(plan.controllers, spawn.controllers, strict=True)
        ),
    )


class NeoRacerBatch(NeoRacerRace):
    """nbatch NeoRacerEnv episodes, stepped together on backend: races of
    one Learner."""

    def __init__(
        self,
        backend: Backend,
        nbatch: int,
        config: NeoRacerEnvConfig = NeoRacerEnvConfig(),
    ) -> None:
        super().__init__(backend, nbatch, RaceConfig(config, (Racer(),)))
