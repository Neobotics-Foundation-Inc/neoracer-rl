"""
A race: several cars on one track, each driven by the policy in training, by
a controller of neoracer_mujoco.control, or by a policy of its own.
neoracer_batch.NeoRacerRace steps a batch of races:

    config = RaceConfig(cars=(Racer(), Racer(TrackCentering(), (1.0, 0.0))))
    env = NeoRacerRace(Cpu(nthread=8), nrace=32, config=config)
    state = env.reset(rng)
    state, out = env.step(state, actions)  # (env.nbatch, env.action_dim)

Every car is a car of config.env.car_path, attached behind a prefix of its
own (assets.compose_cars) and started at its Racer's offset from the track's
start. The cars meet only in the physics: each car's LiDAR sees the others,
and they collide.

A Learner is a car of the policy in training. The env takes its action, as
NeoRacerEnv takes one, and calls config.env's observe, terminate and reward
(task.py) with its Step. The Step's rivals are the other cars, so a function
can read their sensors, how far ahead they are, and their contacts with the
car. config.agents groups the learners into agents:

- Each: every learner is an agent of its own, a row of the env's arrays.
  One policy drives them all, each from its own observation (decentralized).
- Joint: the learners of a race are one agent: its observation is theirs side
  by side, its action theirs side by side, 2 numbers each, and its reward
  the sum of theirs (centralized).

A race ends, for every car, when a learner meets an end of terminate, or at
max_episode_steps. The learners that met no end are truncated, and the race
starts over from its spawn. The other cars have no ends of their own: after
a crash they drive on.

The other drivers plan at the end of each step, from what it left, the
command that their cars hold over the next one: a controller from its car's
SensorReadings (batch.readings), with a PDState of its own in each race, and
a Policy from its car's Step. A controller asks for a speed, which the car's
speed loop holds, so a car without one takes only Learners and Policies.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from neoracer_mujoco.control.track_centering import TrackCenteringConfig
from neoracer_mujoco.control.wall_following import WallFollowingConfig

from .neoracer_env import NeoRacerEnvConfig
from .task import Step


@dataclass(frozen=True)
class Learner:
    """The policy in training drives the car."""


@dataclass(frozen=True)
class TrackCentering:
    """neoracer_mujoco.control.track_centering drives the car."""

    config: TrackCenteringConfig = TrackCenteringConfig()


@dataclass(frozen=True)
class WallFollowing:
    """neoracer_mujoco.control.wall_following drives the car."""

    config: WallFollowingConfig = WallFollowingConfig()


@dataclass(frozen=True)
class Policy:
    """A policy of its own drives the car: a trained one, for example."""

    act: Callable[[Step], npt.NDArray[np.float32]]
    """The car's action in each race, shape (nrace, 2), as a Learner's, from
    the car's Step."""


type Driver = Learner | TrackCentering | WallFollowing | Policy


@dataclass(frozen=True)
class Racer:
    """A car of a race."""

    driver: Driver = Learner()
    offset: tuple[float, float] = (0.0, 0.0)
    """Where the car starts (m), from the track's start, in its frame: +X
    forward, +Y left (assets.Entry)."""


@dataclass(frozen=True)
class Each:
    """Every learner is an agent: learner k of race r is row r * learners + k
    of the env's arrays."""


@dataclass(frozen=True)
class Joint:
    """The learners of a race are one agent: race r is row r of the env's
    arrays."""


type Agents = Each | Joint


@dataclass(frozen=True)
class RaceConfig:
    """The cars of a race, in order, and the agents that their learners
    make. config.env sets the car, the tracks, the control rate and the task
    of every learner."""

    env: NeoRacerEnvConfig = NeoRacerEnvConfig()
    cars: tuple[Racer, ...] = (Racer(), Racer(TrackCentering(), (1.0, 0.0)))
    agents: Agents = Each()

    def __post_init__(self) -> None:
        drivers = [racer.driver for racer in self.cars]
        if not any(isinstance(d, Learner) for d in drivers):
            raise ValueError("a race needs a Learner")
        controlled = any(isinstance(d, TrackCentering | WallFollowing) for d in drivers)
        if controlled and self.env.car.speed_loop is None:
            raise ValueError(
                "a controller asks for a speed, and the car has no speed loop to hold it"
            )

    @property
    def learners(self) -> tuple[int, ...]:
        """The indices of the cars that a Learner drives."""
        return tuple(
            k for k, racer in enumerate(self.cars) if isinstance(racer.driver, Learner)
        )
