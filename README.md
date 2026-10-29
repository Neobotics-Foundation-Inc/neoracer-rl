# NeoRacer RL

Reinforcement learning for the NeoRacer, a 1/12-scale autonomous racing car simulated by [neoracer-mujoco](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco). This repository turns the simulator into a [Gymnasium](https://gymnasium.farama.org) environment: what a policy sees, what it does, what it is rewarded for, and when an episode ends. It also has a batched version for training, and a small PPO agent.

It adds no physics and no sensors. Every observation comes from neoracer-mujoco's model of the real car's sensors, so a trained policy uses only what the car can measure.

## One episode

`NeoRacerEnv` is one car on one track:

```python
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig

env = NeoRacerEnv(NeoRacerEnvConfig())
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

By default the track is `loop_corridor`, a closed corridor 1 m wide. The car starts on the bottom straight and has to drive around it counterclockwise. Each step holds the action for 1/30 s, the period of the real LiDAR's scan.

The action is two numbers in [-1, 1]:

| Action | +1 means |
|---|---|
| 0: speed | the firmware's max speed, 6 m/s, forward |
| 1: steering | full lock to the left, 0.4405 rad |

The speed goes to the car's firmware, as `rc.drive.set_speed_angle` sends it on the car. The firmware's speed loop sets the throttle every 20 ms to hold that speed. The motor reaches about 3.5 m/s on the battery, so a speed past 0.6 asks for more than the car can go and holds full throttle.

The observation is twenty numbers:

| Observation | |
|---|---|
| 0–7 | distances from the LiDAR at 0°, 45°, …, 315° counterclockwise from the front (m), -1 for no hit or a direction the car blocks |
| 8–10 | acceleration (m/s²) |
| 11–13 | rotation rate (rad/s) |
| 14–17 | orientation quaternion |
| 18 | steering angle (rad) |
| 19 | speed from the motor encoder (m/s) |

The distances are those of the last complete turn of the LiDAR's head, as its driver reports them, so a distance is up to two turns old. On the NeoRacer the car blocks the LiDAR's rear quarter, and the beam straight back reads -1.

The speed is the mean speed of the four wheels, which is what the car's motor encoder measures. When a wheel slips, it differs from the true speed.

## Reward and the end of an episode

Progress is how far around the track the car has gone, as an angle, so one lap is 2π on any track. Each step is rewarded with the progress made during it, in radians. The episode ends the first time one of these happens, with a one-time bonus or penalty:

| Reason | When | Reward |
|---|---|---|
| `lap_complete` | the total progress reaches 2π | +10 |
| `crash` | the car touches a wall | −5 |
| `stuck` | less than 0.05 rad of progress in 4 s | −5 |
| `rollover` | the car tilts past 60° | −10 |

If none of them happens, the episode is truncated after 60 s. A crash and getting stuck cost the same, so a car that tries the first corner and hits the wall still earns more than one that never moves. Nobody has tuned these weights yet.

These are the defaults of the three functions that make the task, in `task.py`. Each takes a `Step`, what the env knows after a step: every sensor reading, the car's own and the ones that only a simulator has, whether the car touched a wall, its progress around the track, and the action.

| Function | Default | Returns |
|---|---|---|
| `observe(step)` | `CarObservation()` | the observation |
| `terminate(step)` | `LapEnds()` | the ends, by name, each true where the episode ends that way |
| `reward(step, ends)` | `ProgressReward()` | the reward |

An episode that meets several ends in one step has the first, in the order that `terminate` gives them, and `reward` sees only that one. The defaults are dataclasses, so their settings print, compare and save with the config: `ProgressReward(crash_penalty=3.0)`. To change the task, pass your own functions:

```python
import numpy as np
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig, ProgressReward


def reward(step, ends):
    """The lap task's reward, less a cost for steering hard."""
    return ProgressReward()(step, ends) - 0.01 * np.abs(step.action[:, 1])


env = NeoRacerEnv(NeoRacerEnvConfig(reward=reward))
```

The functions work on rows: a `Step` holds one row in `NeoRacerEnv` and one per episode in `NeoRacerBatch` (below), so the same function serves both. `step.read(sensor)` gives a sensor's reading in every row, for example `step.read(step.car.truth.pos)` for the car's true position.

## Changing the task

`NeoRacerEnvConfig` holds the whole task: the track, the spawn, the share of the max speed that a speed of 1 asks for, how much to randomize the physics, and the three functions above. To change the track or the physics, make a new config:

```python
from neoracer_mujoco.physics import PhysicsRanges, Range
from neoracer_rl.envs import GeneratedTracks

config = NeoRacerEnvConfig(
    tracks=GeneratedTracks(difficulty=2, count=16),
    physics=PhysicsRanges(wheel_friction=Range(0.5, 1.2), chassis_mass=Range(0.7, 1.5)),
)
env = NeoRacerEnv(config)
```

`GeneratedTracks` is a pool of tracks from neoracer-mujoco's generator, and each reset picks one from the pool. `physics` gives a range for each factor on the car's physics, and each reset draws a value from each range. A policy then can't rely on one tire or one mass.

Progress has to mean the same on both kinds of track. So the env keeps each compiled track together with the layout that progress is measured on:

```python
@dataclass(frozen=True)
class Arena:
    scene: Scene
    """The car on the track, compiled."""
    layout: LoopCorridor | Track
    """What lap_angle measures progress on."""
```

`lap_angle(arena, car, sensordata)` computes a car's progress from the layout:
- **On `loop_corridor`**, it is the car's angle around the center of the loop.
- **On a generated `Track`**, it is the distance along the centerline, scaled so that a lap is 2π.

The reward, the stuck check and the lap check only see that angle, so they work the same on every track.

## Many episodes at once

`NeoRacerBatch` runs `nbatch` episodes of the same config, one per row. It keeps no episode state of its own. `reset` returns a `BatchState`, and `step` takes one and returns the next, along with the step's results:

```python
import numpy as np

from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch

env = NeoRacerBatch(Cpu(nthread=4), nbatch=64, config=config)
rng = np.random.default_rng(0)
state = env.reset(rng)
state, out = env.step(state, rng.uniform(-1, 1, (64, 2)))
out.obs, out.reward, out.done  # (64, 20), (64,), (64,)
env.close()
```

When a row's episode ends, the row starts over in the same step. `out.obs` is the last observation of the old episode, and `state.obs` is the first of the new one. A row keeps the same spawn and physics until `env.respawn(state, rng)` draws new ones. The PPO script calls `respawn` once per rollout.

The batch uses the env's `lap_angle` and the config's `observe`, `terminate` and `reward`. On the `Cpu` backend, a row's first episode steps exactly like `NeoRacerEnv` from the same spawn, bit for bit. The rows' speed loops tick together, so a row that starts over keeps the batch's ticks, while `NeoRacerEnv.reset` starts its own over. `Warp()` runs the rows on a CUDA GPU with MuJoCo Warp, all on one track.

## Racing other cars

A race puts several cars on one track. Each car has a driver: `Learner()`, the policy in training; `TrackCentering()` or `WallFollowing()`, the controllers of `neoracer_mujoco.control`; or `Policy(act)`, a policy of its own, such as a frozen copy of an earlier one. `NeoRacerRace` steps `nrace` races at once, as `NeoRacerBatch` steps episodes:

```python
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerRace
from neoracer_rl.envs import Racer, RaceConfig, Start, TrackCentering

# The policy in training, and a controller 1 m ahead of it.
config = RaceConfig(cars=(Racer(), Racer(TrackCentering(), Start(offset=(1.0, 0.0)))))
env = NeoRacerRace(Cpu(nthread=4), nrace=32, config=config)
state = env.reset(rng)
state, out = env.step(state, rng.uniform(-1, 1, (env.nbatch, env.action_dim)))
out.laps  # (32, 2): how far every car has gone, in laps
```

Each learner observes, ends and earns with `config.env`'s `observe`, `terminate` and `reward`, called with its own `Step`. `step.rivals` holds the other cars: each one's `car`, whose sensors and truth a function can read, its `progress`, how far `ahead` of the learner it is, and whether it touched the learner in the step. A reward for gaining on the first of them:

```python
def reward(step, ends):
    rival = step.rivals[0]
    gain = step.progress.delta - rival.progress.delta
    return ProgressReward()(step, ends) + 0.5 * gain
```

`config.agents` says who the agents are. With `Each()`, the default, every learner is an agent, a row of the arrays, and one policy drives them all, each from its own observation. With `Joint()`, the learners of a race are one agent, which sees their observations side by side, gives their actions side by side, two numbers each, and earns the sum of their rewards.

A racer's `start` is where its car starts and which way it faces. Its `offset` is meters along the track from the start, and across it to the left, and its `heading` is radians to the left of the track's direction there: 0, the default, faces along the track, and `math.pi` faces back along it. So a grid's back row follows a bend behind the start. A pool of generated tracks leaves out a track where a spawn of `config.env.spawn` could put a car against a wall, so the pool of a grid can hold other tracks than the pool of one car with the same seeds.

A race ends for every car when a learner meets an end, or at `max_episode_steps`. The other learners are truncated, and the race starts over from its spawn. The cars that a controller or a `Policy` drives have no ends of their own. `NeoRacerBatch` is the race of one learner, so a race steps its cars as the batch steps its one car.

## Training a policy

`examples/train_ppo.py` is a small PPO baseline in JAX and optax. The policy is a Gaussian over speed and steering, with an MLP for the mean.

```sh
python -m examples.train_ppo --total-steps 400000 --seed 1 --out ppo.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
```

`--random-physics` trains with random physics, `--tracks generated` on generated tracks, and `--backend warp` on a GPU. [EXPERIMENTS.md](EXPERIMENTS.md) has the results. For example, three of five seeds trained on `loop_corridor` alone learn to lap it, seed 1, above, among them, and one of the three laps 18 of 20 episodes on generated tracks it never saw.

`examples/train_race.py` trains a policy to race a car that `TrackCentering` drives, 1 m ahead of it at the start, with the same PPO, and evaluates a policy in such races:

```sh
python -m examples.train_race --total-steps 400000 --seed 1 --out race.npz
python -m examples.train_race --eval race.npz --races 20
```

Touching the other car ends the race as a crash does. In [EXPERIMENTS.md](EXPERIMENTS.md#racing-a-controller), policies that lap alone learn in races to follow the car ahead and lap behind it, but none passes it. `train` takes a `RaceConfig`, and the policy has as many actions as the env's agent, so a `Joint` agent of several learners trains with the same code.

`examples/train_v2.py` trains a policy for another car, MIT's RACECAR Neo V2, on what that car's `racecar_core` library shows a program: a LiDAR scan up to two turns of its head old, and its encoder's speed. [examples/racecar_neo_v2](examples/racecar_neo_v2/README.md) holds the program that drives the real car with it.

## Where the code lives

- `src/neoracer_rl/envs/task.py` has the task: `Step`, the progress of a batch of rows, and the default observation, ends and reward.
- `src/neoracer_rl/envs/neoracer_env.py` has `NeoRacerEnv`, its config, and the tracks with how progress is measured on them.
- `src/neoracer_rl/envs/race.py` has the drivers, the agents and `RaceConfig`, and `src/neoracer_rl/envs/neoracer_batch.py` has `NeoRacerRace`, which steps races, and `NeoRacerBatch`, the race of one learner.
- A new part of the task, such as another reward term, is a function of a `Step` in `task.py`, which both the env and the batch then use. Physics and sensors belong in neoracer-mujoco.
- `examples/` has `train_ppo.py`; `train_race.py`, which trains and evaluates a policy in races; `experiments.py`, which trains and evaluates every run of EXPERIMENTS.md and prints its tables; `random_actions.py`, which checks that the environment works; and the RACECAR Neo V2's policy: `train_v2.py`, `drive_v2.py` and `racecar_neo_v2/`.
- `validation/` has the tests, and `users/<github-name>/` personal experiments.

## Setup

With Python 3.14 and [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m examples.random_actions
```

For a CUDA GPU, add the `warp` extra and JAX's CUDA build: `uv pip install -e ".[dev,warp]" "jax[cuda12]"`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), and [AGENTS.md](AGENTS.md) for coding agents. Cite this repository with the "Cite this repository" button on GitHub. The license is GPL-3.0.
