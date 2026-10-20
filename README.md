# NeoRacer RL

[neoracer-mujoco](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco) simulates the NeoRacer RC car: you put the car on a track and step it in your own loop. Learning to drive by reinforcement learning needs more than a step loop. A policy needs three things:
- a fixed list of what it observes;
- a reward for getting somewhere;
- a rule for when an episode is over.

They must mean the same whether one car runs so that you can watch it, or a thousand run so that a policy can train.

This repository defines them once, as a [Gymnasium](https://gymnasium.farama.org) environment and a batched version of it, and trains a small PPO agent on them. It adds no physics and no sensors. Every observation comes from neoracer-mujoco's model of the real car's sensors, so a trained policy uses only what the car can measure.

The sections below follow the order in which the pieces build on one another: one episode, the reward and the end of an episode, a change of task, many episodes at once, and training. After them come the layout of the repository and the setup.

## One episode

`NeoRacerEnv` is one car on one track, behind the Gymnasium API:

```python
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig

env = NeoRacerEnv(NeoRacerEnvConfig())
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

By default the track is `loop_corridor`, a closed corridor 1 m wide. The car starts on its bottom straight, a little off the centerline and turned a little, and has to drive around counterclockwise. Each step holds the action for 1/30 s, the period of the real LiDAR's scan.

The action is two numbers in [-1, 1]:

| Action | +1 means |
|---|---|
| 0: throttle | the battery's full 11.1 V, forward |
| 1: steering | full lock to the left, 0.4405 rad |

The observation is twenty numbers, all from sensors the real car has:

| Observation | |
|---|---|
| 0–7 | distances from eight fixed rangefinders at 0°, 45°, …, 315° counterclockwise from the front (m), -1 for no hit |
| 8–10 | acceleration (m/s²) |
| 11–13 | rotation rate (rad/s) |
| 14–17 | orientation quaternion |
| 18 | steering angle (rad) |
| 19 | speed from the motor encoder (m/s) |

The speed is the mean speed of the four wheels, which is what the car's one motor encoder measures. It differs from the true speed when a wheel slips.

## Progress, reward and the end of an episode

Progress is how far around the track the car has gone, measured as an angle so that one lap is 2π on any track. The reward of a step is the progress the car made in it, in radians. The episode ends at the first of these four, with a one-time bonus or penalty:

| Reason | When | Reward |
|---|---|---|
| `lap_complete` | the total progress reaches 2π | +10 |
| `crash` | the car touches a wall | −5 |
| `stuck` | less than 0.05 rad of progress in 4 s | −5 |
| `rollover` | the car tilts past 60° | −10 |

Otherwise it is truncated after 60 s. A crash and getting stuck cost the same. As a result, a car that tries the first corner and hits the wall earns more than one that never moves.

The reward is a pure function of the step's progress and the reason the episode ended, if it did:

```python
from neoracer_rl.envs import RewardConfig, reward

reward(RewardConfig(), 0.1, None)            # 0.1
reward(RewardConfig(), 0.1, "lap_complete")  # 10.1
reward(RewardConfig(), 0.0, "crash")         # -5.0
```

The reason is a `TerminationReason`, `Literal["rollover", "crash", "stuck", "lap_complete"]`, so the type checker sees every case. Progress works the same way. `advance` takes the last `Progress` and the car's new angle and returns the next `Progress`, and `termination` returns the reason or `None`. Nobody has tuned the weights yet.

## Changing the task

`NeoRacerEnvConfig` is a frozen dataclass that holds the whole task: the track, the rewards, the spawn, the end conditions, how much of the battery voltage full throttle gives, and the ranges for randomizing the physics. A new task is a new config:

```python
from neoracer_mujoco.physics import PhysicsRanges, Range
from neoracer_rl.envs import GeneratedTracks

config = NeoRacerEnvConfig(
    tracks=GeneratedTracks(difficulty=2, count=16),
    physics=PhysicsRanges(wheel_friction=Range(0.5, 1.2), chassis_mass=Range(0.7, 1.5)),
)
env = NeoRacerEnv(config)
```

`GeneratedTracks` is a pool of tracks from neoracer-mujoco's generator, and each reset draws the episode's track from the pool. `physics` gives the range of each factor on the car's physics. Each reset draws a factor from each range, so a policy cannot rely on one tyre or one mass.

Progress has to mean the same on both kinds of track. The env keeps each compiled track next to the layout that progress is measured on:

```python
@dataclass(frozen=True)
class Arena:
    scene: Scene                  # the car on the track, compiled
    layout: LoopCorridor | Track  # what progress is measured on
```

`lap_angle(arena, sensordata)` matches on the layout:
- **On `loop_corridor`** it is the car's angle around the loop's center.
- **On a generated `Track`** it is the distance along the centerline (neoracer-mujoco's `lap_distance`), scaled so that a lap is 2π.

The reward, the stuck check and the lap check see only that angle, so they work unchanged on every track.

## Many episodes at once

Training needs many episodes at once. `NeoRacerBatch` runs `nbatch` episodes of one config, one per row. It holds no episode state: `reset` returns a `BatchState`, and `step` takes one and returns the next, with what the step gave:

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

A row whose episode ends starts over in the same step. `out.obs` is the last observation of the old episode, and `state.obs` the first of the new one. Until `env.respawn(state, rng)` draws new ones, a row starts over from the same spawn and physics. The PPO script calls `respawn` once per rollout.

The batch reuses the env's parts instead of copying them:
- the same arenas;
- `observation` and `lap_angle` applied to the backend's sensor arrays;
- the reward of each reason.

Its progress functions are the env's, written for arrays. On the `Cpu` backend, which runs `mujoco.rollout` on every core, a row steps the same as `NeoRacerEnv`, bit for bit, from the same spawn. `Warp()` runs the rows with MuJoCo Warp on a CUDA GPU, all on one track.

## Training a policy

`examples/train_ppo.py` is a small PPO in JAX and optax, a baseline rather than a tuned agent. The policy is a Gaussian over throttle and steering, its mean given by an MLP. The update is a pure core of the MLP, `gae`, the clipped loss and the update step. Everything that steps environments or writes files sits apart from it.

```sh
python -m examples.train_ppo --total-steps 400000 --seed 1 --out ppo.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
```

`--random-physics` trains with random physics, `--tracks generated` on generated tracks, and `--backend warp` on a GPU. [EXPERIMENTS.md](EXPERIMENTS.md) has what each one learns. For example, every seed trained on 16 generated tracks laps tracks it never saw, while most seeds trained on `loop_corridor` alone never get past its first corner.

## Where the code lives

- `src/neoracer_rl/envs/neoracer_env.py` holds `NeoRacerEnv`, plus the pure functions that define the task: the observation, progress, termination and reward.
- `src/neoracer_rl/envs/neoracer_batch.py` holds `NeoRacerBatch` and the array versions of those functions.
- A new part of the task, such as another reward term, goes in `neoracer_env.py` as a pure function, so that the env and the batch both use it. Physics and sensors go in neoracer-mujoco.
- `examples/` holds `train_ppo.py`, and `random_actions.py`, which runs episodes of random actions to check that the environment works.
- `validation/` holds the tests, and `users/<github-name>/` personal experiments.

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
