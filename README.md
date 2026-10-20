# NeoRacer RL

Reinforcement learning for the NeoRacer RC car: a [Gymnasium](https://gymnasium.farama.org) environment, a batched version of it for CPU or GPU, and a small PPO agent that learns to lap a track. The simulator is [neoracer-mujoco](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco).

## Setup

With Python 3.14 and [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m examples.random_actions
```

## The environment

```python
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig

env = NeoRacerEnv(NeoRacerEnvConfig())
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The car must lap a corridor counterclockwise without touching a wall, rolling over or getting stuck. It sees only what the real car's sensors measure. A step is 1/30 s.

| Action | ±1 means |
|---|---|
| 0: throttle, + forward | the battery's full 11.1 V |
| 1: steering, + left | full lock, 0.4405 rad |

| Observation | |
|---|---|
| 0–7 | LiDAR distances at 0°, 45°, …, 315° (m), -1 for no hit |
| 8–10 | acceleration (m/s²) |
| 11–13 | rotation rate (rad/s) |
| 14–17 | orientation quaternion |
| 18 | steering angle (rad) |
| 19 | speed from the motor encoder (m/s) |

The reward is the progress around the loop in radians, plus +10 for a lap or a penalty for a crash, getting stuck or a rollover. An episode ends at the first of those, or after 60 s.

`NeoRacerEnvConfig` sets the rewards, the spawn, the end conditions, a range of physics to randomize per episode (`physics`), and the track: `LoopCorridor()` or a pool of `GeneratedTracks(difficulty, count)`.

## Many environments at once

`NeoRacerBatch` runs many episodes at once, one per row, with the same config:

```python
import numpy as np

from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch

env = NeoRacerBatch(Cpu(nthread=4), nbatch=64)
rng = np.random.default_rng(0)
state = env.reset(rng)
state, out = env.step(state, rng.uniform(-1, 1, (64, 2)))
env.close()
```

`Cpu` uses every core, and a row steps exactly like `NeoRacerEnv`. `Warp()` runs on a CUDA GPU (`uv pip install -e ".[dev,warp]" "jax[cuda12]"`).

## Train a policy

```sh
python -m examples.train_ppo --total-steps 400000 --seed 1 --out ppo.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
```

`--random-physics` trains with random physics, and `--tracks generated` on generated tracks. [EXPERIMENTS.md](EXPERIMENTS.md) has what each of these learns.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), and [AGENTS.md](AGENTS.md) for coding agents. Cite this repository with the "Cite this repository" button on GitHub. The license is GPL-3.0.
