# NeoRacer RL

Reinforcement learning for the NeoRacer, a 1/12-scale RC car. This repo has a [Gymnasium](https://gymnasium.farama.org) environment for the car and a small PPO agent that learns to drive it around a track. The simulator itself, with the car model, the tracks and the sensors, is [neoracer-mujoco](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco). This package installs it as a dependency.

## Setup

You need Python 3.15. With [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.15
source .venv/bin/activate
uv pip install -e ".[dev]"
```

This also installs neoracer-mujoco from its GitHub Release. That wheel contains the car meshes, so you do not need Git LFS. PyTorch has no wheels for Python 3.15 yet, so the learning code uses [JAX](https://docs.jax.dev) and [optax](https://optax.readthedocs.io).

To check that the environment works, run a few episodes with random actions:

```sh
python -m examples.gymnasium_random_smoke
```

## The environment

`NeoRacerEnv` puts the car on `loop_corridor`, a closed loop of walls about 1 m wide. Each step aims for 1/30 s, the scan rate of the real car's LiDAR. The env rounds that to a whole number of 2 ms physics steps, so a step is 34 ms of simulated time.

The agent sees 20 numbers, all from the car's own sensors: 8 LiDAR distances (m), the IMU's acceleration, rotation rate and orientation, the steering angle and the wheel speed. It chooses 2 numbers between -1 and 1, a throttle and a steering command.

The reward is the angle the car moves around the loop in the forward direction, in radians, so one lap is worth about 6.3. A completed lap adds 10, touching a wall costs 0.1 per step, getting stuck costs 5 and rolling over costs 10. An episode ends when the car completes a lap, makes no progress for about 4 s, rolls over, or reaches 1800 steps (about 61 s).

```python
from neoracer_rl.envs import NeoRacerEnv

env = NeoRacerEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["lap_frac"])  # laps driven so far; negative if the car drove backward
```

## Train a policy

`examples/train_ppo.py` trains a policy with PPO (proximal policy optimization), a common policy-gradient method. It runs 8 copies of the environment and prints one line per update, with how many episodes ended in a lap, got stuck or rolled over.

```sh
python -m examples.train_ppo --total-steps 300000 --seed 0 --out ppo.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
python -m examples.train_ppo --eval-baseline random --episodes 20
```

On a 4-core CPU the script runs about 500 environment steps per second. With seed 0 the first lap shows up after 70k to 90k steps, and after about 200k steps almost every episode is a lap. On 20 evaluation episodes:

```
policy                 return   outcome
ppo seed 0, 1M steps   +16.32   20/20 laps
ppo seed 0, 400k       +16.31   20/20 laps
random actions          -9.08   20/20 stuck
zero action             -5.00   20/20 stuck
```

A lap is worth at most about 16.3, so the trained policy is close to the limit on this track. Two settings matter. The initial action noise has to be large (standard deviation 1.0), or the policy learns to stop before the first corner. The script also scales the throttle by 0.1, because most of the throttle range is too fast for this track.

## Layout

```
src/neoracer_rl/envs/   NeoRacerEnv and its reward, progress and termination functions
examples/               training and smoke-test scripts
validation/             pytest tests
users/<name>/           personal experiments, not checked by ruff, ty, the tests or CI
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Coding agents follow [AGENTS.md](AGENTS.md).

## License

GPL-3.0, the same as neoracer-mujoco. See [LICENSE](LICENSE).
