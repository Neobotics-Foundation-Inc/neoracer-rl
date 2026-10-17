# NeoRacer RL

Reinforcement learning for the NeoRacer RC car: a [Gymnasium](https://gymnasium.farama.org) environment, a batched version of it for CPU or GPU, and a small PPO agent that learns to lap a track. The simulator, with the car, the tracks and the sensors, is [neoracer-mujoco](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco), which this package installs from its GitHub Release.

## Setup

With Python 3.15 and [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.15 && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m examples.random_actions   # a few episodes with random actions
```

The learning code uses [JAX](https://docs.jax.dev) and [optax](https://optax.readthedocs.io), since PyTorch has no wheels for Python 3.15 yet.

## The environment

```python
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig

env = NeoRacerEnv(NeoRacerEnvConfig())
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The car must lap a closed corridor counterclockwise without touching a wall, rolling over or getting stuck. By default the corridor is `loop_corridor`, about 1 m wide. The agent sees only what the real car's sensors measure, so a policy trained here can in principle run on the car. A step applies one action for 12 physics steps of 1/360 s: 1/30 s, the scan period of the car's LiDAR. `NeoRacerEnv` is not registered with `gymnasium.make`; make it directly. `render_mode="human"` opens the MuJoCo viewer.

### Actions

`Box(-1, 1, (2,), float32)`. Each value is clipped to [-1, 1] and multiplied by its actuator's limit.

| Num | Action | Actuator | At ±1 |
|---|---|---|---|
| 0 | throttle, + forward | `motor`, one DC motor driving all four wheels through open differentials | the battery's 11.1 V, times `throttle_scale` |
| 1 | steering, + left | `steer_servo` | full lock, 0.4405 rad |

At full throttle the car tops out at about 3.5 m/s, too fast for the corners at a fixed throttle: a simple LiDAR controller laps at 0.5 but hits the wall at 0.75. A policy has to learn to slow down for them.

### Observations

`Box(low, high, (20,), float32)`, all from the car's sensors, in the car's frame (+x forward, +y left, +z up):

| Num | Observation | Min | Max | Unit |
|---|---|---|---|---|
| 0–7 | LiDAR distance at 0°, 45°, …, 315°, counterclockwise from straight ahead; -1 for no hit | -1 | 50 | m |
| 8–10 | acceleration, gravity included | -1000 | 1000 | m/s² |
| 11–13 | rotation rate | -100 | 100 | rad/s |
| 14–17 | orientation quaternion (w, x, y, z), world frame | -1 | 1 | |
| 18 | steering angle | -2 | 2 | rad |
| 19 | speed, from the motor encoder | -40 | 40 | m/s |

### Rewards

Each step's reward is the angle the car moved around the loop, in radians (one lap is 2π), plus, on the last step, +10 for a lap, −5 for a crash, −5 for getting stuck or −10 for a rollover. A lap returns about 2π + 10 ≈ 16.3. The crash and stuck penalties are equal on purpose: if hitting a wall cost more than standing still, not moving would be a trap the learner has to climb out of. `RewardConfig` holds the weights.

`info` holds `progress_rad`, `lap_frac`, `speed_ms`, `upright_cos`, `contact`, `episode_time_s` and `termination_reason`.

### Episodes

The car starts on the bottom straight, facing forward, moved sideways by up to 0.15 m and turned by up to 0.15 rad (`SpawnRange`), and settles. The episode ends at the first of:

1. rollover: the car tilts more than 60°;
2. crash: any part of the car touches a wall;
3. stuck: less than 0.05 rad of progress over the last 4 s;
4. lap: the progress reaches 2π;
5. truncation: 1800 steps, about 60 s.

`TerminationConfig` holds the thresholds.

### Options

`NeoRacerEnvConfig` takes, besides the rewards, spawn and thresholds above:

- **`throttle_scale`**: the share of the battery voltage at full throttle, default 1.0.
- **`physics`**: ranges for the car's wheel friction, chassis mass, motor torque and steering stiffness, as factors on the car XML's values. `reset()` draws one value of each per episode. By default every factor is 1.0.
- **`tracks`**: `LoopCorridor()`, the default, or `GeneratedTracks(difficulty, count, first_seed)`, a pool of tracks from neoracer-mujoco's generator, of difficulty 0 (gentle and wide) to 3 (tight and narrow). `reset()` draws each episode's track from the pool. On a generated track, progress is the distance along the centerline as an angle, so a lap is 2π on every track.

```python
from neoracer_mujoco.physics import PhysicsRanges, Range

from neoracer_rl.envs import GeneratedTracks, NeoRacerEnv, NeoRacerEnvConfig

env = NeoRacerEnv(
    NeoRacerEnvConfig(
        physics=PhysicsRanges(wheel_friction=Range(0.5, 1.2)),
        tracks=GeneratedTracks(difficulty=1, count=16),
    )
)
```

## Many environments at once

`NeoRacerBatch` runs one episode per row, each with the actions, observations, rewards and ends of `NeoRacerEnv`, from the same `NeoRacerEnvConfig`. A row whose episode ends starts over in the same step. Call `respawn(state, rng)` now and then, for example once per PPO update, to draw new spawns, physics and tracks for the rows.

```python
import numpy as np

from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch

env = NeoRacerBatch(Cpu(nthread=4), nbatch=64)
rng = np.random.default_rng(0)
state = env.reset(rng)
state, out = env.step(state, rng.uniform(-1, 1, (64, 2)))
print(out.obs.shape, out.reward.shape)  # (64, 20) (64,)
env.close()
```

- **`Cpu`** steps the rows with `mujoco.rollout` on `nthread` threads. A row steps exactly like `NeoRacerEnv` from the same spawn. On 4 cores, 64 rows step about 5,000 env steps a second on `loop_corridor` and 2,000 on generated tracks.
- **`Warp()`** steps them with MJX on MuJoCo Warp, on a CUDA GPU (`uv pip install -e ".[dev,warp]" "jax[cuda12]"`). Its rows are close to `NeoRacerEnv`'s but not equal. It steps one track per batch.

## Train a policy

`examples/train_ppo.py` trains a policy with PPO on 8 rows of a `NeoRacerBatch` and prints one line per update. On a GPU, add `--backend warp` and raise `--num-envs`. A seed gives the same run on 1 or 4 threads.

```sh
python -m examples.train_ppo --total-steps 400000 --seed 1 --out ppo.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
python -m examples.train_ppo --eval-baseline random --episodes 20
```

`--random-physics` trains with the ranges of `RANDOM_PHYSICS` in `train_ppo.py`, wide enough to see an effect but not measured on the car. `--tracks generated` trains on a pool of generated tracks (`--pool`, `--difficulty`); `--first-track-seed` picks the pool's seeds, for example to evaluate on tracks the policy never saw.

### Results

We trained each regimen for 400k steps per seed, on a 4-core 2.8 GHz Xeon, where a run takes 7 to 13 minutes. Then we evaluated each policy with its mean action, on env seeds from 10000.

```
regimen                          seeds that lap   first lap           laps in training, last 50k steps
loop_corridor                    0 of 5           -                   -
loop_corridor, random physics    3 of 5           124k, 157k, 276k    87%, 91%, 69%
16 generated tracks              2 of 3           77k, 81k            89%, 91%
```

On `loop_corridor` alone, no seed learns to lap. Each creeps to the first corner, 15% to 22% of the lap, and stops there or touches its wall. Random actions and the zero action do no better: their returns are about -5, against about -4 for these policies. Learning to lap from `loop_corridor` alone is rare, not impossible: on neoracer-mujoco 0.10.0, whose physics differs from 0.11.0's only by rounding, seed 3 learned it after 293k steps, and that policy still laps 19 of 20 times on 0.11.0. So compare changes over several seeds, not one.

Random physics and generated tracks give the learner more variety, and more of their seeds learn to lap. Each policy trained with random physics was evaluated with one factor changed at a time, on 10 episodes each; the table gives its laps out of 10:

```
                 seed  0   1   2   3   4
wheel_friction  0.3    0   0   3   0   1
                0.5   10  10   0   0  10
                0.75  10  10   2   0  10
                1.0   10  10   1   0  10
                1.2   10  10   0   0  10
chassis_mass    0.7   10  10   0   0  10
                1.5   10  10   3   0  10
motor_torque    0.5   10  10   4   0  10
                0.75  10  10   2   0  10
                1.5   10  10   0   0  10
steer_stiffness 0.5   10  10   2   0  10
                2.0   10  10   0   0   9
```

The three that learned lap under every change but the lowest grip. The policies trained on `loop_corridor` lap under none.

The two policies trained on generated tracks that learned lap all 20 episodes on `loop_corridor`, on their 16 training tracks, and on 16 held-out tracks from seed 1000 that they never saw. The policies trained on `loop_corridor` lap none of the generated tracks. Generated tracks are slower to train on: about 560 env steps per second on 4 cores, against 820 on `loop_corridor`.

## Layout

```
src/neoracer_rl/envs/   NeoRacerEnv, NeoRacerBatch, and their reward, progress and termination functions
examples/               training, and a run with random actions
validation/             tests
users/<name>/           personal experiments, not checked by ruff, ty, the tests or CI
```

## Citing, contributing and license

Cite this environment and the simulator with the "Cite this repository" button on GitHub, which reads `CITATION.cff`; `.github/citation.py` writes that file. See [CONTRIBUTING.md](CONTRIBUTING.md), and [AGENTS.md](AGENTS.md) for coding agents. The license is GPL-3.0, as for neoracer-mujoco.
