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
python -m examples.random_actions
```

## The environment

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (2,), float32)` |
| Observation Space | `Box(low, high, (20,), float32)`, with the bounds in the table below |
| import | `from neoracer_rl.envs import NeoRacerEnv` |

### Description

The car drives on `loop_corridor`, a closed corridor about 1 m wide, or on generated tracks (see `Tracks`). The goal is to complete one lap counterclockwise (seen from above) without touching a wall, rolling over or getting stuck. The agent sees only what the real car's sensors measure, and nothing from the simulator's internal state, so a policy trained here can in principle run on the car.

Each step applies one action for 12 physics steps of 1/360 s, so a step is 1/30 s of simulated time, the scan rate of the car's LiDAR.

### Action Space

The action space is a `Box(-1, 1, (2,), float32)`. The env clips each value to [-1, 1] and multiplies it by the limit of its actuator. The throttle is also multiplied by `throttle_scale` (default 1.0, the full battery voltage). At the full voltage the car reaches about 3.5 m/s. That is too fast for the corners at a fixed throttle (a simple LiDAR controller laps at 0.5 but hits the wall at 0.75), but PPO can learn to slow down for them: training seed 3 of the baseline below laps with `throttle_scale` 1.0.

| Num | Action | Control Min | Control Max | Actuator (in neoracer.xml) | Type (Unit) |
|---|---|---|---|---|---|
| 0 | Throttle, + drives forward | -1 | 1 | `motor`, one DC motor driving all four wheels through open differentials | voltage, up to 7.4 (V) |
| 1 | Steering, + turns left | -1 | 1 | `steer_servo` | steering angle, up to 0.4 (rad) |

### Observation Space

The observation space is a `Box` of 20 float32 values, all read from the car's sensors:

| Num | Observation | Min | Max | Sensor | Type (Unit) |
|---|---|---|---|---|---|
| 0–7 | LiDAR distance at 0°, 45°, …, 315° (0° is straight ahead, angles go counterclockwise); -1 if the beam hits nothing | -1 | 50 | `lidar_000` … `lidar_315` | distance (m) |
| 8–10 | Acceleration along the car's x, y, z axes, including gravity | -1000 | 1000 | IMU accelerometer | acceleration (m/s²) |
| 11–13 | Rotation rate about the car's x, y, z axes | -100 | 100 | IMU gyroscope | angular velocity (rad/s) |
| 14–17 | Orientation of the car as a quaternion (w, x, y, z) in the world frame | -1 | 1 | IMU | unitless |
| 18 | Steering angle | -2 | 2 | `steer_cmd_pos` | angle (rad) |
| 19 | Speed, estimated from the wheel speed | -40 | 40 | motor encoder | velocity (m/s) |

The car's frame is +x forward, +y left and +z up.

### Rewards

The reward for each step is: *reward* = *progress* + *end_bonus*.

- *progress*: the angle, in radians, that the car moved around the center of the loop during the step, positive in the forward direction. One lap is worth 2π, about 6.3. It is multiplied by `progress_weight` (default 1).
- *end_bonus*: on the last step only, +`lap_bonus` (default 10) for a completed lap, −`crash_penalty` (default 5) if the car hit a wall, −`stuck_penalty` (default 5) if it got stuck, −`rollover_penalty` (default 10) if it rolled over.

The crash and stuck penalties are equal, so a car that drives toward the first corner and hits the wall returns more than a car that never moves. If hitting the wall cost more than standing still, "do not move" would be a trap that a learner has to climb out of.

A lap therefore returns about 2π + 10 ≈ 16.3, slightly more if the last step goes past 2π. `info` holds `progress_rad` and `lap_frac` (the progress so far, in radians and in laps; negative if the car drove backward), `speed_ms`, `upright_cos`, `contact`, `episode_time_s` and `termination_reason`.

### Starting State

The car starts on the bottom straight at (0, -2.5), facing the forward direction of the loop, and settles for 400 physics steps. `reset()` then moves it sideways by a value from 𝒰(-0.15, 0.15) m, turns it by a value from 𝒰(-0.15, 0.15) rad, and settles it for 50 more physics steps.

`reset()` then draws the car's physics for the episode from `config.physics` (see `Physics randomization`). By default it changes nothing.

### Physics randomization

Nobody knows the real car's grip, mass or motor strength exactly. To train a policy that copes with that, `NeoRacerEnvConfig.physics` gives a range for each of four factors on the values of the car XML. `reset()` draws one value of each from its range, and the episode runs with them:

| Factor | Changes |
|---|---|
| `wheel_friction` | the friction of the wheel contacts, and so how much the wheels slip |
| `chassis_mass` | the mass and inertia of the car body |
| `motor_torque` | the motor's torque at every speed, for an action |
| `steer_stiffness` | the stiffness of the steering servo |

```python
from neoracer_mujoco.physics import PhysicsRanges, Range

from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig

ranges = PhysicsRanges(wheel_friction=Range(0.5, 1.2), motor_torque=Range(0.7, 1.3))
env = NeoRacerEnv(NeoRacerEnvConfig(physics=ranges))
obs, info = env.reset(seed=0)
print(env.physics)  # the factors of this episode
```

The car settles at its spawn with the physics of the XML, so with another mass its suspension settles again in the first steps of the episode. A range of one value takes no random draw, so the default ranges leave every episode as it was. `NeoRacerBatch` draws the physics of each row at each `respawn`, in the same order as `reset()`, so a row on the CPU still steps like `NeoRacerEnv`, bit for bit.

### Tracks

`NeoRacerEnvConfig.tracks` picks where the car drives. `LoopCorridor()`, the default, is `loop_corridor`. `GeneratedTracks(difficulty, count, first_seed)` is a pool of tracks from neoracer-mujoco's track generator: the first `count` seeds from `first_seed` on that give a track, of difficulty 0 (gentle and wide) to 3 (tight and narrow). `reset()` draws the track of each episode from the pool, before the spawn offset and the physics:

```python
from neoracer_rl.envs import GeneratedTracks, NeoRacerEnv, NeoRacerEnvConfig

env = NeoRacerEnv(NeoRacerEnvConfig(tracks=GeneratedTracks(difficulty=1, count=16)))
obs, info = env.reset(seed=0)
```

On a generated track the car starts at the beginning of the centerline, facing along it, and progress is the distance along the centerline, as an angle: one lap is 2π on every track, so the rewards and the episode ends are the same as on `loop_corridor`. The pool is generated when the env is made, about 0.2 s per track.

`NeoRacerBatch` draws the track of each row at each `respawn`, so one batch steps on many tracks. Every track has 192 boxes per wall, so all of them have the same sizes, as `CpuBatch` needs. MuJoCo Warp collides against the walls of the model it was made from, so `Warp()` steps one track per batch and refuses a pool.

A generated track has hundreds of wall boxes, which the collision check and the eight LiDAR rays test at every physics step. Training on them is slower: on 4 cores, 64 rows step about 2,000 env steps per second on generated tracks against 5,000 on `loop_corridor`.

### Episode End

#### Termination

The episode terminates when one of these happens, checked in this order:

1. Rollover: the car tilts more than 60° (the cosine of its tilt drops below 0.5).
2. Crash: any part of the car touches a wall.
3. Stuck: over the last 120 steps (about 4 s), the car's net progress around the loop was less than 0.05 rad in either direction. A car standing still counts.
4. Lap: the car's total progress reaches 2π.

#### Truncation

The episode is truncated after 1800 steps (61.2 s of simulated time).

### Arguments

`NeoRacerEnv` is not registered with `gymnasium.make` yet. Create it directly and pass a `NeoRacerEnvConfig`:

```python
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig, RewardConfig

env = NeoRacerEnv(NeoRacerEnvConfig(reward=RewardConfig(lap_bonus=20.0)))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `car_path` | **str** | neoracer-mujoco's `neoracer.xml` | Path to the car model |
| `tracks` | **LoopCorridor \| GeneratedTracks** | `LoopCorridor()` | Where the car drives (see `Tracks`) |
| `control_hz` | **float** | `30.0` | Target action rate, rounded to whole physics steps (see `Description`) |
| `throttle_scale` | **float** | `1.0` | Fraction of the battery voltage that `action[0] = 1` gives the motor (see `Action Space`) |
| `max_episode_steps` | **int** | `1800` | Steps before truncation (see `Episode End`) |
| `spawn` | **SpawnRange** | `lateral_m=0.15, yaw_rad=0.15, resettle_steps=50` | Random start offsets (see `Starting State`) |
| `termination` | **TerminationConfig** | `upright_rollover_cos=0.5, stuck_window_s=4.0, stuck_progress_rad=0.05` | Rollover and stuck thresholds (see `Episode End`) |
| `reward` | **RewardConfig** | `progress_weight=1.0, rollover_penalty=10.0, crash_penalty=5.0, stuck_penalty=5.0, lap_bonus=10.0` | Reward weights (see `Rewards`) |
| `obs_bounds` | **ObservationBounds** | see the table in `Observation Space` | Bounds of the observation space |
| `physics` | **PhysicsRanges** | every factor 1.0 | Ranges of the car's physics (see `Physics randomization`) |

`render_mode="human"` is a separate argument of `NeoRacerEnv`. It opens the MuJoCo viewer.

## Many environments at once

`NeoRacerBatch` runs many episodes of the same environment together, one per row. Each row has the actions, observations, rewards and episode ends of `NeoRacerEnv`, and takes the same `NeoRacerEnvConfig`. It has no Gymnasium API: `reset` returns a state, and `step` takes a state and one action per row and returns the next state and what the step gave. A row whose episode ends starts over in the same step, so `out.obs` is the last observation of that episode and `state.obs` the first of the next.

A row starts over from the spawn that the last `reset` or `respawn` drew for it. Call `env.respawn(state, rng)` now and then, for example once per PPO update, so that the rows see as many spawns as `NeoRacerEnv` does.

You choose where the physics runs.

### Optimized for CPU

`Cpu` steps the rows with `mujoco.rollout` on `nthread` threads; one per core uses the whole CPU. A row then steps exactly like `NeoRacerEnv` from the same spawn.

```python
import numpy as np

from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerBatch

env = NeoRacerBatch(Cpu(nthread=4), nbatch=64)
rng = np.random.default_rng(0)
state = env.reset(rng)
state, out = env.step(state, rng.uniform(-1, 1, (64, 2)))
print(out.obs.shape, out.reward.shape, out.done.sum())  # (64, 20) (64,) 0
env.close()
```

### Optimized for GPU

`Warp` steps the rows with MJX on MuJoCo Warp, on a CUDA GPU. Install the `warp` extra and the CUDA build of JAX:

```sh
uv pip install -e ".[dev,warp]" "jax[cuda12]"
```

```python
from neoracer_rl.envs.neoracer_batch import NeoRacerBatch, Warp

env = NeoRacerBatch(Warp(), nbatch=4096)
```

The arrays that go in and out are numpy arrays on both backends. Warp computes collisions and constraints with its own code, so its rows are close to `NeoRacerEnv`, not equal. Without a GPU, JAX runs Warp on the CPU, correctly but much more slowly than `Cpu`.

## Train a policy

`examples/train_ppo.py` trains a policy with PPO (proximal policy optimization), a common policy-gradient method. It runs 8 environments in a `NeoRacerBatch` on the CPU and prints one line per update, with how many episodes ended in a lap, a crash, got stuck or rolled over. On a GPU, add `--backend warp` and raise `--num-envs`. The evaluation runs `NeoRacerEnv`.

```sh
python -m examples.train_ppo --total-steps 400000 --seed 1 --out ppo.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
python -m examples.train_ppo --eval-baseline random --episodes 20
```

The thread count does not change the result: a seed gives the same run on 1 or 4 threads.

### Baseline

We trained seeds 0 to 4 for 400k steps each, on a 4-core 2.8 GHz Xeon, where a run takes about 7 minutes at about 1,030 steps per second. Then we evaluated each policy with its mean action on 20 episodes, env seeds 10000 to 10019. The middle columns give the share of the training episodes that ended in a lap, in each block of 50k steps.

```
                 laps in training, by 50k steps of training     evaluation, 20 episodes
seed  first lap   50k 100k 150k 200k 250k 300k 350k 400k     return   outcome
0     -            0%   0%   0%   0%   0%   0%   0%   0%      -3.96   16 stuck, 4 crash
1     -            0%   0%   0%   0%   0%   0%   0%   0%      -3.89   19 crash, 1 stuck
2     -            0%   0%   0%   0%   0%   0%   0%   0%      -3.73   16 stuck, 4 crash
3     293k         0%   0%   0%   0%   0%  13%  81%  84%     +14.72   18 lap, 2 crash
4     -            0%   0%   0%   0%   0%   0%   0%   0%      -3.97   20 stuck
random actions                                                -5.06   18 stuck, 2 crash
zero action                                                   -5.00   20 stuck
```

One of the five seeds learns to lap, late: its first lap comes after 293k steps. It drives the lap slowly, about 800 steps (27 s) without touching a wall, for a return of about 16.3, and crashes in 2 of the 20 evaluation episodes. The other four seeds learn the same thing: they creep for about 520 to 640 steps (17 to 21 s) to the first corner, 16% to 20% of the lap, and stop there or touch its wall. So compare changes over several seeds, not one.

The initial action noise is large (standard deviation 1.0) on purpose; the comment on `init_log_std` in `PPOParams` says why.

### Training with random physics

`--random-physics` trains and evaluates with the ranges of `RANDOM_PHYSICS` in `train_ppo.py`: wheel friction 0.3 to 1.2, chassis mass 0.7 to 1.5, motor torque 0.5 to 1.5 and steering stiffness 0.5 to 2.0. These are an example to start from, wide enough to see an effect, not measured on the real car.

We trained seeds 0 to 4 for 400k steps with them, as for the baseline, and evaluated every policy of both sets with one factor changed at a time, on 10 episodes each. The table gives the laps out of 10 of each seed:

```
                       trained on the XML     trained with random physics
                       seed 0  1  2  3  4     seed 0  1  2  3  4
wheel_friction  0.3         0  0  0 10  0          0  0  0  0  0
                0.5         0  0  0 10  0          0 10  0  0  0
                0.75        0  0  0 10  0          0 10  0  0  0
                1.0         0  0  0 10  0          0 10  0  0  0
                1.2         0  0  0 10  0          0 10  0  0  0
chassis_mass    0.7         0  0  0  8  0          0 10  0  0  0
                1.5         0  0  0 10  0          0 10  0  0  0
motor_torque    0.5         0  0  0 10  0          0 10  0  0  0
                0.75        0  0  0 10  0          0 10  0  0  0
                1.5         0  0  0  1  0          0 10  0  0  0
steer_stiffness 0.5         0  0  0  9  0          0 10  0  0  0
                2.0         0  0  0  7  0          0 10  0  0  0
```

The one policy trained on the XML that laps keeps lapping under most changes, even with 0.3 of the grip. With 1.5 times the motor torque it laps only 1 of 10 times: the same actions drive it faster than its corners allow. With random physics one seed of five learned to lap, seed 1, from 92k steps on. It laps 10 of 10 on every change but the lowest grip. The run time is about the same: on 4 cores, 980 env steps per second with random physics against 1,030 without.

### Training on generated tracks

`--tracks generated` trains on a pool of generated tracks (`--pool`, default 16, and `--difficulty`, default 1), and `--first-track-seed` picks the seeds of the pool, for example to evaluate on tracks that the policy never saw:

```sh
python -m examples.train_ppo --tracks generated --total-steps 400000 --seed 0 --out gen.npz
python -m examples.train_ppo --eval gen.npz --tracks generated --first-track-seed 1000 --episodes 20
```

We trained seeds 0 to 2 for 400k steps on the 16 tracks of difficulty 1 from seed 0. All three learned to lap: their first laps came after 139k, 139k and 72k steps, and in the last 50k steps 86%, 99% and 85% of their training episodes were laps. Then we evaluated each policy, and the baseline policies trained on `loop_corridor`, on 20 episodes each of `loop_corridor`, of the training pool, and of a held-out pool of 16 tracks from seed 1000:

```
                                    laps out of 20
policy                      loop_corridor   training pool   held-out pool
loop_corridor, seed 0                   0               0               0
               seed 1                   0               0               0
               seed 2                   0               0               0
               seed 3                  18               0               0
               seed 4                   0               0               0
generated,     seed 0                  19              20              19
               seed 1                  17              20              19
               seed 2                  20              16              18
```

The policies trained on generated tracks lap 18 or 19 of the 20 held-out tracks, which they never saw, about as reliably as those they trained on, and `loop_corridor` too. The policy trained on `loop_corridor`, which turns only left, laps none of the generated tracks, which also turn right. It is slower per step: 700 env steps per second on 4 cores, against 1,030 on `loop_corridor`.

## Layout

```
src/neoracer_rl/envs/   NeoRacerEnv, NeoRacerBatch, and their reward, progress and termination functions
examples/               training, and a run with random actions
validation/             pytest tests
users/<name>/           personal experiments, not checked by ruff, ty, the tests or CI
```

## Citing

If you use this environment in your work, cite it with the "Cite this repository" button on GitHub, which reads `CITATION.cff`, and cite the simulator, neoracer-mujoco, the same way. The authors in `CITATION.cff` are the five people with the most commits in the git history. A commit counts for its author and for each person in a `Co-authored-by` trailer, `.mailmap` merges the names and emails of one person, and AI tools and bots do not count. `.github/citation.py` writes the file, so do not edit it by hand.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Coding agents follow [AGENTS.md](AGENTS.md).

## License

GPL-3.0, the same as neoracer-mujoco. See [LICENSE](LICENSE).
