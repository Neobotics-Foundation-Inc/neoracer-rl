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

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (2,), float32)` |
| Observation Space | `Box(low, high, (20,), float32)`, with the bounds in the table below |
| import | `from neoracer_rl.envs import NeoRacerEnv` |

### Description

The car drives on `loop_corridor`, a closed corridor about 1 m wide. The goal is to complete one lap counterclockwise (seen from above) without rolling over or getting stuck against a wall. The agent sees only what the real car's sensors measure, and nothing from the simulator's internal state, so a policy trained here can in principle run on the car.

Each step applies one action for 17 physics steps of 2 ms, so a step is 34 ms of simulated time. That is the closest whole number of physics steps to 1/30 s, the scan rate of the car's LiDAR.

### Action Space

The action space is a `Box(-1, 1, (2,), float32)`. The env clips each value to [-1, 1] and multiplies it by the limit of its actuator.

| Num | Action | Control Min | Control Max | Actuator (in neoracer.xml) | Type (Unit) |
|---|---|---|---|---|---|
| 0 | Throttle, + drives forward | -1 | 1 | `fl_motor`, `fr_motor`, `rl_motor`, `rr_motor` | torque on each wheel, up to 1.0 (N m) |
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

The reward for each step is: *reward* = *progress* − *contact_cost* + *end_bonus*.

- *progress*: the angle, in radians, that the car moved around the center of the loop during the step, positive in the forward direction. One lap is worth 2π, about 6.3. It is multiplied by `progress_weight` (default 1).
- *contact_cost*: `contact_penalty` (default 0.1) if any part of the car touched a wall during the step, else 0.
- *end_bonus*: on the last step only, +`lap_bonus` (default 10) for a completed lap, −`stuck_penalty` (default 5) if the car got stuck, −`rollover_penalty` (default 10) if it rolled over.

A clean lap therefore returns at most about 16.3. `info` holds `progress_rad` and `lap_frac` (the progress so far, in radians and in laps; negative if the car drove backward), `speed_ms`, `upright_cos`, `contact`, `episode_time_s` and `termination_reason`.

### Starting State

The car starts on the bottom straight at (0, -2.5), facing the forward direction of the loop, and settles for 400 physics steps. `reset()` then moves it sideways by a value from 𝒰(-0.15, 0.15) m, turns it by a value from 𝒰(-0.15, 0.15) rad, and settles it for 50 more physics steps.

### Episode End

#### Termination

The episode terminates when one of these happens, checked in this order:

1. Rollover: the car tilts more than 60° (the cosine of its tilt drops below 0.5).
2. Stuck: the car moved less than 0.05 rad around the loop during the last 120 steps (about 4 s).
3. Lap: the car's total progress reaches 2π.

#### Truncation

The episode is truncated after 1800 steps (61.2 s of simulated time).

### Arguments

`NeoRacerEnv` is not registered with `gymnasium.make` yet. Create it directly and pass a `NeoRacerEnvConfig`:

```python
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig, RewardConfig

env = NeoRacerEnv(NeoRacerEnvConfig(reward=RewardConfig(contact_penalty=0.2)))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `car_path` | **str** | neoracer-mujoco's `neoracer.xml` | Path to the car model |
| `control_hz` | **float** | `30.0` | Target action rate, rounded to whole physics steps (see `Description`) |
| `max_episode_steps` | **int** | `1800` | Steps before truncation (see `Episode End`) |
| `spawn` | **SpawnRange** | `lateral_m=0.15, yaw_rad=0.15, resettle_steps=50` | Random start offsets (see `Starting State`) |
| `termination` | **TerminationConfig** | `upright_rollover_cos=0.5, stuck_window_s=4.0, stuck_progress_rad=0.05` | Rollover and stuck thresholds (see `Episode End`) |
| `reward` | **RewardConfig** | `progress_weight=1.0, contact_penalty=0.1, rollover_penalty=10.0, stuck_penalty=5.0, lap_bonus=10.0` | Reward weights (see `Rewards`) |
| `obs_bounds` | **ObservationBounds** | see the table in `Observation Space` | Bounds of the observation space |

`render_mode="human"` is a separate argument of `NeoRacerEnv`. It opens the MuJoCo viewer.

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
