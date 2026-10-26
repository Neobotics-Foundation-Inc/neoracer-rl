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

The reward is a plain function of the step's progress and the reason the episode ended, if it did:

```python
from neoracer_rl.envs import RewardConfig, reward

reward(RewardConfig(), 0.1, None)  # 0.1
reward(RewardConfig(), 0.1, "lap_complete")  # 10.1
reward(RewardConfig(), 0.0, "crash")  # -5.0
```

The reason is a `Literal["rollover", "crash", "stuck", "lap_complete"]`, so the type checker sees every case. Progress and termination are plain functions too: `advance` returns the next `Progress` from the last one and the car's new angle, and `termination` returns the reason or `None`.

## Changing the task

`NeoRacerEnvConfig` holds the whole task: the track, the rewards, the spawn, the end conditions, the share of the max speed that a speed of 1 asks for, and how much to randomize the physics. To change the task, make a new config:

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

`lap_angle(arena, sensordata)` computes progress from the layout:
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

The batch uses the env's `observation`, `lap_angle` and rewards, and array versions of its progress and termination checks. On the `Cpu` backend, a row's first episode steps exactly like `NeoRacerEnv` from the same spawn, bit for bit. The rows' speed loops tick together, so a row that starts over keeps the batch's ticks, while `NeoRacerEnv.reset` starts its own over. `Warp()` runs the rows on a CUDA GPU with MuJoCo Warp, all on one track.

## Training a policy

`examples/train_ppo.py` is a small PPO baseline in JAX and optax. The policy is a Gaussian over speed and steering, with an MLP for the mean.

```sh
python -m examples.train_ppo --total-steps 400000 --seed 1 --out ppo.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
```

`--random-physics` trains with random physics, `--tracks generated` on generated tracks, and `--backend warp` on a GPU. [EXPERIMENTS.md](EXPERIMENTS.md) has the results. For example, three of five seeds trained on `loop_corridor` alone learn to lap it, seed 1, above, among them, and one of the three laps 18 of 20 episodes on generated tracks it never saw.

`examples/train_v2.py` trains a policy for another car, MIT's RACECAR Neo V2, on what that car's `racecar_core` library shows a program: a LiDAR scan up to two turns of its head old, and its encoder's speed. [examples/racecar_neo_v2](examples/racecar_neo_v2/README.md) holds the program that drives the real car with it.

## Where the code lives

- `src/neoracer_rl/envs/neoracer_env.py` has `NeoRacerEnv` and the functions that define the task: the observation, progress, termination and reward.
- `src/neoracer_rl/envs/neoracer_batch.py` has `NeoRacerBatch` and the array versions of those functions.
- A new part of the task, such as another reward term, goes in `neoracer_env.py` as a function that both the env and the batch can use. Physics and sensors belong in neoracer-mujoco.
- `examples/` has `train_ppo.py`; `experiments.py`, which trains and evaluates every run of EXPERIMENTS.md and prints its tables; `random_actions.py`, which checks that the environment works; and the RACECAR Neo V2's policy: `train_v2.py`, `drive_v2.py` and `racecar_neo_v2/`.
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
