# A driving policy for the RACECAR Neo V2

MIT's RACECAR Neo V2 is a small autonomous car with a spinning LiDAR, a drive board with a hall encoder, and a Raspberry Pi 5 that runs students' Python programs through its `racecar_core` library. This folder holds a policy trained with PPO in the V2's MuJoCo model from neoracer-mujoco, and a `racecar_core` program that drives the real car with it. The car needs only these files:

- `driver.py`: the program.
- `policy.py`: what the policy sees and how it acts. It needs numpy only and runs on the car's Python 3.12.
- `v2_policy.npz`: the trained weights.

Every 1/30 s the policy reads eleven numbers: the range in eight directions, every 45 degrees around the car, from the LiDAR's scan (5 m for no return or farther); the speed from the encoder; and its own last action. It sets the throttle and the steering.

## Running it on the car

Copy this folder next to the car's labs, so that the `racecar_core` library is two directories up, and run the program from the folder:

```sh
python3 driver.py
```

Hold the right bumper while it drives, and release it to stop. The car also stops when the LiDAR's scan has not changed for half a second, as when the LiDAR stops sending (`racecar_core` then repeats its last scan), and drives again on the next scan. The settings at the top of `driver.py`:

- `SPEED_LIMIT` is the share of the policy's throttle that the car gets. The car's speed controller holds 0.72 m/s at a quarter of full throttle, 0.83 m/s at half and 1.07 m/s at full.
- `DIRECTION = "clockwise"` drives the other way round. The policy then sees the scan mirrored and steers mirrored, since the car is close to symmetric.
- `REQUIRE_DEADMAN` drives only while the right bumper is held.

The track needs closed walls about a metre apart, taller than the LiDAR, which sits about 19 cm above the floor. The policy has only seen `loop_corridor`, a corridor 1 m wide around a 6 m by 4 m block with chamfered corners. A different width or tighter corners may need retraining.

The policy has only driven in simulation. The model is fitted to one reference car's calibration runs, but the real LiDAR, encoder and timing may differ. For a first run, set a low `SPEED_LIMIT`, use an open space, and keep a hand on the bumper.

## How it was trained

```sh
python3 -m examples.train_v2 --total-steps 2500000 --seed 0 --out v2_policy.npz
```

This took about 37 minutes with `--nthread 2`, on a 4-core machine that trained a second seed at the same time. Each episode drew the car's grip, mass, motor torque and steering stiffness from wide ranges (`PHYSICS` in `train_v2.py`). The policy saw the LiDAR as `racecar_core` gives it: the head turns about 7.4 times a second and measures each direction as it passes it, and `get_samples()` returns the last complete turn, so a range is up to two turns, 270 ms, old. It saw that with noise and lost beams, both held for the whole turn as `get_samples()` repeats it, the encoder reading between 5% low and 20% high, and, in half the episodes, its actions reaching the car a step late. Not every seed learns to lap: seed 1, trained the same way, learned to drive the wrong way round and never completed a lap.

`python3 -m examples.train_v2 --eval examples/racecar_neo_v2/v2_policy.npz` drives 32 episodes of up to 60 s with the same random physics and sensing. 28 complete a lap, in 39.3 s on average, and 4 touch a wall.

## In simulation

`python3 -m examples.drive_v2` runs `driver.py` unchanged on the model through neoracer-mujoco's `racecar_core` adapter, which gives it the V2's spinning 1080-sample LiDAR and its hall encoder. There it drives `loop_corridor` without touching a wall:

| Run | Laps | Wall contacts |
| --- | --- | --- |
| counterclockwise, 5 min | 8.2, 36 s a lap | 0 |
| `--direction clockwise`, 2 min | 3.2 | 0 |
| `--speed-limit 0.5`, 1 min | 1.5 | 0 |
| `--speed-limit 0.25`, 1 min | 1.5 | 0 |

If the car's `racecar_core` ignores `set_max_speed`, the car keeps its start max speed, 0.25, and drives like the last row.
