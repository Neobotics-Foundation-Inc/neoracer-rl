# Experiments

PPO on the NeoRacer environment, with neoracer-mujoco 0.11.0. Each run trains for 400k steps on 8 rows of a `NeoRacerBatch` on the CPU, on a 4-core 2.8 GHz Xeon, where a run takes 7 to 13 minutes. Each policy is then evaluated with its mean action, on env seeds from 10000. A seed gives the same run on 1 or 4 threads.

```sh
python -m examples.train_ppo --total-steps 400000 --seed 0 --out ppo.npz
python -m examples.train_ppo --total-steps 400000 --seed 0 --random-physics --out ppo_dr.npz
python -m examples.train_ppo --total-steps 400000 --seed 0 --tracks generated --out ppo_gen.npz
python -m examples.train_ppo --eval ppo.npz --episodes 20
python -m examples.train_ppo --eval ppo_gen.npz --tracks generated --first-track-seed 1000 --episodes 20
```

## Which regimens learn to lap

```
regimen                          seeds that lap   first lap           laps in training, last 50k steps
loop_corridor                    0 of 5           -                   -
loop_corridor, random physics    3 of 5           124k, 157k, 276k    87%, 91%, 69%
16 generated tracks              2 of 3           77k, 81k            89%, 91%
```

On `loop_corridor` alone, no seed learns to lap. Each creeps to the first corner, 15% to 22% of the lap, and stops there or touches its wall, for a return of about -4. Random actions and the zero action return about -5.

Learning to lap from `loop_corridor` alone is rare, not impossible. On neoracer-mujoco 0.10.0, whose physics differs from 0.11.0's only by rounding, seed 3 learned it after 293k steps, and that policy still laps 19 of 20 times on 0.11.0. Training is deterministic, so the difference comes from the physics: a change in the 5th digit sends a run down another path. Compare changes over several seeds, not one.

Random physics (`--random-physics`, the ranges of `RANDOM_PHYSICS` in `train_ppo.py`) and generated tracks (`--tracks generated`, 16 tracks of difficulty 1) give the learner more variety, and more of their seeds learn to lap.

## Random physics

Each policy trained with random physics, evaluated with one factor changed at a time, laps out of 10:

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

The three that learned lap under every change but the lowest grip. The policies trained on `loop_corridor` lap under none. The ranges are wide enough to show an effect; they were not measured on the car.

## Generated tracks

The two policies trained on generated tracks that learned lap 20 of 20 episodes on `loop_corridor`, on their 16 training tracks, and on 16 held-out tracks from seed 1000 that they never saw. The policies trained on `loop_corridor` lap none of the generated tracks. Generated tracks are slower to train on: about 560 env steps a second on 4 cores, against 820 on `loop_corridor`.
