# Experiments

PPO on the NeoRacer environment, with neoracer-mujoco 0.11.2. Each run trains for 400k steps on 8 rows of a `NeoRacerBatch` on the CPU, on a 4-core 2.1 GHz Xeon, where a run takes 4 to 7 minutes. Each policy is then evaluated with its mean action, on env seeds from 10000. A seed gives the same run on 1 or 4 threads.

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
loop_corridor                    1 of 5           66k                 99%
loop_corridor, random physics    3 of 5           117k, 43k, 193k     94%, 92%, 91%
16 generated tracks              3 of 3           90k, 80k, 106k      99%, 97%, 93%
```

On `loop_corridor` alone, seed 1 learns to lap and laps 20 of 20 evaluation episodes. The other four creep to the first corner, 15% of the lap, and stop there or touch its wall, for a return of about -4. Random actions and the zero action return about -5.

Which seeds learn changes with the physics. On 0.10.0 seed 3 learned `loop_corridor`; on 0.11.0, whose physics differs from 0.10.0's only by rounding, no seed did; on 0.11.2, whose steering has ten times the inertia so the car no longer shimmies in part-throttle turns, seed 1 does. Training is deterministic, so a change in the physics alone sends a run down another path. Compare changes over several seeds, not one.

Random physics (`--random-physics`, the ranges of `RANDOM_PHYSICS` in `train_ppo.py`) and generated tracks (`--tracks generated`, 16 tracks of difficulty 1) give the learner more variety, and more of their seeds learn to lap.

## Random physics

Each policy trained with random physics, evaluated with one factor changed at a time, laps out of 10:

```
                 seed  0   1   2   3   4
wheel_friction  0.3    0   0   0   0   0
                0.5   10  10   0  10   0
                0.75  10  10   0  10   0
                1.0   10  10   0  10   0
                1.2   10  10   0  10   0
chassis_mass    0.7   10  10   0  10   0
                1.5   10  10   0  10   0
motor_torque    0.5   10  10   0  10   0
                0.75  10  10   0  10   0
                1.5   10  10   0   9   0
steer_stiffness 0.5   10  10   0  10   0
                2.0   10  10   0  10   0
```

The three that learned lap under every change but the lowest grip. So does the one policy that learned `loop_corridor` alone; the other four policies trained on `loop_corridor` lap under none. The ranges are wide enough to show an effect; they were not measured on the car.

## Generated tracks

The three policies trained on generated tracks lap 20 of 20 episodes on `loop_corridor`, on their 16 training tracks, and on 16 held-out tracks from seed 1000 that they never saw. The policy that learned `loop_corridor` alone laps 6 of 20 on the training tracks and 3 of 20 on the held-out ones; the other policies trained on `loop_corridor` lap none of the generated tracks. Generated tracks are slower to train on: about 940 env steps a second on 4 cores, against 1,750 on `loop_corridor`.
