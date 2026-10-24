# Experiments

PPO on the NeoRacer environment, with neoracer-mujoco 0.20.0. The action asks the car's firmware speed loop for a speed, as the car's computer does, and the loop sets the throttle. Each run trains for 400k steps on 8 rows of a `NeoRacerBatch` on the CPU, on a 4-core 2.1 GHz Xeon, where a run takes 6 to 7.5 minutes. Each policy is then evaluated with its mean action, on env seeds from 10000. A seed gives the same run on 1 or 4 threads.

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
loop_corridor                    1 of 5           127k                92%
loop_corridor, random physics    3 of 5           125k, 283k, 125k    81%, 67%, 80%
16 generated tracks              2 of 3           182k, 172k          67%, 77%
```

On `loop_corridor` alone, seed 1 learns to lap and laps 20 of 20 evaluation episodes. The other four drive 12% to 48% of the lap on average and then stop or touch a wall, for a return of -2 to -4. Random actions and the zero action return about -5.

Which seeds learn changes with the physics. On 0.10.0 seed 3 learned `loop_corridor`; on 0.11.0, whose physics differs from 0.10.0's only by rounding, no seed did; on 0.11.2, whose steering has ten times the inertia so the car no longer shimmies in part-throttle turns, seed 1 did. With the speed loop, seed 1 still does, and the three regimens have 1, 3 and 2 seeds that lap, against 1, 3 and 3 when the action was the throttle. Training is deterministic, so a change in the physics alone sends a run down another path. Compare changes over several seeds, not one.

Random physics (`--random-physics`, the ranges of `RANDOM_PHYSICS` in `train_ppo.py`) and generated tracks (`--tracks generated`, 16 tracks of difficulty 1) give the learner more variety, and more of their seeds learn to lap.

## Random physics

Each policy trained with random physics, evaluated with one factor changed at a time, laps out of 10:

```
                 seed  0   1   2   3   4
wheel_friction  0.3    0  10   0   0   0
                0.5    4   9   0   0   9
                0.75  10  10   0   0  10
                1.0   10  10   0   0  10
                1.2   10  10   0   0  10
chassis_mass    0.7    9  10   0   0  10
                1.5   10  10   0   0  10
motor_torque    0.5   10  10   0   0  10
                0.75  10  10   0   0  10
                1.5    9  10   0   0  10
steer_stiffness 0.5   10  10   0   0  10
                2.0   10   8   0   0  10
```

The three that learned lap under every change but the low grips, missing at most two laps of ten. Seed 1 laps even on the lowest grip, where no policy lapped when the action was the throttle; seeds 0 and 4 lap none at 0.3, and 4 and 9 at 0.5. The one policy that learned `loop_corridor` alone laps under every change but the two lowest grips, 6 of 10 at 0.5 and none at 0.3; the other four policies trained on `loop_corridor` lap under none. The ranges are wide enough to show an effect; they were not measured on the car.

## Generated tracks

The two policies trained on generated tracks that learned to lap lap 20 of 20 episodes on `loop_corridor`, 19 and 17 of 20 on their 16 training tracks, and 18 and 19 of 20 on 16 held-out tracks from seed 1000 that they never saw. The third, seed 1, never laps: on its training tracks, 15 of 20 episodes end at the 60 s limit, about half a lap in. The policy that learned `loop_corridor` alone laps 15 of 20 on the training tracks and 12 of 20 on the held-out ones, against 6 and 3 when the action was the throttle; the other policies trained on `loop_corridor` lap none of the generated tracks. Generated tracks are slower to train on: about 950 env steps a second on 4 cores, against 1,190 on `loop_corridor`.
