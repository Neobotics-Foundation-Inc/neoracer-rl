# Experiments

These results come from commit [e19d01a](https://github.com/Neobotics-Foundation-Inc/neoracer-rl/commit/e19d01a8f9e534be6692c3bc0d2a16219405704d), which pins neoracer-mujoco 0.25.0, on a 4-core 2.1 GHz Xeon. To make them again, check out that commit and run

```sh
python -m examples.experiments runs
```

It trains every run below, evaluates every policy, and prints the tables of this page.

Each run is PPO on the NeoRacer environment for 400k steps, on 8 rows of a `NeoRacerBatch` on the CPU. The action asks the car's firmware speed loop for a speed, as the car's computer does, and the loop sets the throttle. Each policy is then evaluated with its mean action, on env seeds from 10000. Training is deterministic: a seed gives the same run on 1 or 4 threads. Alone on 4 threads, a run takes 6 minutes on `loop_corridor` and 7.4 on generated tracks.

One run and its evaluation, by hand:

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
loop_corridor                    3 of 5           162k, 174k, 293k    96%, 86%, 100%
loop_corridor, random physics    3 of 5           170k, 129k, 131k    77%, 87%, 89%
16 generated tracks              1 of 3           106k                61%
```

On `loop_corridor` alone, seeds 1, 3 and 4 learn to lap, and each laps 20 of 20 evaluation episodes. Seeds 0 and 2 drive about a tenth of a lap the wrong way and then stop or touch a wall, for a return of about -5.7. Random actions and the zero action return about -5.

Because training is deterministic, a change in the physics alone sends a run down another path, so which seeds learn changes with the simulator's physics. Compare changes over several seeds, not one.

Random physics (`--random-physics`, the ranges of `RANDOM_PHYSICS` in `train_ppo.py`) and generated tracks (`--tracks generated`, 16 tracks of difficulty 1) give the learner more variety. Here they do not make more seeds learn to lap, and the policies that learned `loop_corridor` alone hold up as well off it, under other physics and on other tracks, as the next two sections show.

## Random physics

Each policy, evaluated on `loop_corridor` with one factor changed at a time, laps out of 10:

```
Trained on loop_corridor:
                 seed  0   1   2   3   4
wheel_friction  0.3    0   0   0   0   1
                0.5    0  10   0   9  10
                0.75   0  10   0  10  10
                1.0    0  10   0  10  10
                1.2    0  10   0  10  10
chassis_mass    0.7    0  10   0  10  10
                1.5    0  10   0  10  10
motor_torque    0.5    0  10   0  10  10
                0.75   0  10   0  10  10
                1.5    0  10   0  10   9
steer_stiffness 0.5    0  10   0  10  10
                2.0    0  10   0  10   9

Trained on loop_corridor, random physics:
                 seed  0   1   2   3   4
wheel_friction  0.3    0   0   0   0   4
                0.5    6  10   0   0  10
                0.75  10  10   0   0  10
                1.0   10  10   0   0  10
                1.2   10  10   0   0  10
chassis_mass    0.7   10  10   0   0  10
                1.5   10  10   0   0   9
motor_torque    0.5   10  10   0   0  10
                0.75  10  10   0   0  10
                1.5    8  10   0   0   9
steer_stiffness 0.5   10  10   0   0  10
                2.0    8  10   0   0   9
```

The three policies trained with random physics that learned lap under every change but the low grips, missing at most two laps of ten. On the lowest grip, seed 4 laps 4 of 10 and seeds 0 and 1 none; at 0.5, seed 0 laps 6. The three that learned `loop_corridor` alone lap 9 or 10 of 10 under every change but the lowest grip, where seed 4 laps 1 and seeds 1 and 3 none. The policies that never learned lap under no change. The ranges are wide enough to show an effect; they were not measured on the car.

## Generated tracks

The one policy trained on generated tracks that learned to lap laps 17 of 20 episodes on `loop_corridor`, 18 of 20 on its 16 training tracks, and 15 of 20 on 16 held-out tracks from seed 1000 that it never saw. The other two never lap: on its training tracks, seed 1 ends 17 of 20 episodes at the 60 s limit, a fifth of a lap the wrong way on average, and seed 2 stops at the start. The policies that learned `loop_corridor` alone lap tracks they never saw too: seed 3 laps 19 of 20 on the generated training tracks and 18 of 20 on the held-out ones, seed 1 14 and 12, and seed 4 5 and 3. Generated tracks are slower to train on: about 940 env steps a second on 4 cores, against 1,150 on `loop_corridor`.

## Racing a controller

These results come from the code of commit [592e47a](https://github.com/Neobotics-Foundation-Inc/neoracer-rl/commit/592e47a), which pins neoracer-mujoco 0.27.0, on the same Xeon, four runs at a time on one thread each. In each race of `examples/train_race.py`, the learner starts at `loop_corridor`'s start and a `TrackCentering` car 1 m ahead of it, about 0.06 laps. Touching that car ends the race as a wall does, for the crash penalty. A run is 400k steps on 8 races, the PPO of the runs above. To make them again:

```sh
for s in 0 1 2 3 4; do
  python -m examples.train_ppo --total-steps 400000 --seed $s --out l$s.npz
  python -m examples.train_race --total-steps 400000 --seed $s --out r$s.npz
done
for s in 1 3 4; do
  python -m examples.train_race --total-steps 400000 --seed $s --init-from l$s.npz --out f$s.npz
done
python -m examples.train_race --eval f4.npz --races 20
python -m examples.train_ppo --eval f4.npz --episodes 20
```

The runs trained alone are the `loop_corridor` runs above, update for update: seeds 1, 3 and 4 first lap at 162k, 174k and 293k steps. Each policy, in 20 races from the spawns of `default_rng(10000)`, and its mean lap time alone over 20 episodes:

```
policy                    races: lap  touch  crash  stuck  ahead   lap alone
trained alone, seed 0               0     14      0      6      0   none
               seed 1               0     20      0      0      0   10.9 s
               seed 2               0     17      3      0      0   none
               seed 3               0     15      3      2      0   10.4 s
               seed 4              20      0      0      0      0   24.4 s
trained in races, seed 0            0     18      0      2      0
                  seed 1            0      0      0     20      0
                  seed 2            0     11      0      9      0
                  seed 3            0     16      0      4      0
                  seed 4            0     17      0      3      0
alone, then races, seed 1          10      5      5      0      0   15.4 s
                   seed 3           2     17      1      0      0   14.1 s
                   seed 4          14      5      1      0      0   13.3 s
```

No policy passes the controller's car. The controller laps in about 17 s. The two fast policies trained alone, seeds 1 and 3, catch it within the first fifth of a lap and run into it in nearly every race. Seed 4 laps in 24 s, slower than the controller, so it never catches it: it laps every race, half a lap behind.

Trained in races from the start, no seed learns to lap in 400k steps, in training or in the races after. Four run into the car ahead, and seed 1 learns to stay where it is.

Started from the policies that lap alone, 400k steps in races teach them to follow. Seed 4 laps 14 of 20 races and seed 1 10, each finishing within a tenth of a lap behind the controller's car; seed 3 still runs into it in 17. Their pace alone moves toward the controller's: seeds 1 and 3 slow from about 11 s a lap to 15 and 14, and seed 4 speeds up from 24 s to 13. The task rewards progress and costs a touch, and nothing in it rewards a pass, so in a corridor 1 m wide, staying behind is what pays; a reward for gaining on the car ahead, as the README's example gives, is the next thing to try.

A race of two cars trains at about 315 steps a second on each of four busy cores, against 580 for one car: 21 minutes for 400k steps.
