# Experiments

Deep Q-learning on the NeoRacer, driving `loop_corridor`. These results come
from commit 18ccc8e, which pins neoracer-mujoco 0.29.0, on an 8-core Apple M
laptop. To make them again, check out that commit and run

```sh
python3 users/aayan/experiments.py runs
```

It trains every seed below, then prints these tables. Five seeds of three
learning rates take about three hours when five train at a time.

Each run is deep Q-learning for 300k environment steps, with double Q-learning,
a 128 by 128 network and a replay buffer of 100k transitions. The action is one
of fifteen pairs: a speed from `(0.05, 0.10, 0.18)` of the firmware loop's
maximum and a steering angle from `(-1, -0.5, 0, 0.5, 1)`. Every 25k steps the
policy is evaluated greedily over 3 episodes on env seeds from 9000.

Training is deterministic. A seed gives the same run: seed 0 of the default
setting reaches its best, a lap fraction of 0.958, at step 175k every time.

## Keep the best network, not the last

A run reaches a full lap and then comes apart. Across the five seeds of the
default setting, every one peaked near 175k steps and every one was worse at
300k.

```
setting     seeds              best lap  last lap  reached a lap  peaked at

lr3e-4          5  0.928 [0.800, 1.001]     0.463         5 of 5       175k
lr5e-4          5  0.873 [0.694, 0.989]     0.404         5 of 5       200k
lr1e-3          5  0.698 [0.556, 0.847]     0.252         3 of 5       225k
```

A lap fraction of 1.0 is a finished lap, so the default setting averages 0.93
at its best and 0.46 by the end. The interval is a bootstrap 95% interval on
the mean, and "reached a lap" counts the seeds that completed one during the
evaluation they scored best on.

The gap between those two columns is the whole reason `dqn.py` now evaluates
while it trains and saves the best network it has seen. Reading the last
network instead halves the measured result and, worse, gives a video of a car
driving into a wall for a seed that had been lapping cleanly an hour earlier.

## Whether the learning rate matters

```
setting     difference          95% interval       p       d

lr5e-4          -0.056      [-0.281, +0.145]   0.529   -0.32
lr1e-3          -0.230      [-0.402, -0.042]   0.055   -1.39
```

Each row compares with `lr3e-4` on the best lap fraction. The interval is a
bootstrap 95% interval on the difference of means and the p-value comes from a
permutation test, which suits five seeds a side better than a t-test.

Raising the learning rate does not help and the largest value hurts. 5e-4 is
within noise of the default. 1e-3 is worse by 0.23 of a lap with an interval
that stays below zero, and two of its five seeds never completed a lap at all.
It also peaks later, at 225k against 175k, which is what a run that is still
thrashing looks like. The default of 3e-4 was already the right choice.

## What the car is actually doing

The policies that lap take 31 to 37 seconds. The tuned classical controllers in
`baselines.py` lap in about 13 and a tabular Sarsa table in about 53, so deep
Q-learning sits between them.

Most of that gap is the action set, not the policy. The fastest speed the agent
can ask for is 0.18, which settles at about 1.28 m/s, while the controllers
command speeds directly and run at 2.2 to 3.0. The rest is that it does not
hurry: `ProgressReward` pays per radian of progress with no cost per step, so
crawling is free, and the greedy policy spends more of its time at the slowest
of its three speeds than at the fastest. Widening the speeds and adding a cost
per step are the next two things to try, and they have to be tried together,
since neither helps alone.

## Still open

- One track. Everything here is `loop_corridor`, and the environment can train
  on generated tracks through `GeneratedTracks`.
- Three speeds and five steering angles is a coarse way to act, and it caps the
  car well below what the firmware loop will give.
- Five seeds is enough to rule out a large effect and not enough to find a
  small one. The interval on the default setting runs from 0.80 to 1.00.
