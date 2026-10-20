# aayan

Tabular reinforcement learning on the NeoRacer, using the methods from chapters
4 to 6 of Sutton and Barto's *Reinforcement Learning: An Introduction*. These
are the classical table based methods rather than a deep learning library, so
the interesting part is how a continuous driving problem is cut down until a
table can hold it.

Everything here is an experiment. Nothing in `src/`, `examples/` or
`validation/` imports from this folder.

## What is here

| file | what it does |
|---|---|
| `tabular.py` | the chapter 4 to 6 agents, and the discretizer that makes a table possible |
| `dqn.py` | deep Q-learning, the same problem with a network instead of a table |
| `baselines.py` | classical controllers to measure a learned policy against |
| `multicar.py` | puts more than one car on a track |
| `duel.py` | a Gymnasium environment where the agent races an opponent |
| `run.py` | scores the controllers, tunes their gains, and runs duels |
| `store.py` | where tables and results go, so a run never overwrites the last |
| `tracks.py` | names the tracks, composes them, and gives a centre line |
| `record.py` | writes an mp4 of a policy driving |
| `panel.py`, `panel.html` | a browser panel over all of it |

## Running it

The package needs Python 3.15. From the repository root:

```sh
python3 users/aayan/tabular.py sarsa --episodes 800   # train one method
python3 users/aayan/tabular.py all --episodes 800     # train and compare all five
python3 users/aayan/tabular.py --eval runs/sarsa.npz  # score a saved table
python3 users/aayan/dqn.py --steps 300000             # deep Q-learning
python3 users/aayan/dqn.py --hidden 256 256 --learning-rate 1e-4
python3 users/aayan/run.py solo                       # score the classical controllers
python3 users/aayan/run.py tune pd_centering          # sweep one controller's gains
python3 users/aayan/run.py duel --agent runs/sarsa.npz --opponent pure_pursuit
python3 users/aayan/record.py runs/sarsa.npz          # write an mp4 of it driving
python3 users/aayan/tracks.py                         # list the tracks and check they load
python3 users/aayan/multicar.py                       # put two cars on the track
python3 users/aayan/panel.py                          # a browser panel for all of it
```

`record.py` needs `imageio` and `imageio-ffmpeg`, which are not dependencies of
the project. Install them into the same environment if you want video.

## Where experiments go

Everything lands in `runs/`, which git ignores because it can all be rebuilt.

A trained table is named after what made it, as in `sarsa_800ep_s0.npz`, and a
repeat of the same command adds `_2` rather than replacing the file. Each one
carries the settings that produced it, including the package versions, which
matter because the simulator moves fast. Read them back with
`store.load_table`, or see `tabular.py --eval`, which prints them.

Every run also appends one line to `runs/results.jsonl`: training, evaluation,
a solo score, a tuning sweep, a duel. The file is only ever appended to. A
tuning sweep additionally writes every setting it tried to
`runs/tune_<controller>.jsonl`, because the sweep is the slow part and the
printed top six throws the rest away.

```sh
python3 users/aayan/store.py     # list what has been run and what is saved
```

## The five methods

Each one is the algorithm box from the book, written out directly.

| | book | what it does |
|---|---|---|
| `dp` | 4.4 | value iteration, on an MDP estimated from random rollouts |
| `mc` | 5.4 | on policy first visit Monte Carlo control |
| `sarsa` | 6.4 | on policy temporal difference control |
| `qlearning` | 6.5 | off policy temporal difference control |
| `esarsa` | 6.6 | Expected Sarsa |

## Making a continuous problem fit in a table

Chapters 4 to 6 store one number for every state and action pair. `NeoRacerEnv`
gives a 20 number observation and takes a 2 number action, so neither a row nor
a column of that table exists until you invent them. That choice, rather than
the algorithms, is where the difficulty lives.

The state keeps three signals from the 8 beam LiDAR ring.

- **lateral**, the left beam minus the right beam, in metres. Am I off centre?
  5 bins.
- **skew**, the rear side beam minus the front side beam, left minus right, in
  metres. Am I pointed along the wall or into it? 5 bins.
- **forward**, the distance to the wall ahead, in metres. Is a corner coming?
  4 bins.

That gives 100 states against 15 actions, which is three throttle settings
crossed with five steering angles. A few hundred episodes fill a table that
size, which is the point. These methods are meant to be watched while they
learn.

`skew` comes from the simulator's wall following controller, which estimates
heading against a wall the same way. Without it the state cannot tell running
parallel to a wall from driving into it, and the agent scores below a random
policy.

## Two measurements that mattered more than the algorithms

**Torque.** The drive motors take a torque in newton metres over a control
range of -1 to 1, and the car is light. The simulator's `run.py` calls 0.08 a
cruise of about 1.5 m/s and 0.35 enough to flip the car at speed. An action
grid that reaches 1.0 only teaches a policy to drive into the nearest wall, so
the three throttle settings here stay between 0.05 and 0.18.

**The default reward is dominated by contact.** At `contact_penalty` 0.1 per
step and 30 Hz, one second spent scraping a wall costs 3.0, against the 6.28
that a whole lap pays in progress reward. Over one measured episode 49% of the
steps were in contact, which cost 14.0 against 2.4 earned in progress. Under
that balance the best thing a cautious agent can do is stop moving, and that is
what every method here learned. Training therefore prices contact at 0.02.
Evaluation always uses the default reward, so the numbers below stay comparable
with the random baseline in `examples/gymnasium_random_smoke.py`.

## Results

Greedy evaluation over 5 fixed seeds, scored on the default reward, after 800
training episodes each.

<!-- RESULTS -->

Sarsa beating Q-learning is the cliff walking result from example 6.6 showing
up in a simulator. Q-learning learns the value of the greedy path, which here
hugs the walls, while Sarsa learns the value of the path it actually drives,
exploration included, so it keeps a margin and stops wedging itself in corners.

Value iteration finishing last is a model problem rather than an algorithm
problem. It converges cleanly, but on an MDP estimated from random rollouts
that reach only part of the state and action pairs and rarely get far enough
round the track to see what a corner does. The estimate is sharpest where it
matters least. That is chapter 4's assumption, that the transition
probabilities are given, failing honestly.

## Deep Q-learning

`dqn.py` is the same problem with one piece swapped. It keeps the 15 action
grid and the environment's observation, and replaces the 100 row table with a
network that reads all 20 numbers. That isolates the question worth asking:
does a learned function beat bins drawn by hand?

It has the pieces from the 2015 paper. A replay buffer, so updates are not all
from one trajectory. A target network held still between syncs, so the
regression has a fixed aim. Epsilon decayed from 1 to a floor. Double
Q-learning is on by default, where the online network picks the next action
and the target network scores it, which removes most of the overestimation a
plain max introduces. Observations are normalized by a running mean and
variance, the same way `examples/train_ppo.py` does it, because the raw
numbers mix metres with m/s squared with a quaternion.

JAX and optax rather than PyTorch, which has no wheels for Python 3.15. The
network is built the way the PPO example builds its own.

Everything is on the command line:

```sh
python3 users/aayan/dqn.py --steps 300000
python3 users/aayan/dqn.py --hidden 256 256 128 --learning-rate 1e-4
python3 users/aayan/dqn.py --batch-size 128 --buffer-size 200000 --no-double
python3 users/aayan/dqn.py --eval runs/dqn_300000steps_s0.npz
```

| flag | default | what it is |
|---|---|---|
| `--hidden` | `128 128` | widths of the hidden layers, any number of them |
| `--learning-rate` | `3e-4` | Adam step size |
| `--batch-size` | `64` | transitions per update |
| `--buffer-size` | `100000` | transitions kept for replay |
| `--warmup-steps` | `2000` | steps collected before the first update |
| `--train-every` | `1` | environment steps between updates |
| `--target-sync-every` | `1000` | steps between target network copies |
| `--gamma` | `0.99` | discount |
| `--eps-start`, `--eps-end`, `--eps-fraction` | `1.0`, `0.05`, `0.3` | the epsilon schedule |
| `--no-double` | off | plain DQN instead of double |

The defaults are measured rather than assumed. The classic DQN updates once
every four environment steps; here that was clearly worse, 0.145 mean lap
against 0.321 for updating every step, so the default updates every step.

### What it does, measured 2026-10-18

Greedy evaluation over 5 seeds on `loop_corridor`, against the Sarsa table
trained on the same action grid and the same environment.

| | steps or episodes | mean return | mean lap | laps |
|---|---|---|---|---|
| Sarsa table | 800 episodes | +16.29 | 1.000 | 5/5 |
| DQN | 300k steps | -2.98 | 0.321 | 0/5 |
| DQN | 800k steps | -1.61 | 0.540 | 0/5 |

The table wins, clearly. More steps keep helping DQN, 0.321 to 0.540, but at
800k it still crashes on every evaluation episode rather than finishing.

It is not driving too fast, which was my first guess. Over one episode the
network averages 0.35 m/s against the table's 0.41, and both peak at the same
0.62, because both pick from the same 15 actions. It simply corners less
reliably.

The likelier reason is that the discretizer is a strong prior rather than a
limitation. Lateral offset, wall skew and forward clearance are the three
quantities a corridor follower needs, chosen by hand from knowing the problem,
and the table has 1500 numbers to fill. The network gets the raw observation
and has to discover something equivalent inside about 20,000 weights. On this
budget the hand-built representation is worth more than the capacity to learn
one. That is the comparison worth having, and it points the other way from
what I expected.

## Racing a second car

`multicar.py` puts more than one NeoRacer on a track. The simulator attaches a
car to a track by grafting its body into the track's spec, and MjSpec takes a
name prefix, so attaching the same car file twice gives two complete and
independent cars in one model: `a_motor` and `b_motor`, `a_imu_accel` and
`b_imu_accel`. Nothing in the simulator looks a part up by a global name, it
binds spec elements, so both cars bind to the same model without clashing.

`duel.py` wraps that as a Gymnasium environment. The agent drives one car and
an opponent drives the other. The agent's observation is the same 20 numbers
`NeoRacerEnv` gives, in the same order, so a policy trained alone races without
changing a line, and it sees the other car only the way the real car would, as
a shorter LiDAR beam. Reward and termination come from the same public
functions `NeoRacerEnv` uses, so a return in a duel means what a return there
means.

```sh
python3 users/aayan/run.py duel --agent runs/sarsa.npz --opponent pure_pursuit
python3 users/aayan/run.py duel --opponent runs/mc.npz --abreast
```

The opponent is any of the controllers below, or a table saved earlier, so a
policy can race the one that trained before it. What the opponent cannot do is
block on purpose. It drives its own race and the agent has to get past it,
which is enough to stop the track being empty.

A Sarsa table trained alone for 800 episodes finishes every lap by itself, in
about 53 s. Put a controller on the same track and that changes:

| opponent | agent lap | opponent lap | agent finished |
|---|---|---|---|
| `pd_centering` | 1.000 | 1.847 | 4/4 |
| `follow_the_gap` | 1.001 | 2.979 | 4/4 |
| `wall_follow` | 0.899 | 2.967 | 2/4 |
| `pure_pursuit` | 0.818 | 1.726 | 1/4 |

Every controller laps the agent, which is no surprise when they go round in
13 s and it takes 53 s. The interesting column is the last one. Against the
two fastest the agent stops finishing at all: it crashes partway, usually
while being caught. Trained alone, it never had to hold a line with another
car beside it, and the state it learned has no way to tell a wall from a car,
since both are just a short beam.

## The classical controllers

A learning curve on its own says little, so `baselines.py` holds the drivers a
tabular agent has to beat. Three of them read only the observation, which is
what the learner sees. Pure pursuit instead follows the track's centre line
using the car's true pose, so it is a reference rather than a fair opponent,
and `run.py solo` marks it.

| | what it does | sees |
|---|---|---|
| `pd_centering` | steers to keep the left and right LiDAR beams equal | observation |
| `wall_follow` | holds a fixed distance from one wall, using distance and skew | observation |
| `follow_the_gap` | steers toward the longest free direction | observation |
| `pure_pursuit` | follows the centre line by the geometric steering law | pose and track |

Gains live in a frozen params record on each controller, and `run.py tune`
sweeps them. It ranks by whether a setting finishes the lap, then by lap time,
then by how far it got, because finishing matters more than the score along
the way. Tuning moved every one of them a long way: pd_centering went from
24.9 s a lap to 13.2 s, and pure pursuit from 22.8 s to 13.6 s, mostly by
raising the target speed once the steering was stiff enough to hold a line.

Scored over 5 episodes on `loop_corridor`, after tuning:

| controller | mean return | mean lap | lap time | laps |
|---|---|---|---|---|
| `pd_centering` | +16.29 | 1.001 | 13.3 s | 5/5 |
| `wall_follow` | +16.30 | 1.003 | 13.3 s | 5/5 |
| `follow_the_gap` | +16.30 | 1.002 | 14.4 s | 5/5 |
| `pure_pursuit` | +16.30 | 1.003 | 13.6 s | 5/5 |

All four finish every lap, and they land within a second of each other.
Knowing the centre line turns out not to help much on this track: the corridor
is 1 m wide and every controller ends up limited by the same thing, how early
it slows for a corner. The LiDAR ones only learn about a corner when a beam
reaches it, which costs them nothing here because the corners are gentle. A
faster track, or a shorter lookahead, should separate them.

Not here: MPC, which needs a model of the car and a solver inside the control
loop, and SLAM, which solves a problem this simulator does not pose, since the
track is known and the pose is exact.

## The panel

`panel.py` serves a page on 127.0.0.1 that collects the rest of this folder:
browse and preview tracks including generated ones, start a training run and
watch its learning curve fill in, then compare saved runs and play a video of
one. It uses only the standard library, and training runs inside the server
process so the curve moves while it trains.

Driving a track by hand belongs to the simulator. The panel writes a track to
an XML file and gives you the command, because `examples/manual_drive.py` in
neoracer-mujoco already takes any track with `--track` and needs the main
thread for its window.

## Still open

- A duel runs on `loop_corridor` only. The environment can now train on
  generated tracks through `GeneratedTracks`, but `multicar.py` places its
  cars using the one spawn offset that track needs, and a generated loop would
  need its start line read off the centre line.
- The opponent drives its own race and never reacts to the agent. Blocking,
  overtaking and defending are the interesting part and none of them exist
  yet.
- The state has no memory and no idea where on the lap the car is, so two
  corners that look alike are the same state. That aliasing is why these
  methods level off on this track.
