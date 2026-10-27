"""
Laws for examples/train_race.py: touching the other car ends a race as a
crash does, and an evaluation reports the first episode of each race.
"""

import jax
import numpy as np

from examples.train_ppo import PPOParams, Trained, empty_obs_stats, init_actor_critic
from examples.train_race import RaceReward, evaluate_races, race_config
from neoracer_rl.envs.neoracer_batch import Cpu, NeoRacerRace


def test_touching_the_other_car_ends_the_race_with_the_crash_penalty():
    """A learner at full speed, its opponent 0.6 m ahead, touches it before
    any wall. The race ends as a touch, and the step's reward is the
    progress less the touch penalty."""
    config = race_config("track-centering", 0.6)
    env = NeoRacerRace(Cpu(nthread=0), 1, config)
    state = env.reset(np.random.default_rng(0))
    full = np.array([[1.0, 0.0]], np.float32)
    state, out = env.step(state, full)
    while not out.done[0]:
        state, out = env.step(state, full)
    env.close()
    assert out.ends[out.reason[0] - 1] == "touch"
    assert out.ends.index("touch") == out.ends.index("crash") + 1
    assert out.reward[0] < 0.1 - RaceReward().touch_penalty


def test_an_evaluation_reports_the_first_episode_of_each_race():
    """Each race gives one result, with the laps of both cars, and the
    learner's lead is its laps less the opponent's and the opponent's head
    start: below 0 for a learner that barely moves."""
    params = PPOParams(hidden=(8,))
    net = init_actor_critic(jax.random.key(0), params)
    policy = Trained(net, empty_obs_stats(), params)
    results = evaluate_races(policy, 3, 0, race_config("track-centering", 1.0), 0)
    assert len(results) == 3
    for r in results:
        assert r.lead < r.laps[0] - r.laps[1] < 0.0
