"""
Two cars on one track: the agent learns, the opponent is already driving.

DuelEnv is a Gymnasium environment that puts two NeoRacers on loop_corridor
and hands one of them to a learning algorithm. The other is driven by whatever
you pass as the opponent: one of the classical controllers in baselines.py, or
a table that tabular.py saved earlier.

The agent's observation is the same 20 numbers NeoRacerEnv gives, in the same
order, so a policy trained alone can race without changing a line. It sees the
other car only the way the real car would, as a shorter LiDAR beam. Its reward
and its terminations come from the same public functions NeoRacerEnv uses, so
a return here means what a return there means.

    env = DuelEnv(opponent=baselines.PDCentering())
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(action)

    env = DuelEnv(opponent=from_table("runs/sarsa.npz"))   # race a learned policy

What the opponent cannot do is block on purpose. It drives its own race and
the agent has to get past it, which is the point: the track stops being empty
without the opponent needing a theory of the agent.
"""

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from neoracer_rl.envs import NeoRacerEnvConfig
from neoracer_mujoco import sim
from neoracer_rl.envs.neoracer_env import (
    _observation_space,
    advance,
    apply_spawn_offset,
    angle_around_center,
    is_stuck,
    reward,
    start_progress,
    termination,
)

import baselines
import multicar
import tabular
import tracks

AGENT = "a"
OPPONENT = "b"


@dataclass(frozen=True)
class DuelConfig:
    """How a duel is set up. The parts that decide reward and termination come
    from NeoRacerEnvConfig, so they match the single car environment."""

    track: str = "loop_corridor"
    abreast: bool = False  # side by side, rather than one behind the other
    agent_leads: bool = False  # put the agent in front at the start
    env: NeoRacerEnvConfig = field(default_factory=NeoRacerEnvConfig)


def from_table(path: str):
    """An opponent that drives a table saved by tabular.py."""
    table = np.load(path)["Q"]

    class TablePolicy:
        is_privileged = False
        name = f"table:{path}"

        def reset(self) -> None:
            return None

        def act(self, obs, dt=1 / 30, pose=None):
            return tabular.ACTIONS[int(np.argmax(table[tabular.discretize(obs)]))]

    return TablePolicy()


class DuelEnv(gym.Env):
    """One learning car and one already driving, on the same track."""

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, opponent=None, config: DuelConfig = DuelConfig()) -> None:
        super().__init__()
        self.config = config
        self.opponent = opponent if opponent is not None else baselines.PDCentering()

        order = (AGENT, OPPONENT) if config.agent_leads else (OPPONENT, AGENT)
        self.scene = multicar.build(config.track, order, abreast=config.abreast)
        self.model = self.scene.model
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = _observation_space(config.env.obs_bounds)

        self._dt = 1.0 / config.env.control_hz
        self._substeps = max(1, round(self._dt / self.model.opt.timestep))
        self._window = max(1, round(config.env.termination.stuck_window_s * config.env.control_hz))
        self._needs_pose = getattr(self.opponent, "is_privileged", False)
        if self._needs_pose and config.track != "loop_corridor":
            raise ValueError(
                f"{self.opponent.name} follows a centre line, which this folder "
                f"only has for loop_corridor and generated tracks, not {config.track!r}"
            )

    def _angle(self, prefix: str) -> float:
        return float(
            angle_around_center(self.model, self.scene.cars[prefix], self.data.sensordata)
        )

    def _upright_cos(self, prefix: str) -> float:
        zaxis = self.data.bind_scalar(self.scene.cars[prefix].truth.zaxis).data
        return float(zaxis[2])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Let the suspension settle, then nudge each car inside the spawn box,
        # the same two steps NeoRacerEnv takes. Without the nudge every episode
        # of a duel is the same run, because nothing else here is random.
        sim.apply_ctrl_and_step(self.model, self.data, 0.0, sim.SETTLE_STEPS)
        for prefix in (AGENT, OPPONENT):
            apply_spawn_offset(
                self.model,
                self.data,
                self.scene.cars[prefix],
                self.config.env.spawn,
                self.np_random,
            )
        mujoco.mj_forward(self.model, self.data)
        self.opponent.reset()
        self._steps = 0
        self._progress = {p: start_progress(self._angle(p)) for p in (AGENT, OPPONENT)}
        obs = multicar.observe(self.scene, self.data, AGENT)
        return obs, self._info(None)

    def _info(self, reason):
        return {
            "lap_frac": self._progress[AGENT].total / (2 * np.pi),
            "opponent_lap_frac": self._progress[OPPONENT].total / (2 * np.pi),
            "gap_m": multicar.gap_between(self.scene, self.data, AGENT, OPPONENT),
            "ahead": self._progress[AGENT].total > self._progress[OPPONENT].total,
            "termination_reason": reason,
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        multicar.drive(
            self.scene, self.data, AGENT,
            float(action[0]) * self.config.env.throttle_scale, float(action[1]),
        )

        opp_obs = multicar.observe(self.scene, self.data, OPPONENT)
        opp_pose = multicar.pose(self.scene, self.data, OPPONENT) if self._needs_pose else None
        opp_throttle, opp_steer = self.opponent.act(opp_obs, self._dt, opp_pose)
        multicar.drive(
            self.scene, self.data, OPPONENT,
            opp_throttle * self.config.env.throttle_scale, opp_steer,
        )

        self.data.qacc_warmstart[:] = 0
        hit_wall = False
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
            if multicar.touching_wall(self.scene, self.data, AGENT):
                hit_wall = True
        self._steps += 1

        for p in (AGENT, OPPONENT):
            self._progress[p] = advance(self._progress[p], self._angle(p), self._window)
        mine = self._progress[AGENT]

        stuck = is_stuck(
            mine, self._window, self.config.env.termination.stuck_progress_rad
        )
        reason = termination(
            self.config.env.termination,
            self._upright_cos(AGENT),
            hit_wall,
            stuck,
            mine.total,
        )
        r = reward(self.config.env.reward, mine.delta, reason)
        truncated = reason is None and self._steps >= self.config.env.max_episode_steps
        obs = multicar.observe(self.scene, self.data, AGENT)
        return obs, r, reason is not None, truncated, self._info(reason)

    def close(self) -> None:
        return None


def race(opponent, policy=None, episodes: int = 3, config: DuelConfig = DuelConfig(), seed_base: int = 1000):
    """Run a few duels and report how the agent did against this opponent.

    policy is a function from an observation to an action. With none, the
    agent sits still, which is the floor to measure anything else against.
    """
    env = DuelEnv(opponent=opponent, config=config)
    rows = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        total, done = 0.0, False
        while not done:
            action = policy(obs) if policy is not None else np.zeros(2, dtype=np.float32)
            obs, r, terminated, truncated, info = env.step(action)
            total += r
            done = terminated or truncated
        rows.append((total, info["lap_frac"], info["opponent_lap_frac"], info["termination_reason"]))
    env.close()
    return rows


def make_opponent(name: str, track: str = "loop_corridor"):
    """Build a controller by name, or load a table saved by tabular.py."""
    match name:
        case "pd_centering":
            return baselines.PDCentering()
        case "wall_follow":
            return baselines.WallFollow()
        case "follow_the_gap":
            return baselines.FollowTheGap()
        case "pure_pursuit":
            return baselines.PurePursuit(tracks.centreline(track))
        case path:
            return from_table(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--opponent",
        default="pd_centering",
        help="pd_centering, wall_follow, follow_the_gap, pure_pursuit, "
        "or the path of a table from tabular.py",
    )
    parser.add_argument("--agent", default=None, help="a table for the agent to drive")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--abreast", action="store_true", help="start side by side")
    args = parser.parse_args()

    config = DuelConfig(abreast=args.abreast)
    opponent = make_opponent(args.opponent, config.track)
    policy = None
    if args.agent is not None:
        table = np.load(args.agent)["Q"]
        policy = lambda obs: np.array(  # noqa: E731
            tabular.ACTIONS[int(np.argmax(table[tabular.discretize(obs)]))],
            dtype=np.float32,
        )

    print(f"agent {args.agent or 'standing still'} against {opponent.name}")
    for ep, (total, mine, theirs, reason) in enumerate(
        race(opponent, policy, args.episodes, config)
    ):
        print(
            f"  episode {ep}: return {total:+7.2f}  agent lap {mine:.3f}  "
            f"opponent lap {theirs:.3f}  ended {reason or 'truncated'}"
        )
