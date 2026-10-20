"""
Drive the NeoRacer with the tabular methods from Sutton and Barto chapters 4 to 6.

    python3 users/aayan/tabular.py sarsa               # train one method
    python3 users/aayan/tabular.py all --episodes 400  # train and compare all
    python3 users/aayan/tabular.py --eval runs/sarsa.npz

Each method is the algorithm box from the book, written out directly:

    dp         4.4  value iteration, on an MDP estimated from rollouts, since
                    the car's real dynamics are never given to us
    mc         5.4  on policy first visit Monte Carlo control, epsilon greedy
    sarsa      6.4  on policy temporal difference control
    qlearning  6.5  off policy temporal difference control
    esarsa     6.6  Expected Sarsa

Why this needs a discretizer

These chapters are tabular: they store one value for every state and action
pair. NeoRacerEnv gives a 20 number observation and takes a 2 number action, so
neither a row nor a column of that table exists until you invent them. That
choice, rather than the algorithms, is where the difficulty lives.

The state keeps three signals from the 8 beam LiDAR ring, which is the first
eight numbers of the observation, one beam every 45 degrees, with index 0 dead
ahead, 1 front left, 2 left, 6 right and 7 front right:

    lateral   the left beam minus the right beam, in metres. Am I off centre?
              5 bins.
    skew      the rear side beam minus the front side beam, left minus right,
              in metres. Am I pointed along the wall or into it? 5 bins.
    forward   the distance to the wall ahead, in metres. Is a corner coming?
              4 bins.

skew comes from the simulator's wall_following controller, which estimates
heading against a wall the same way. It earns its place. With only clearance
and speed in the state, and no heading signal at all, an agent cannot tell
running parallel to a wall from driving into it, and scores below a random
policy.

That gives 100 states against 15 actions, which is small enough that a few
hundred episodes fill the table. That is the point: these methods are meant to
be watched while they learn, not scaled up.

The state still throws a lot away. skew is a local heading against a nearby
wall, not a heading on the track, so the agent cannot tell a left bend from a
right one until the beams reach it. It has no memory and no idea where on the
lap it is, so two corners that look alike are the same state. That aliasing is
why these methods level off on this track.
"""

import argparse
import pathlib
import time

import numpy as np

from neoracer_rl.envs import NeoRacerEnv

import store

_HERE = pathlib.Path(__file__).resolve().parent
RUNS = _HERE / "runs"

# ---------------------------------------------------------------------------
# Actions: a 15-point grid over the env's continuous Box(2).
# throttle and steer are both normalized to [-1, 1]; +steer = left.
# No reverse -- it is never useful on a one-way ring and doubling the action
# count would just slow the table down.
# ---------------------------------------------------------------------------
# The drive motors take a torque in newton metres over ctrlrange -1 to 1, and
# the car is light. The simulator's run.py calls 0.08 N*m a cruise of about
# 1.5 m/s and 0.35 enough to flip the car at speed, and its wall follower
# cruises at 0.10. A grid that reaches 1.0 only teaches a policy to drive into
# the nearest wall, so these three stay in the cruising range.
THROTTLES = (0.05, 0.10, 0.18)
STEERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
ACTIONS = [(t, s) for t in THROTTLES for s in STEERS]
N_ACTIONS = len(ACTIONS)

# Bin edges. np.digitize returns the bin index for a value against these.
_LATERAL_EDGES = np.array([-0.30, -0.10, 0.10, 0.30])  # -> 5 bins
_SKEW_EDGES = np.array([-0.30, -0.08, 0.08, 0.30])  # -> 5 bins
_FORWARD_EDGES = np.array([0.40, 0.80, 1.60])  # -> 4 bins

N_LATERAL = len(_LATERAL_EDGES) + 1
N_SKEW = len(_SKEW_EDGES) + 1
N_FORWARD = len(_FORWARD_EDGES) + 1
N_STATES = N_LATERAL * N_SKEW * N_FORWARD

# A beam that hits nothing reports -1. Treat that as "far", not as "a wall
# 1 metre behind me", which is what the raw -1 would discretize to.
_FAR_M = 5.0


# ---------------------------------------------------------------------------
# Reward
#
# Earlier versions of the environment charged a contact_penalty for every step
# the car spent touching a wall, which at 0.1 per step and 30 Hz made one
# second of scraping cost half of what a whole lap paid. Training needed a
# gentler contact price to stop the agents learning to stand still.
#
# That is no longer true. Touching a wall now ends the episode with a one-time
# crash_penalty equal to the stuck penalty, so an episode that makes progress
# and then crashes already returns more than one that never moves. There is
# nothing left to reshape, and every method here trains on the environment's
# own reward.
# ---------------------------------------------------------------------------


def make_env() -> NeoRacerEnv:
    """The environment every method trains on and is scored on."""
    return NeoRacerEnv()


def features(obs: np.ndarray) -> tuple[float, float, float]:
    """The three raw signals the state is built from. Exposed for inspection.

    Beam indices follow sensors.LIDAR_BEAM_ORDER: 0=0deg (ahead), 1=45,
    2=90 (left), 3=135, 4=180, 5=225, 6=270 (right), 7=315.
    """
    beams = np.asarray(obs[:8], dtype=float).copy()
    beams[beams < 0.0] = _FAR_M  # -1 = no hit

    lateral = beams[2] - beams[6]  # +ve = more room on the left

    # wall_skew, straight out of control/wall_following.py: rear-side beam
    # minus front-side beam on the same side estimates the nose angle relative
    # to that wall. Taking left minus right makes it symmetric and signed:
    # +ve = nose swinging toward the left wall.
    skew_left = beams[3] - beams[1]  # 135deg - 45deg
    skew_right = beams[5] - beams[7]  # 225deg - 315deg
    skew = skew_left - skew_right

    return lateral, skew, beams[0]


def discretize(obs: np.ndarray) -> int:
    """20-dim continuous observation -> one integer state index in [0, N_STATES).

    Speed is deliberately NOT part of the state. With throttle capped in the
    cruising range the car barely varies in speed, so the bins were near-constant
    -- they tripled the table for nothing. Heading (skew) earns that room far
    better: without it the agent cannot tell "running parallel to the wall" from
    "driving into it", which is the whole task.
    """
    lateral, skew, forward = features(obs)
    i_lat = int(np.digitize(lateral, _LATERAL_EDGES))
    i_skw = int(np.digitize(skew, _SKEW_EDGES))
    i_fwd = int(np.digitize(forward, _FORWARD_EDGES))
    return (i_lat * N_SKEW + i_skw) * N_FORWARD + i_fwd


def _eps_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    """Pick greedily with prob 1-eps, uniformly at random otherwise (S&B 5.4)."""
    if rng.random() < eps:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(Q[s]))


def _run_episode(env, Q, eps, rng, learn=None, gamma=0.99, on_step=None,
                 reset_seed=None):
    """One episode. `learn` is a per-step callback; returns (return, info, steps).

    Keeping the rollout in one place means each algorithm below is only its own
    update rule, which is the part worth comparing.

    `on_step(env)` is an optional observer -- panel.py uses it to trace where the
    car actually went, so a live view can draw the path while training runs. It
    never influences the agent.
    """
    # Seeding only the FIRST reset of a run seeds the env's RNG stream for the
    # whole run: later resets draw from it in order. Without this the spawn
    # randomisation comes from OS entropy and two runs with the same --seed
    # diverge, which makes comparing methods meaningless.
    obs, _ = env.reset(seed=reset_seed)
    s = discretize(obs)
    a = _eps_greedy(Q, s, eps, rng)
    total, steps, done = 0.0, 0, False
    info: dict = {}
    traj = []

    while not done:
        obs2, r, terminated, truncated, info = env.step(np.array(ACTIONS[a], dtype=np.float32))
        s2 = discretize(obs2)
        a2 = _eps_greedy(Q, s2, eps, rng)
        done = bool(terminated or truncated)
        traj.append((s, a, float(r)))
        if learn is not None:
            learn(s, a, float(r), s2, a2, terminated)
        if on_step is not None:
            on_step(env)
        s, a = s2, a2
        total += float(r)
        steps += 1

    return total, info, steps, traj


# ---------------------------------------------------------------------------
# Chapter 6 -- temporal-difference control. The three differ by one term.
# ---------------------------------------------------------------------------
def train_td(method, episodes, alpha, gamma, eps0, eps_min, seed, log, on_step=None):
    rng = np.random.default_rng(seed)
    env = make_env()
    Q = np.zeros((N_STATES, N_ACTIONS))
    history = []

    for ep in range(episodes):
        # Linear decay to eps_min over the first 80% of training, then greedy-ish.
        eps = max(eps_min, eps0 * (1.0 - ep / (0.8 * episodes)))

        def learn(s, a, r, s2, a2, terminated, _Q=Q):
            if terminated:
                target = r  # no bootstrap past a terminal state (S&B 6.5)
            elif method == "sarsa":
                target = r + gamma * _Q[s2, a2]  # on-policy: the action actually taken
            elif method == "qlearning":
                target = r + gamma * _Q[s2].max()  # off-policy: the greedy action
            elif method == "esarsa":
                # Expectation under the eps-greedy policy (S&B 6.6).
                pi = np.full(N_ACTIONS, eps / N_ACTIONS)
                pi[int(np.argmax(_Q[s2]))] += 1.0 - eps
                target = r + gamma * float(pi @ _Q[s2])
            else:
                raise ValueError(method)
            _Q[s, a] += alpha * (target - _Q[s, a])

        total, info, steps, _ = _run_episode(env, Q, eps, rng, learn=learn, gamma=gamma,
                                             on_step=on_step,
                                             reset_seed=seed if ep == 0 else None)
        history.append((total, float(info.get("lap_frac", 0.0))))
        log(ep, episodes, total, info, steps, eps)

    env.close()
    return Q, history


# ---------------------------------------------------------------------------
# Chapter 5.4 -- on-policy first-visit Monte Carlo control.
# No bootstrapping: wait for the episode to end, then credit each first visit
# with the actual return that followed it.
# ---------------------------------------------------------------------------
def train_mc(episodes, gamma, eps0, eps_min, seed, log, on_step=None):
    rng = np.random.default_rng(seed)
    env = make_env()
    Q = np.zeros((N_STATES, N_ACTIONS))
    counts = np.zeros((N_STATES, N_ACTIONS))  # for the incremental sample average
    history = []

    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1.0 - ep / (0.8 * episodes)))
        total, info, steps, traj = _run_episode(env, Q, eps, rng, learn=None, gamma=gamma,
                                                on_step=on_step,
                                                reset_seed=seed if ep == 0 else None)

        # Walk backwards accumulating G, updating only first visits.
        seen = set()
        first = {}
        for t, (s, a, _r) in enumerate(traj):
            if (s, a) not in first:
                first[(s, a)] = t
        G = 0.0
        for t in range(len(traj) - 1, -1, -1):
            s, a, r = traj[t]
            G = r + gamma * G
            if first[(s, a)] == t and (s, a) not in seen:
                seen.add((s, a))
                counts[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / counts[s, a]

        history.append((total, float(info.get("lap_frac", 0.0))))
        log(ep, episodes, total, info, steps, eps)

    env.close()
    return Q, history


# ---------------------------------------------------------------------------
# Chapter 4.4 -- value iteration.
#
# The book's DP assumes p(s',r | s,a) is GIVEN. Here it is not: the car's
# dynamics are a MuJoCo simulation, not a table. So this does the honest
# approximation -- drive around collecting transitions, build the maximum-
# likelihood estimate of the MDP from the counts, then run textbook value
# iteration on that estimate. The policy is only as good as the model, and the
# model only covers state-action pairs the exploration actually visited.
# ---------------------------------------------------------------------------
def train_dp(episodes, gamma, seed, log, theta=1e-4, max_sweeps=2000, on_step=None):
    rng = np.random.default_rng(seed)
    env = make_env()

    trans = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # counts N(s,a,s')
    rsum = np.zeros((N_STATES, N_ACTIONS))
    rcount = np.zeros((N_STATES, N_ACTIONS))
    terminal = np.zeros((N_STATES, N_ACTIONS))
    Q_dummy = np.zeros((N_STATES, N_ACTIONS))
    history = []

    # --- collect: uniformly random actions, so coverage is as broad as we get
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        s = discretize(obs)
        total, done, steps = 0.0, False, 0
        info: dict = {}
        while not done:
            a = int(rng.integers(N_ACTIONS))
            obs2, r, terminated, truncated, info = env.step(
                np.array(ACTIONS[a], dtype=np.float32)
            )
            s2 = discretize(obs2)
            trans[s, a, s2] += 1
            rsum[s, a] += float(r)
            rcount[s, a] += 1
            if terminated:
                terminal[s, a] += 1
            if on_step is not None:
                on_step(env)
            s = s2
            total += float(r)
            steps += 1
            done = bool(terminated or truncated)
        history.append((total, float(info.get("lap_frac", 0.0))))
        log(ep, episodes, total, info, steps, 1.0)

    env.close()

    # --- estimate the MDP
    visited = rcount > 0
    P = np.zeros_like(trans)
    nz = trans.sum(axis=2, keepdims=True)
    np.divide(trans, nz, out=P, where=nz > 0)
    R = np.zeros((N_STATES, N_ACTIONS))
    np.divide(rsum, rcount, out=R, where=visited)
    # Fraction of times (s,a) ended the episode -- damps the bootstrap, so a
    # terminal transition is not credited with the value of its successor.
    cont = np.ones((N_STATES, N_ACTIONS))
    np.divide(rcount - terminal, rcount, out=cont, where=visited)

    coverage = visited.mean()
    print(
        f"  model built from {int(rcount.sum())} transitions; "
        f"{coverage:.1%} of the {N_STATES}x{N_ACTIONS} (s,a) pairs were ever visited"
    )

    # --- value iteration (S&B 4.4)
    V = np.zeros(N_STATES)
    for sweep in range(max_sweeps):
        Q = R + gamma * cont * (P @ V)
        Q[~visited] = -np.inf  # never recommend an action we have no data for
        V_new = np.where(np.isfinite(Q).any(axis=1), np.max(Q, axis=1, initial=-np.inf), 0.0)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            print(f"  value iteration converged after {sweep + 1} sweeps (delta={delta:.2e})")
            break
    else:
        print(f"  value iteration hit {max_sweeps} sweeps without converging (delta={delta:.2e})")

    Q = R + gamma * cont * (P @ V)
    Q[~visited] = -np.inf
    Q_dummy = np.where(np.isfinite(Q), Q, 0.0)
    return Q_dummy, history


# ---------------------------------------------------------------------------
def evaluate(Q: np.ndarray, episodes: int = 5, seed_base: int = 1000, verbose=True):
    """Greedy rollouts on the environment's own reward, so the numbers stay
    comparable with the random baseline in examples/random_actions.py."""
    env = NeoRacerEnv()
    returns, laps, outcomes = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total, done, steps = 0.0, False, 0
        info: dict = {}
        while not done:
            a = int(np.argmax(Q[discretize(obs)]))
            obs, r, terminated, truncated, info = env.step(
                np.array(ACTIONS[a], dtype=np.float32)
            )
            total += float(r)
            steps += 1
            done = bool(terminated or truncated)
        returns.append(total)
        laps.append(float(info.get("lap_frac", np.nan)))
        outcomes.append(info.get("termination_reason") or "truncated")
        if verbose:
            print(
                f"  episode {ep:2d}: steps={steps:5d}  return={total:+7.2f}  "
                f"outcome={str(outcomes[-1]):15s} lap_frac={laps[-1]:.3f}"
            )
    env.close()
    return float(np.mean(returns)), float(np.nanmean(laps)), outcomes


def _make_logger(every=20):
    def log(ep, total_eps, ret, info, steps, eps):
        if ep % every == 0 or ep == total_eps - 1:
            print(
                f"    ep {ep:4d}/{total_eps}  return={ret:+7.2f}  steps={steps:5d}  "
                f"lap={float(info.get('lap_frac', 0.0)):.3f}  "
                f"eps={eps:.2f}  {info.get('termination_reason') or 'truncated'}"
            )

    return log


METHODS = ("dp", "mc", "sarsa", "qlearning", "esarsa")


def train_one(method, args):
    log = _make_logger(args.log_every)
    t0 = time.time()
    print(f"\n=== {method} ===")
    if method == "mc":
        Q, hist = train_mc(args.episodes, args.gamma, args.eps, args.eps_min, args.seed,
                           log)
    elif method == "dp":
        Q, hist = train_dp(args.episodes, args.gamma, args.seed, log)
    else:
        Q, hist = train_td(method, args.episodes, args.alpha, args.gamma, args.eps,
                           args.eps_min, args.seed, log)
    dt = time.time() - t0
    settings = {
        "method": method,
        "episodes": args.episodes,
        "seed": args.seed,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "eps": args.eps,
        "eps_min": args.eps_min,
        "minutes": round(dt / 60, 2),
    }
    out = store.save_table(Q, np.array(hist), settings, name=args.name)
    print(f"  trained in {dt/60:.1f} min -> {out}")
    print("  greedy evaluation:")
    mean_ret, mean_lap, outcomes = evaluate(Q, args.eval_episodes)
    laps_done = sum(o == "lap_complete" for o in outcomes)
    print(
        f"  => mean return {mean_ret:+.2f}   mean lap {mean_lap:.3f}   "
        f"laps completed {laps_done}/{len(outcomes)}"
    )
    result = {"method": method, "return": round(mean_ret, 3), "lap": round(mean_lap, 4),
              "laps": laps_done, "n": len(outcomes), "minutes": round(dt / 60, 2)}
    store.log_result("train", {**settings, **result, "table": out.name})
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", nargs="?", default="sarsa", choices=(*METHODS, "all"))
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--eps-min", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval", metavar="RUN.npz", help="evaluate a saved table, don't train")
    ap.add_argument("--name", default=None,
                    help="name the saved table, instead of <method>_<episodes>ep_s<seed>")

    a = ap.parse_args()

    print(f"state space: {N_LATERAL} lateral x {N_SKEW} skew x {N_FORWARD} forward "
          f"= {N_STATES} states, {N_ACTIONS} actions "
          f"(throttles {THROTTLES}, steers {STEERS})")


    if a.eval:
        Q, settings = store.load_table(a.eval)
        if settings:
            print("trained with: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
        mean_ret, mean_lap, outcomes = evaluate(Q, a.eval_episodes)
        print(f"\nmean return {mean_ret:+.2f}   mean lap fraction {mean_lap:.3f}")
        store.log_result("eval", {"table": str(a.eval), "episodes": a.eval_episodes,
                                  "return": round(mean_ret, 3), "lap": round(mean_lap, 4),
                                  "laps": sum(o == "lap_complete" for o in outcomes)})
    elif a.method == "all":
        rows = [train_one(m, a) for m in METHODS]
        print("\n" + "=" * 72)
        print(f"{'method':<12}{'mean return':>14}{'mean lap':>12}{'laps':>10}{'minutes':>10}")
        print("-" * 72)
        for r in sorted(rows, key=lambda r: -r["lap"]):
            print(f"{r['method']:<12}{r['return']:>+14.2f}{r['lap']:>12.3f}"
                  f"{r['laps']:>7}/{r['n']}{r['minutes']:>10.1f}")
        print("random baseline: return -5 to -13, lap 0.01-0.18, 0 laps")
    else:
        train_one(a.method, a)
