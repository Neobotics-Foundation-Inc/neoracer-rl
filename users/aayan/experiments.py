"""
Every run of EXPERIMENTS.md: train the seeds, compare the settings, print the
tables. Run it from the repository root:

    python3 users/aayan/experiments.py runs

runs/ gets each network and the log of its training (runs/lr3e-4_s0.npz,
runs/lr3e-4_s0.log). A seed whose network is already there does not train
again, so a stopped run goes on where it stopped; delete a file to run that
seed again. Five seeds of three learning rates take about three hours when
five train at a time, and about eleven minutes each.

The question this answers is whether the learning rate matters. Everything
else is held still and only that moves, because deep Q-learning varies more
between seeds than most of the changes worth testing. Five seeds a setting is
the fewest that says anything, and the tables report the spread, not just the
mean.

Each network saved is the best its run evaluated, not the last. Why that
matters is in dqn.py, and the gap between the two columns of the first table
below is the reason.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


SEEDS = 5
STEPS = 300_000
EVAL_EVERY = 25_000
BOOTSTRAP = 10_000
PERMUTATIONS = 10_000
RNG = np.random.default_rng(0)


@dataclass(frozen=True)
class Setting:
    """One value of the swept hyperparameter, and what to call it."""

    name: str
    """Short name, used for the files and the tables."""
    flags: tuple[str, ...]
    """What to pass dqn.py on top of the defaults."""


SETTINGS = (
    Setting("lr3e-4", ()),  # the default
    Setting("lr5e-4", ("--learning-rate", "0.0005")),
    Setting("lr1e-3", ("--learning-rate", "0.001")),
)


def network(out: Path, setting: Setting, seed: int) -> Path:
    return out / f"{setting.name}_s{seed}.npz"


def train(out: Path, setting: Setting, seed: int) -> None:
    """Train one seed, unless its network is already there."""
    path = network(out, setting, seed)
    if path.exists():
        print(f"  {path.name} is already there")
        return
    log = path.with_suffix(".log")
    command = [
        sys.executable, str(HERE / "dqn.py"),
        "--steps", str(STEPS), "--seed", str(seed),
        "--eval-every", str(EVAL_EVERY),
        "--name", path.stem,
        *setting.flags,
    ]
    started = time.time()
    with log.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    print(f"  {path.name} in {(time.time() - started) / 60:.1f} min")


def read(out: Path, setting: Setting, seed: int) -> dict | None:
    """What one trained seed came to, or None if it has not trained."""
    path = network(out, setting, seed)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        settings = json.loads(str(data["settings"]))
        evals = data["evals"]
    return {
        "seed": seed,
        "best_lap": settings["best_lap"],
        "best_step": settings["best_step"],
        "best_laps_done": settings["best_laps_done"],
        "final_lap": settings["final_lap"],
        "peak_step": int(evals[int(np.argmax(evals[:, 2])), 0]) if len(evals) else 0,
    }


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """The mean, and a bootstrap 95% interval on it."""
    if not values:
        return 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=float)
    draws = RNG.choice(array, size=(BOOTSTRAP, array.size), replace=True).mean(axis=1)
    return float(array.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def compare(a: list[float], b: list[float]) -> dict:
    """b against a: the difference of means, an interval on it, and a p-value.

    The p-value comes from a permutation test, which suits five seeds a side
    better than a t-test does, since nothing here is normal and the samples
    are small.
    """
    if len(a) < 2 or len(b) < 2:
        return {}
    first, second = np.asarray(a, float), np.asarray(b, float)
    diff = float(second.mean() - first.mean())
    pooled = np.concatenate([first, second])
    draws = np.array([
        RNG.choice(second, second.size).mean() - RNG.choice(first, first.size).mean()
        for _ in range(1000)
    ])
    counts = 0
    for _ in range(PERMUTATIONS):
        shuffled = RNG.permutation(pooled)
        if abs(shuffled[: first.size].mean() - shuffled[first.size :].mean()) >= abs(diff):
            counts += 1
    spread = np.sqrt((first.var(ddof=1) + second.var(ddof=1)) / 2) or 1e-9
    return {
        "diff": diff,
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "p": counts / PERMUTATIONS,
        "d": diff / float(spread),
    }


def tables(out: Path) -> str:
    """The tables of EXPERIMENTS.md, from whatever has trained."""
    rows = {s.name: [r for seed in range(SEEDS) if (r := read(out, s, seed))]
            for s in SETTINGS}
    lines = ["```",
             f"{'setting':<10}{'seeds':>7}{'best lap':>22}{'last lap':>10}"
             f"{'reached a lap':>15}{'peaked at':>11}",
             ""]
    for s in SETTINGS:
        got = rows[s.name]
        if not got:
            lines.append(f"{s.name:<10}{'none yet':>7}")
            continue
        mean, lo, hi = mean_ci([r["best_lap"] for r in got])
        last = float(np.mean([r["final_lap"] for r in got]))
        lapped = sum(1 for r in got if r["best_laps_done"])
        peak = int(np.median([r["peak_step"] for r in got]))
        best = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        reached = f"{lapped} of {len(got)}"
        lines.append(
            f"{s.name:<10}{len(got):>7}{best:>22}{last:>10.3f}"
            f"{reached:>15}{f'{peak // 1000}k':>11}"
        )
    lines.append("```")

    base = SETTINGS[0]
    lines += ["", f"Against `{base.name}`, on the best lap fraction:", "", "```",
              f"{'setting':<10}{'difference':>12}{'95% interval':>22}{'p':>8}{'d':>8}", ""]
    first = [r["best_lap"] for r in rows[base.name]]
    for s in SETTINGS[1:]:
        c = compare(first, [r["best_lap"] for r in rows[s.name]])
        if not c:
            lines.append(f"{s.name:<10}{'needs 2 seeds a side':>12}")
            continue
        interval = f"[{c['lo']:+.3f}, {c['hi']:+.3f}]"
        lines.append(
            f"{s.name:<10}{c['diff']:>+12.3f}{interval:>22}"
            f"{c['p']:>8.3f}{c['d']:>+8.2f}"
        )
    lines.append("```")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", nargs="?", default=str(HERE / "runs"),
                        help="where the networks and logs go")
    parser.add_argument("--tables-only", action="store_true",
                        help="print the tables from what has trained, and train nothing")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.tables_only:
        for setting, seed in itertools.product(SETTINGS, range(SEEDS)):
            train(out, setting, seed)
    print()
    print(tables(out))


if __name__ == "__main__":
    main()
