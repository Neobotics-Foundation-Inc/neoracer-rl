"""
Every run of EXPERIMENTS.md: train the policies, evaluate them, and print the
tables. Run it from the repository root on a 4-core machine:

    python -m examples.experiments runs

runs/ gets each policy and the log of its training (runs/l1.npz, runs/l1.log)
and the evaluations (runs/evaluations.json). A step whose output is there
does not run again, so a stopped run goes on where it stopped; delete a file
to run its step again. On a 4-core 2.1 GHz Xeon the training takes about 1.5
hours and the evaluations about 40 minutes.

A run is named by its regimen and its seed: l for loop_corridor, d for
loop_corridor with random physics, g for 16 generated tracks. Seed 0 of each
regimen trains alone on 4 threads, for the time a run takes. The others
train 4 at a time on 1 thread each, and so does l0 again, as l0t1, which
must give l0's run update for update.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from neoracer_mujoco.physics import PhysicsRanges, Range

from examples.train_ppo import (
    EvalPolicy,
    PPOParams,
    RandomActions,
    Trained,
    ZeroAction,
    evaluate,
    load_policy,
    parse_log,
)
from neoracer_rl.envs import GeneratedTracks, LoopCorridor, NeoRacerEnvConfig
from neoracer_rl.envs.neoracer_env import Tracks

TOTAL_STEPS = 400_000
EVAL_SEED = 10_000
"""The env seed of the first evaluation episode."""
EPISODES = 20
"""Evaluation episodes on a track."""
SWEEP_EPISODES = 10
"""Evaluation episodes for each value of a physics factor."""
LAST_STEPS = 50_000
"""The span of training, at its end, over which the lap share is counted."""
CORES = 4
"""The cores that the runs share: seed 0 trains on all of them, the others
on one each, as many at a time."""


@dataclass(frozen=True)
class Regimen:
    name: str
    """As the tables print it."""
    prefix: str
    """The first letter of its runs' names."""
    flags: tuple[str, ...]
    """train_ppo's options for it."""
    seeds: int
    on_tracks: bool
    """Whether its policies are evaluated on TRACKS."""
    under_physics: bool
    """Whether its policies are evaluated under each value of SWEEP."""


REGIMENS = (
    Regimen("loop_corridor", "l", (), 5, on_tracks=True, under_physics=True),
    Regimen(
        "loop_corridor, random physics",
        "d",
        ("--random-physics",),
        5,
        on_tracks=False,
        under_physics=True,
    ),
    Regimen(
        "16 generated tracks",
        "g",
        ("--tracks", "generated"),
        3,
        on_tracks=True,
        under_physics=False,
    ),
)
TRACKS: dict[str, Tracks] = {
    "loop": LoopCorridor(),
    "pool": GeneratedTracks(),
    "held out": GeneratedTracks(first_seed=1000),
}
"""Where the policies are evaluated: loop_corridor, the 16 tracks of
--tracks generated, and 16 tracks from seed 1000 that no run trained on."""
SWEEP = {
    "wheel_friction": (0.3, 0.5, 0.75, 1.0, 1.2),
    "chassis_mass": (0.7, 1.5),
    "motor_torque": (0.5, 0.75, 1.5),
    "steer_stiffness": (0.5, 2.0),
}
"""The physics factors that the policies are evaluated under, one at a time."""


@dataclass(frozen=True)
class Run:
    name: str
    regimen: Regimen
    seed: int
    nthread: int


def runs() -> tuple[list[Run], list[Run]]:
    """The runs that train alone, on every core, and the runs that train
    together, on one core each."""
    alone = [Run(f"{r.prefix}0", r, 0, CORES) for r in REGIMENS]
    together = [Run("l0t1", REGIMENS[0], 0, 1)]
    together += [
        Run(f"{r.prefix}{s}", r, s, 1) for r in REGIMENS for s in range(1, r.seeds)
    ]
    return alone, together


def train(out: Path, run: Run) -> None:
    """Train run with train_ppo, its output in out/<name>.log, the time it
    took on the log's last line. The log appears only once it is whole, and a
    run with its log and its policy is not trained again."""
    log_path = out / f"{run.name}.log"
    if log_path.exists() and (out / f"{run.name}.npz").exists():
        return
    command = [
        sys.executable,
        *("-m", "examples.train_ppo"),
        *("--total-steps", str(TOTAL_STEPS)),
        *("--seed", str(run.seed)),
        *("--nthread", str(run.nthread)),
        *run.regimen.flags,
        *("--out", str(out / f"{run.name}.npz")),
    ]
    start = time.monotonic()
    part = out / f"{run.name}.log.part"
    with part.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"wall {time.monotonic() - start:.0f} s\n")
    part.replace(log_path)


type Job = tuple[str, str, tuple[str, float] | None, int]
"""A policy (a run's name, random or zero), the tracks, a physics factor and
its value or None, and the episodes."""


def jobs() -> Iterator[Job]:
    yield "random", "loop", None, EPISODES
    yield "zero", "loop", None, EPISODES
    for r in REGIMENS:
        if r.on_tracks:
            for s in range(r.seeds):
                for tracks in TRACKS:
                    yield f"{r.prefix}{s}", tracks, None, EPISODES
    for r in REGIMENS:
        if r.under_physics:
            for s in range(r.seeds):
                for factor, values in SWEEP.items():
                    for v in values:
                        yield f"{r.prefix}{s}", "loop", (factor, v), SWEEP_EPISODES


def key(job: Job) -> str:
    name, tracks, factor, _ = job
    return f"{name} {tracks}" + ("" if factor is None else f" {factor[0]}={factor[1]}")


def run_job(job: tuple[Path, Job]) -> tuple[str, dict[str, Any]]:
    """Evaluate one job's policy with its mean action, on env seeds from
    EVAL_SEED."""
    out, (name, tracks, factor, episodes) = job
    physics = PhysicsRanges()
    if factor is not None:
        physics = PhysicsRanges(**{factor[0]: Range(factor[1], factor[1])})
    config = NeoRacerEnvConfig(tracks=TRACKS[tracks], physics=physics)
    policy: EvalPolicy
    match name:
        case "random":
            policy = RandomActions()
        case "zero":
            policy = ZeroAction()
        case _:
            net, stats = load_policy(str(out / f"{name}.npz"))
            policy = Trained(net, stats, PPOParams())
    results = evaluate(policy, episodes, EVAL_SEED, np.random.default_rng(0), config)
    return key((name, tracks, factor, episodes)), {
        "laps": sum(r.outcome == "lap_complete" for r in results),
        "episodes": len(results),
        "mean_return": float(np.mean([r.episode_return for r in results])),
        "mean_lap_frac": float(np.mean([r.lap_frac for r in results])),
        "outcomes": [r.outcome for r in results],
    }


def evaluate_all(out: Path) -> dict[str, dict[str, Any]]:
    """Every job's evaluation, from out/evaluations.json once it is there. The
    file appears only once it is whole."""
    path = out / "evaluations.json"
    if not path.exists():
        with Pool(CORES) as pool:
            results = dict(pool.imap_unordered(run_job, [(out, j) for j in jobs()]))
        part = out / "evaluations.json.part"
        part.write_text(json.dumps(results, indent=1, sort_keys=True))
        part.replace(path)
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Training:
    """What a run's log says of its training."""

    first_lap: int | None
    """The env steps at the end of the first update with a lap."""
    lap_share: float
    """The share of the episodes of the last LAST_STEPS steps that lap."""
    steps_per_s: int
    """The median over the updates."""
    wall_s: int
    seeded: tuple[str, ...]
    """The update lines without their speed (LogLine.seeded)."""


def training(path: Path) -> Training:
    text = path.read_text().splitlines()
    rows = [line for line in map(parse_log, text) if line is not None]
    last = rows[-1].env_steps
    first = next((r.env_steps for r in rows if r.laps), None)
    tail = [r for r in rows if r.env_steps > last - LAST_STEPS]
    episodes = sum(r.episodes for r in tail)
    wall = next(int(line.split()[1]) for line in text if line.startswith("wall "))
    return Training(
        first_lap=first,
        lap_share=sum(r.laps for r in tail) / episodes if episodes else float("nan"),
        steps_per_s=int(statistics.median(r.steps_per_s for r in rows)),
        wall_s=wall,
        seeded=tuple(r.seeded for r in rows),
    )


def regimen_table(out: Path) -> str:
    header = f"{'regimen':32s} {'seeds that lap':16s} {'first lap':19s} "
    rows = [header + f"laps in training, last {LAST_STEPS // 1000}k steps"]
    for r in REGIMENS:
        runs = [training(out / f"{r.prefix}{s}.log") for s in range(r.seeds)]
        lapped = [(t.first_lap, t.lap_share) for t in runs if t.first_lap is not None]
        firsts = ", ".join(f"{round(first / 1000)}k" for first, _ in lapped)
        shares = ", ".join(f"{share:.0%}" for _, share in lapped)
        rows.append(
            f"{r.name:32s} {f'{len(lapped)} of {r.seeds}':16s} {firsts:19s} {shares}"
        )
    return "\n".join(rows)


def sweep_table(evaluations: dict[str, dict[str, Any]], r: Regimen) -> str:
    """The laps out of SWEEP_EPISODES of each of r's policies under each value
    of each factor of SWEEP."""
    rows = [f"{'':17s}seed" + "".join(f"{s:3d} " for s in range(r.seeds)).rstrip()]
    for factor, values in SWEEP.items():
        for k, v in enumerate(values):
            label = factor if k == 0 else ""
            laps = [
                evaluations[f"{r.prefix}{s} loop {factor}={v}"]["laps"]
                for s in range(r.seeds)
            ]
            rows.append(f"{label:16s}{v!s:4s} " + " ".join(f"{n:3d}" for n in laps))
    return "\n".join(rows)


def track_lines(evaluations: dict[str, dict[str, Any]], r: Regimen) -> str:
    """How each of r's policies does on each of TRACKS."""
    rows = []
    for name in (f"{r.prefix}{s}" for s in range(r.seeds)):
        for tracks in TRACKS:
            e = evaluations[f"{name} {tracks}"]
            outcomes = dict(Counter(e["outcomes"]))
            rows.append(
                f"{name:4s} {tracks:9s} {e['laps']:2d} of {e['episodes']}  "
                f"return {e['mean_return']:+6.2f}  lap {e['mean_lap_frac']:+.2f}  "
                f"{outcomes}"
            )
    return "\n".join(rows)


def report(out: Path, evaluations: dict[str, dict[str, Any]]) -> str:
    alone, _ = runs()
    timing = [training(out / f"{run.name}.log") for run in alone]
    same = training(out / "l0.log").seeded == training(out / "l0t1.log").seeded
    baselines = [
        f"{name}: return {evaluations[f'{name} loop']['mean_return']:+.2f}"
        for name in ("random", "zero")
    ]
    sweeps = [
        f"Trained on {r.name}:\n" + sweep_table(evaluations, r)
        for r in REGIMENS
        if r.under_physics
    ]
    tracks = [track_lines(evaluations, r) for r in REGIMENS if r.on_tracks]
    return "\n\n".join(
        [
            "## Which regimens learn to lap\n\n" + regimen_table(out),
            f"Seed 0 alone on {CORES} threads: "
            + "; ".join(
                f"{run.name} {t.wall_s / 60:.1f} min at {t.steps_per_s} steps/s"
                for run, t in zip(alone, timing, strict=True)
            )
            + f". l0 on 1 thread gives l0's run update for update: {same}.",
            "Baselines on loop_corridor: " + ", ".join(baselines) + ".",
            f"## Physics, laps out of {SWEEP_EPISODES}\n\n" + "\n\n".join(sweeps),
            "## Tracks\n\n" + "\n".join(tracks),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the folder of the runs")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    alone, together = runs()
    for run in alone:
        train(out, run)
    with ThreadPoolExecutor(CORES) as threads:
        list(threads.map(lambda run: train(out, run), together))
    print(report(out, evaluate_all(out)))


if __name__ == "__main__":
    main()
