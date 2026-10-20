"""
Where experiments go, so that running one never costs you the last.

Two things are kept, both under runs/, which git ignores:

    runs/<name>.npz        a trained table, with the settings that made it
    runs/results.jsonl     one line per run, appended, never rewritten

A table is named after what produced it, as in sarsa_800ep_s0.npz. If that
name is taken the next one becomes sarsa_800ep_s0_2.npz, so a repeat of the
same command adds a file instead of replacing one. Saving to a fixed name per
method, which is what this used to do, quietly destroyed a long run whenever a
short one followed it.

Every npz also carries a settings dictionary: the method, the episode count,
the seed, the step size and so on, plus the version of the two packages it ran
against. A table on its own cannot say where it came from, and these move fast
enough that it matters.

    path = save_table(Q, history, settings)
    Q, settings = load_table("runs/sarsa_800ep_s0.npz")
    log_result("duel", {"agent": "...", "opponent": "...", "lap": 1.0})
"""

import json
import pathlib
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np

RUNS = pathlib.Path(__file__).resolve().parent / "runs"
RESULTS = RUNS / "results.jsonl"


def versions() -> dict[str, str]:
    """The installed versions this run used. They change often."""
    out = {}
    for package in ("neoracer-rl", "neoracer-mujoco", "mujoco", "gymnasium"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not installed"
    return out


def unique(stem: str, suffix: str = ".npz") -> pathlib.Path:
    """runs/<stem><suffix>, or the next free _2, _3 and so on."""
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{stem}{suffix}"
    n = 2
    while path.exists():
        path = RUNS / f"{stem}_{n}{suffix}"
        n += 1
    return path


def table_name(settings: dict) -> str:
    """The descriptive stem for a table, from the settings that made it."""
    return (
        f"{settings.get('method', 'table')}"
        f"_{settings.get('episodes', 0)}ep"
        f"_s{settings.get('seed', 0)}"
    )


def save_table(Q: np.ndarray, history, settings: dict, name: str | None = None) -> pathlib.Path:
    """Write a table and its settings to a new file. Returns the path used."""
    full = {**settings, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **versions()}
    path = unique(name or table_name(settings))
    np.savez(
        path,
        Q=np.asarray(Q),
        history=np.asarray(history),
        settings=json.dumps(full),
    )
    return path


def load_table(path: str | pathlib.Path) -> tuple[np.ndarray, dict]:
    """A saved table and the settings that made it. Older files have none."""
    with np.load(path) as data:
        Q = data["Q"]
        raw = str(data["settings"]) if "settings" in data else "{}"
    return Q, json.loads(raw)


def load_settings(path: str | pathlib.Path) -> dict:
    """Just the settings of any saved run, whether it holds a table or a
    network. dqn.py writes weights rather than a Q, so this does not ask for
    one."""
    with np.load(path) as data:
        return json.loads(str(data["settings"])) if "settings" in data else {}


def log_result(kind: str, record: dict) -> None:
    """Append one run to runs/results.jsonl. Nothing is ever overwritten."""
    RUNS.mkdir(exist_ok=True)
    line = {
        "kind": kind,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **record,
        **versions(),
    }
    with RESULTS.open("a") as f:
        f.write(json.dumps(line) + "\n")


def results(kind: str | None = None) -> list[dict]:
    """Every run logged so far, newest last, optionally of one kind."""
    if not RESULTS.exists():
        return []
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


if __name__ == "__main__":
    rows = results()
    if not rows:
        print(f"nothing logged yet. {RESULTS} appears after the first run.")
    else:
        print(f"{len(rows)} runs in {RESULTS}:\n")
        for r in rows[-20:]:
            summary = " ".join(
                f"{k}={v}"
                for k, v in r.items()
                if k not in ("kind", "at", *versions())
            )
            print(f"  {r['at']}  {r['kind']:<6} {summary}")
    tables = sorted(RUNS.glob("*.npz"))
    print(f"\n{len(tables)} saved tables:")
    for path in tables:
        try:
            settings = load_settings(path)
            kind = settings.get("method", "?")
            made = settings.get("saved_at", "before settings were saved")
        except Exception as exc:
            kind, made = "?", f"unreadable: {exc}"
        print(f"  {path.name:<34} {kind:<6} {made}")
