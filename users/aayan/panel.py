"""
The control panel: one page for everything in this folder.

    python3 users/aayan/panel.py            # then open the URL it prints
    python3 users/aayan/panel.py --port 9000
    python3 users/aayan/panel.py --no-open  # do not open a browser

What it gives you:

  Tracks    every hand-written track plus procedurally generated ones
            (seed + difficulty), each with a rendered preview. Drive any of
            them yourself with the arrow keys.
  Train     pick a method and watch the learning curve fill in live, with the
            car's path from the episode running right now.
  Runs      every saved table, its learning curves overlaid for comparison,
            and a video of it driving.
  Notes     what each method is, and the two measurements that mattered more
            than any of them.

Stdlib only -- http.server and threads, no web framework. Training runs in a
background thread inside this process and reports every episode, which is why
the curve moves while it trains rather than after.
"""

import argparse
import io
import json
import os
import pathlib
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from neoracer_mujoco import assets

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import record  # noqa: E402
import tabular  # noqa: E402
import tracks  # noqa: E402

RUNS = _HERE / "runs"
PREVIEWS = RUNS / "previews"

# MuJoCo's offscreen renderer is happy off the main thread but one context at a
# time is the safe assumption, so preview rendering is serialized.
_render_lock = threading.Lock()

# Everything the browser polls for. Guarded by _state_lock; the training thread
# is the only writer.
_state_lock = threading.Lock()
TRAIN: dict = {
    "running": False,
    "method": None,
    "episodes": 0,
    "curve": [],       # one entry per finished episode
    "path": [],        # car xy for the episode in progress
    "message": "",
    "error": None,
    "finished": False,
    "result": None,
}
_stop_flag = threading.Event()


class _Stopped(Exception):
    """Raised inside the training thread to unwind when Stop is pressed."""


def _unique(stem: str) -> str:
    """stem, or stem_2, stem_3... so a repeat run never overwrites an earlier one."""
    if not (RUNS / f"{stem}.npz").exists():
        return stem
    n = 2
    while (RUNS / f"{stem}_{n}.npz").exists():
        n += 1
    return f"{stem}_{n}"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _train_worker(method: str, episodes: int, alpha: float,
                  gamma: float, eps: float, seed: int, save: bool) -> None:
    live_path: list = []

    def on_step(env):
        if _stop_flag.is_set():
            raise _Stopped()
        # Subsample: a 600-step episode does not need 600 points to show a shape.
        if len(live_path) < 400:
            p = env.data.body("car").xpos
            live_path.append([round(float(p[0]), 3), round(float(p[1]), 3)])

    def log(ep, total_eps, ret, info, steps, eps_now):
        nonlocal live_path
        with _state_lock:
            TRAIN["curve"].append({
                "ep": ep,
                "ret": round(float(ret), 3),
                "lap": round(float(info.get("lap_frac", 0.0)), 4),
                "steps": int(steps),
                "eps": round(float(eps_now), 4),
                "outcome": info.get("termination_reason") or "truncated",
            })
            TRAIN["path"] = live_path
            TRAIN["message"] = f"episode {ep + 1} of {total_eps}"
        live_path = []
        if _stop_flag.is_set():
            raise _Stopped()

    try:
        args = dict(log=log, on_step=on_step)
        if method == "mc":
            Q, _ = tabular.train_mc(episodes, gamma, eps, 0.02, seed, **args)
        elif method == "dp":
            Q, _ = tabular.train_dp(episodes, gamma, seed, **args)
        else:
            Q, _ = tabular.train_td(method, episodes, alpha, gamma, eps, 0.02,
                                    seed, **args)

        with _state_lock:
            TRAIN["message"] = "evaluating greedily on the stock reward..."
        mean_ret, mean_lap, outcomes = tabular.evaluate(Q, 5, verbose=False)
        laps = sum(o == "lap_complete" for o in outcomes)

        saved_as = None
        if save:
            RUNS.mkdir(exist_ok=True)
            hist = [[c["ret"], c["lap"]] for c in TRAIN["curve"]]
            # Descriptive, unique name -- NOT "<method>.npz". The CLI writes that
            # one, and a quick experiment here must never clobber a long run from
            # the command line, and a quick experiment here must not take it.
            stem = f"{method}_{len(hist)}ep_s{seed}"
            saved_as = _unique(stem)
            np.savez(RUNS / f"{saved_as}.npz", Q=Q, history=np.array(hist))

        with _state_lock:
            TRAIN["result"] = {
                "method": method, "return": round(mean_ret, 2),
                "lap": round(mean_lap, 3), "laps": laps, "of": len(outcomes),
                "saved": saved_as,
            }
            TRAIN["message"] = (
                f"done — mean return {mean_ret:+.2f}, mean lap {mean_lap:.3f}, "
                f"{laps}/{len(outcomes)} laps completed"
                + (f", saved as {saved_as}" if saved_as else "")
            )
    except _Stopped:
        with _state_lock:
            TRAIN["message"] = "stopped"
    except Exception:
        with _state_lock:
            TRAIN["error"] = traceback.format_exc(limit=3)
            TRAIN["message"] = "failed — see the error below"
    finally:
        with _state_lock:
            TRAIN["running"] = False
            TRAIN["finished"] = True


def start_training(body: dict) -> dict:
    with _state_lock:
        if TRAIN["running"]:
            return {"ok": False, "error": "already training"}
        method = body.get("method", "sarsa")
        if method not in tabular.METHODS:
            return {"ok": False, "error": f"unknown method {method!r}"}
        episodes = max(1, min(5000, int(body.get("episodes", 400))))
        TRAIN.update(running=True, method=method, episodes=episodes, curve=[],
                     path=[], message="starting...", error=None, finished=False,
                     result=None)
    _stop_flag.clear()
    threading.Thread(
        target=_train_worker,
        args=(method, episodes,
              float(body.get("alpha", 0.1)), float(body.get("gamma", 0.99)),
              float(body.get("eps", 0.3)), int(body.get("seed", 0)),
              bool(body.get("save", True))),
        daemon=True,
    ).start()
    return {"ok": True}


# ---------------------------------------------------------------------------
# tracks and runs
# ---------------------------------------------------------------------------
def preview_png(track: str) -> bytes:
    """A rendered overhead PNG of `track`, cached on disk by name."""
    import imageio.v2 as imageio
    import mujoco

    PREVIEWS.mkdir(parents=True, exist_ok=True)
    cached = PREVIEWS / (track.replace(":", "_") + ".png")
    if cached.exists():
        return cached.read_bytes()

    with _render_lock:
        model = tracks.load(track)
        data = mujoco.MjData(model)
        for _ in range(300):
            mujoco.mj_step(model, data)
        cam = record._camera(model, chase=False)
        with mujoco.Renderer(model, height=record.HEIGHT, width=record.WIDTH) as r:
            r.update_scene(data, camera=cam)
            img = r.render()
    buf = io.BytesIO()
    imageio.imwrite(buf, img, format="PNG")
    cached.write_bytes(buf.getvalue())
    return buf.getvalue()


def runs_payload() -> list:
    out = []
    for npz in sorted(RUNS.glob("*.npz")):
        try:
            with np.load(npz) as z:
                hist = z["history"] if "history" in z else np.zeros((0, 2))
                rows = [{"ret": float(r), "lap": float(lap)} for r, lap in hist]
            out.append({
                "name": npz.stem,
                "episodes": len(rows),
                "curve": rows,
                "video": f"{npz.stem}.mp4" if (RUNS / f"{npz.stem}.mp4").exists() else None,
            })
        except Exception as exc:  # a half-written npz should not break the page
            out.append({"name": npz.stem, "episodes": 0, "curve": [],
                        "video": None, "error": str(exc)})
    return out


def evaluate_run(name: str, episodes: int) -> dict:
    path = RUNS / f"{name}.npz"
    if not path.exists():
        return {"ok": False, "error": f"no run named {name!r}"}
    with np.load(path) as z:
        Q = z["Q"]
    mean_ret, mean_lap, outcomes = tabular.evaluate(Q, episodes, verbose=False)
    return {"ok": True, "return": round(mean_ret, 2), "lap": round(mean_lap, 3),
            "laps": sum(o == "lap_complete" for o in outcomes),
            "of": len(outcomes), "outcomes": outcomes}


def record_run(name: str, chase: bool, seed: int) -> dict:
    path = RUNS / f"{name}.npz"
    if not path.exists():
        return {"ok": False, "error": f"no run named {name!r}"}
    with np.load(path) as z:
        Q = z["Q"]

    def policy(obs):
        return np.array(tabular.ACTIONS[int(np.argmax(Q[tabular.discretize(obs)]))],
                        dtype=np.float32)

    out = RUNS / f"{name}.mp4"
    with _render_lock:
        record.record(policy, out, seed, 30, chase, 70.0, name)
    return {"ok": True, "video": out.name}


def save_track(track: str) -> dict:
    """Write a track to an XML file and give back the command that drives it.

    Driving by hand belongs to the simulator, which ships examples/manual_drive.py
    and takes any track XML with --track. That script owns a glfw window on the
    main thread, so it cannot run inside this server, and it lives in the
    neoracer-mujoco checkout rather than in the installed package. Writing the
    XML and printing the command is the honest way to hand it over, and it works
    for a generated track as well as a shipped one.
    """
    import mujoco

    out_dir = RUNS / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (track.replace(":", "_") + ".xml")
    generated = tracks.parse_generated(track)
    spec = (
        tracks.generated_spec(*generated)
        if generated is not None
        else mujoco.MjSpec.from_file(
            os.path.join(assets.TRACK_DIR, f"{track}.xml")
        )
    )
    out.write_text(spec.to_xml())
    return {
        "ok": True,
        "path": str(out),
        "command": f"python3 -m examples.manual_drive --track {out}",
    }


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the console for training output
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        try:
            if path in ("/", "/index.html"):
                html = (_HERE / "panel.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if path == "/api/init":
                return self._json({
                    "tracks": tracks.shipped(),
                    "methods": list(tabular.METHODS),
                    "maxDifficulty": 3,
                    "states": tabular.N_STATES,
                    "actions": tabular.N_ACTIONS,
                    "throttles": list(tabular.THROTTLES),
                    "steers": list(tabular.STEERS),
                })
            if path == "/api/progress":
                with _state_lock:
                    since = int(params.get("since", 0))
                    return self._json({
                        "running": TRAIN["running"],
                        "method": TRAIN["method"],
                        "episodes": TRAIN["episodes"],
                        "message": TRAIN["message"],
                        "error": TRAIN["error"],
                        "finished": TRAIN["finished"],
                        "result": TRAIN["result"],
                        "total": len(TRAIN["curve"]),
                        "curve": TRAIN["curve"][since:],
                        "path": TRAIN["path"],
                    })
            if path == "/api/runs":
                return self._json({"runs": runs_payload()})
            if path == "/api/track.png":
                track = params.get("track", "loop").replace("%3A", ":")
                return self._send(200, preview_png(track), "image/png")
            if path.startswith("/runs/"):
                f = RUNS / path[len("/runs/"):]
                if f.exists() and f.is_file() and RUNS in f.resolve().parents:
                    ctype = "video/mp4" if f.suffix == ".mp4" else "application/octet-stream"
                    return self._send(200, f.read_bytes(), ctype)
                return self._json({"error": "not found"}, 404)
            return self._json({"error": "not found"}, 404)
        except Exception:
            return self._json({"error": traceback.format_exc(limit=3)}, 500)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"ok": False, "error": "bad json"}, 400)
        try:
            if self.path == "/api/train":
                return self._json(start_training(body))
            if self.path == "/api/stop":
                _stop_flag.set()
                return self._json({"ok": True})
            if self.path == "/api/track-xml":
                return self._json(save_track(body.get("track", "loop_corridor")))
            if self.path == "/api/record":
                return self._json(record_run(body.get("run", "sarsa"),
                                             bool(body.get("chase", False)),
                                             int(body.get("seed", 1000))))
            if self.path == "/api/evaluate":
                return self._json(evaluate_run(body.get("run", "sarsa"),
                                               int(body.get("episodes", 5))))
            return self._json({"error": "not found"}, 404)
        except Exception:
            return self._json({"ok": False, "error": traceback.format_exc(limit=3)}, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-open", dest="open_browser", action="store_false")
    a = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"NeoRacer panel -> {url}")
    print("ctrl-C to stop.")
    if a.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
