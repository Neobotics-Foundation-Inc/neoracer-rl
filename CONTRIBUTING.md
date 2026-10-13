# Contributing

Thank you for helping with the NeoRacer reinforcement learning code. This page explains how to make a change.
Coding agents read AGENTS.md instead.

## Before you push

- Run `ruff check .` and `ruff format .`. CI runs `ruff check .` and `ruff format --check .`.
- Run `ty check`. CI runs it too.
  ty needs the mujoco stubs in `typings/`.
  Make them with `pybind11-stubgen mujoco mujoco.viewer -o typings` after you install or update mujoco.
- Run `python -m pytest validation`. Put a new test in `validation/test_<module>.py`, never in the module it tests.
- Keep the tests green after each commit.
  If a change alters behavior, change its test in the same commit.
- `users/` is the place to experiment. Make your own folder, `users/<your-github-name>/`, and put anything there: scripts, notes, your own environments and training runs.
  ruff, ty, the tests and CI do not check `users/`, and nobody else changes your folder.
  Do not change the folder of another person.
  Code outside `users/` never imports from it. When an experiment is ready to share, copy it to `src/` or `examples/` in its own pull request and make it follow the rules there.
- The simulator itself (the car model, the tracks, the sensors) lives in [neoracer-mujoco](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco).
  If your change needs something new there, open a pull request in that repo first.
- Keep one change in one pull request.
