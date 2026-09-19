# SAP-AI

[![CI](https://github.com/andrey-rublev/SAP-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/andrey-rublev/SAP-AI/actions/workflows/ci.yml)

A reinforcement-learning bot for **Super Auto Pets** (the Steam game).

It learns to shop, build a team, and win battles — first in a fast headless
simulation, and eventually by driving the real Steam client. Super Auto Pets
runs only on the Steam desktop app (there's no web version), so the live-play
layer uses screen capture + OCR + mouse control rather than a browser.

## How it fits together

| File | Role |
| --- | --- |
| [`game.py`](game.py) | `SuperAutoPetsEnv` — a Gymnasium environment (gold, shop, team, battles) |
| [`agent.py`](agent.py) | `QLearningAgent` — tabular Q-learning |
| [`dqn.py`](dqn.py) | `DQNAgent` — a PyTorch DQN that learns from the full observation |
| [`heuristic.py`](heuristic.py) | `HeuristicAgent` — a rule-based baseline (wins ~100% of runs) |
| [`train.py`](train.py) | Trains against the real rules via the `sapai` simulator |
| [`play.py`](play.py) | Runs a policy through the game loop (simulator `--dry-run` or live client) |
| [`autogui.py`](autogui.py) | Drives the actual Steam client (screen capture + OCR + mouse) |
| [`main.py`](main.py) | Dependency-free demo: trains on the built-in env and reports win rate |
| [`metrics.py`](metrics.py) | Training-curve plotting |

## Quickstart

```bash
pip install -r requirements.txt

python main.py                       # train tabular Q-learning on the built-in env
python main.py --episodes 8000 --plot curve.png
python dqn.py                        # train the neural (DQN) agent
python heuristic.py                  # evaluate the rule-based baseline
python play.py --dry-run             # watch a policy play the simulator
python train.py                      # train vs. the sapai rules engine (pip install sapai)
```

## Results (built-in env)

| Policy | run-win rate | avg wins/run |
| --- | --- | --- |
| Random | ~5% | ~3.0 |
| Tabular Q-learning | ~54% | ~9.1 |
| DQN (neural) | ~99% | ~10.0 |
| Heuristic (rule-based) | ~100% | ~10 |

The heuristic shows the arena is solvable with good play; with **action masking**
(the env marks illegal moves in `info['action_mask']` and the agents skip them)
the DQN essentially matches it. (Numbers vary run to run.)

Two mechanics did most of the work: a small per-step penalty on shop actions so
agents can't dither, and action masking so they never waste a turn on an illegal
move — masking alone roughly doubled the tabular agent and took the DQN from a
collapsed 0% to ~99%.

## Tests

```bash
python -m pytest          # 57 tests across the env, agents, and runners
```

## Live play

Super Auto Pets is a Steam desktop game, so live play goes through
[`autogui.py`](autogui.py). Calibrate the screen coordinates in its `LAYOUT` for
your resolution first (run `python autogui.py` to see what the OCR reads). Full
board perception (reading every shop/team pet from the screen) is still a work
in progress, so live play currently reads only gold/hearts.
