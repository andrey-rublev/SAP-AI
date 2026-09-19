# SAP-AI

A reinforcement-learning bot for **Super Auto Pets** (the Steam game).

It learns to shop, build a team, and win battles — first in a fast headless
simulation, and eventually by driving the real Steam client. Super Auto Pets
runs only on the Steam desktop app (there's no web version), so the live-play
layer uses screen capture + OCR + mouse control rather than a browser.

## How it fits together

| File | Role |
| --- | --- |
| [`game.py`](game.py) | `SuperAutoPetsEnv` — a lightweight Gymnasium environment (gold, shop, team, battles) |
| [`agent.py`](agent.py) | `QLearningAgent` — a tabular Q-learning agent |
| [`main.py`](main.py) | Dependency-free demo: trains on the built-in env and reports win rate |
| [`train.py`](train.py) | Trains against the real rules via the `sapai` simulator |
| [`autogui.py`](autogui.py) | Plays the actual Steam client (screen capture + OCR + mouse) |

## Quickstart

```bash
pip install -r requirements.txt
python main.py      # headless demo — needs only gymnasium + numpy
python train.py     # train vs. the sapai rules engine — needs: pip install sapai
```

Before letting `autogui.py` control the live game, calibrate the screen
coordinates in its `LAYOUT` for your resolution (run `python autogui.py` to see
what the OCR currently reads).
