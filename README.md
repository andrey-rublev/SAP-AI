# SAP-AI

A bot that plays **Super Auto Pets** on Windows (Steam). It watches the game
window, decides, clicks, and checks every click took effect — no browser, no
game hooks, just screen capture, OCR and the mouse.

```powershell
cd path\to\SAP-AI
.\play.cmd              # autopilot: opens the game via Steam and keeps playing arenas
.\play.cmd --arenas 3   # stop after three completed arenas
.\play.cmd --check      # verify calibration and OCR without touching the game
```

It runs unattended. Stalls are recovered by observing again or restarting the
game, and Game Over starts the next arena. If you use the computer, it pauses
and resumes once you've been idle for a minute. On a display that isn't
2560x1440 it runs the game in a 16:9 window and scales its calibration.
Stop it any time: hold **Escape**, slam the pointer into a screen corner, or
create `.local/desktop/STOP`. Progress (trophies per arena) is written to
`.local/desktop/autopilot/<run>/summary.json`. Details: [docs/desktop.md](docs/desktop.md).

## How it plays

| Step | What happens |
| --- | --- |
| See | Calibrated templates read the phase, gold, turn, trophies, lives and each pet's attack/health/level. |
| Identify | Unknown pets are hovered once; their tooltip title is read against a fixed list of pet names. |
| Decide | `DesktopPolicy`: merge duplicates, fill empty slots, swap weak pets for stronger offers, roll, end turn. |
| Act | Mouse input on the foreground game window only. |
| Verify | Each action waits for its specific effect (gold change, slot change, phase change); nothing is retried blindly. |

Before species identification and the merge-first strategy, a six-arena soak
run acknowledged 490/490 actions but averaged 1.0 trophies (no merges at all).
Trophies per arena are now the headline metric; see the [changelog](CHANGELOG.md).

`python -m sap_ai.sim.arena` scores policies offline in a simplified arena that
uses the bot's own board types (approximate stats, no abilities):

| Policy (1,000 runs) | Mean trophies | Wins |
| --- | ---: | ---: |
| Current strategy | 4.14 | 234 |
| Previous strategy | 3.92 | 194 |
| Previous strategy, species unknown | 1.77 | 0 |

## Offline lab

A Gymnasium toy arena where pets are single numbers, for RL experiments. Its
policies are never used against the real game; the arena simulator above is the
bridge for evaluating real-game policies offline.

```bash
python -m pip install -e ".[dev,plot]"          # add ,dqn for PyTorch
python -m sap_ai.sim.main --episodes 4000       # tabular Q-learning
python -m sap_ai.sim.dqn --episodes 250         # Double DQN
python -m sap_ai.sim.benchmark                  # compare policies on shared seeds
```

Code lives in `sap_ai/desktop` (the bot) and `sap_ai/sim` (the lab).
Design notes: [docs/architecture.md](docs/architecture.md).

## Tests

```bash
python -m pytest -q
```

Desktop IO is mocked, and recorded native sessions are replayed as fixtures.
Captures, calibration and logs stay private in `.local/` (Git-ignored).
