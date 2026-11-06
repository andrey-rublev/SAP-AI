# SAP-AI

A bot that plays **Super Auto Pets** on Windows (Steam). It watches the game
window, decides, clicks, and checks every click took effect — no browser, no
game hooks, just screen capture, OCR and the mouse.

```powershell
cd path\to\SAP-AI
.\play.cmd              # autopilot: opens the game via Steam and keeps playing arenas
.\play.cmd --arenas 3   # stop after three completed arenas
.\play.cmd --strategy conservative  # previous shop policy
.\play.cmd --check      # verify calibration and OCR without touching the game
```

Autopilot is experimental. It can recover from stalls by observing again or
restarting the game, and Game Over starts the next arena. If you use the computer, it pauses
and resumes once you've been idle for a minute. On a display that isn't
2560x1440 it runs the game in a 16:9 window and scales its calibration.
Stop it any time: hold **Escape**, slam the pointer into a screen corner, or
create `.local/desktop/STOP`. Progress (trophies per arena) is written to
`.local/desktop/autopilot/<run>/summary.json`. Details: [docs/desktop.md](docs/desktop.md).

## How it plays

| Step | What happens |
| --- | --- |
| See | Calibrated templates read the phase, gold, turn, trophies, lives and each pet's attack/health/level. |
| Identify | Read unknown pet names from tooltips; retain known names after acknowledged purchases. |
| Decide | Economy strategy: fill the team, merge duplicates, replace weak pets, search with spare gold when a supported sale can still fund a purchase. |
| Act | Mouse input on the foreground game window only. |
| Verify | Each action waits for its specific effect (gold change, slot change, phase change); nothing is retried blindly. |

Before species identification and the merge-first strategy, a six-arena soak
run acknowledged 490/490 actions but averaged 1.0 trophies (no merges at all).
Trophies per arena are now the headline metric; see the [changelog](CHANGELOG.md).

`python -m sap_ai.sim.arena` scores policies offline in a simplified arena that
uses the bot's own board types (approximate stats, no abilities):

November 6 held-out comparison (500 paired runs per opponent):

| Opponent | Conservative trophies | Economy trophies | Paired gain (95% CI) |
| --- | ---: | ---: | --- |
| Recorded baseline | 4.05 | 4.91 | +0.86 [0.56, 1.17] |
| Conservative | 3.44 | 4.17 | +0.73 [0.45, 1.01] |
| Economy | 3.12 | 3.60 | +0.48 [0.22, 0.74] |

These are ability-free simulator results; live performance remains unmeasured.
[Report and reproduction](docs/strategy-evaluation.md).

## Offline lab

A Gymnasium toy arena where pets are single numbers, for RL experiments. Its
policies are never used against the real game; the arena simulator above is the
bridge for evaluating real-game policies offline.

```bash
python -m pip install -e ".[dev,plot]"          # add ,dqn for PyTorch
python -m sap_ai.sim.main --episodes 4000       # tabular Q-learning
python -m sap_ai.sim.dqn --episodes 250         # Double DQN
python -m sap_ai.sim.benchmark                  # compare toy-arena policies
python -m sap_ai.sim.policy_study --runs 500 --seed 20000  # desktop-policy comparison
```

Code lives in `sap_ai/desktop` (the bot) and `sap_ai/sim` (the lab).
Design notes: [docs/architecture.md](docs/architecture.md).

## Offline diagnostics

Review a saved run without opening SAP:

```bash
python -m sap_ai.desktop.audit --run-dir .local/desktop/autopilot/RUN --output reports/audit.json
```

The report flags missing readings and broken receipts, and includes unfinished
sessions. It reads JSON logs only. See [the foundation review](docs/foundation-review.md).

## Tests

```bash
python -m pytest -q
```

Desktop IO is mocked, and recorded native sessions are replayed as fixtures.
Captures, calibration and logs stay private in `.local/` (Git-ignored).
