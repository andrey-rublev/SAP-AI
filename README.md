# SAP-AI

A bot that plays **Super Auto Pets** on Windows (Steam). It watches the game
window, decides, clicks, and checks every click took effect — no browser, no
game hooks, just screen capture, OCR and the mouse.

```powershell
cd path\to\SAP-AI
.\play.cmd                                  # play one arena (opens the game via Steam)
.\play.cmd --arenas 6 --max-seconds 3600    # a batch; summary reports trophies per arena
.\play.cmd --check                          # verify calibration and OCR without touching the game
```

Stop it any time: hold **Escape**, slam the pointer into a screen corner, or
create `.local/desktop/STOP`. Setup and calibration: [docs/desktop.md](docs/desktop.md).

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

## Offline lab

A separate Gymnasium toy arena where pets are single numbers. It is useful for
RL experiments, but its policies are never used against the real game.

```bash
python -m pip install -e ".[dev,plot]"      # add ,dqn for PyTorch
python main.py --episodes 4000              # tabular Q-learning
python dqn.py --episodes 250                # Double DQN
python benchmark.py                         # compare policies on shared seeds
```

Design notes: [docs/architecture.md](docs/architecture.md).

## Tests

```bash
python -m pytest -q
```

Desktop IO is mocked, and recorded native sessions are replayed as fixtures.
Captures, calibration and logs stay private in `.local/` (Git-ignored).
