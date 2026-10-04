# SAP-AI

A small reinforcement-learning laboratory inspired by **Super Auto Pets**.
Train and compare policies locally, save resumable checkpoints, and inspect
reproducible episode reports. Python 3.10+.

**The built-in arena is a toy model.** Pets are scalar strengths; combat compares
team totals against a growing opponent. It does not model species, health,
abilities, positioning, food, or the current game's complete rules. Success in
this arena is not evidence of real-game performance.

## Start here

```bash
python -m pip install -e ".[dev,plot]"
python play.py                         # watch the improved heuristic offline
python main.py --episodes 4000 --report reports/tabular.json
python benchmark.py --episodes 4000 --eval-episodes 300 --output reports/benchmark.json
python -m pytest -q
```

For the smallest install, `python -m pip install -r requirements.txt` installs
only Gymnasium and NumPy. The editable install also provides `sap-train`,
`sap-benchmark`, `sap-play`, and `sap-dqn` commands.

## Neural training and playback

```bash
python -m pip install -e ".[dqn,plot,dev]"
python dqn.py --episodes 250 --seed 0 --eval-episodes 100 --output reports/dqn.pt --report reports/neural.json
python play.py --policy dqn --checkpoint reports/dqn.pt --episodes 3
python dqn.py --resume reports/dqn.pt --episodes 250 --output reports/dqn.pt
```

DQN uses legal-action Double DQN targets, experience replay, a target network,
Huber loss, and clipped gradients. The CLI defaults to one CPU thread because
this small network often spends more time coordinating extra threads than
computing. `--device cpu`, `--threads`, `--plot`, and `--max-steps` are configurable.

Checkpoints include network dimensions, normalization, hyperparameters,
optimizer, target network, replay, exploration state, and RNG state. Loading uses
PyTorch's `weights_only=True`. Resume starts at a new episode: the environment's
in-flight state and RNG stream are not saved, so it is not a bit-for-bit
continuation of the previous environment trajectory.

For tabular training, use `python main.py --resume qtable.json`. The v2 state
encoder keeps shop-slot identity and combine opportunities. Old built-in
Q-tables used incompatible features and must be retrained; loading rejects the
mismatch instead of silently using an empty policy. Legacy 10-input neural
checkpoints can still be played offline, but require retraining for the new
13-input training environment.

## Measured results

Local CPU measurements after correcting the reward system:

| Policy | Training episodes | Evaluation episodes | Run victories | Average wins/run |
| --- | ---: | ---: | ---: | ---: |
| Legal random | 0 | 300 | 1.0% | 2.47 |
| Heuristic | 0 | 300 | 100.0% | 10.00 |
| Tabular Q-learning | 4,000 | 300 | 52.7% | 8.45 |
| Double DQN | 250 | 100 | 100.0% | 10.00 |

Training seed is 0. Evaluation reset seeds start at 10,000. The first three rows
come from the benchmark command above. The DQN row uses the neural command
above, `--max-steps 300`, and default `epsilon_decay=0.997`; its matched random
baseline was 1/100 victories and 2.26 wins/run. These are single-training-seed
measurements, not guarantees. The 95% Wilson interval for 100/100 is about
96.3%-100%; the tabular interval is 47.0%-58.2%.

Reports contain each episode's seed, reward, wins, lives, turns, steps, and stop
reason. Comparisons use identical reset seeds and only legal random actions.
Evaluation preserves the policy's exploration RNG and does not populate its
Q-table. Explicit truncation counts distinguish unfinished runs from defeats.
Different policies consume randomness differently after reset; shared seeds do
not imply identical later shops or opponents.

The revised heuristic also used 26.5% fewer actions than the previous heuristic
on reset seeds 0-299 (22.51 vs 30.63 per run), while both won all 300 runs under
the corrected environment. This measures reduced unnecessary shopping, not a
higher victory rate.

## How it works

| Module | Responsibility |
| --- | --- |
| `game.py` | Bounded Gymnasium shop/battle loop, legal-action masks, episode diagnostics |
| `agent.py` | Tabular Q-learning, legal Bellman targets, JSON checkpoints |
| `dqn.py` | Double DQN, replay, training, portable PyTorch checkpoints |
| `heuristic.py` | Deterministic baseline choosing actual strength improvements |
| `main.py` | Tabular state encoder, training CLI, plots and reports |
| `evaluation.py` | Seeded evaluation, per-episode records, confidence intervals |
| `benchmark.py` | Comparisons on the same evaluation reset seeds |
| `play.py` | Offline policy playback; explicit experimental live mode |
| `autogui.py` | Saved calibration, OCR, guarded desktop controls |
| `train.py` | Optional custom-shop drill using sapai combat |
| `metrics.py` | Optional headless training-curve plots |

The observation is a `float32` vector of 13 values:

```text
gold, turn, shop[3], team[5], wins, lives, shop_actions_this_turn
```

Actions `0..2` buy the corresponding shop slot, `3` rolls, `4` sells the weakest
pet, and `5` ends the turn. Equal scalar strengths combine by adding one strength.
Buying costs 3 gold, rolling costs 1, selling yields 1, and each new turn grants
10. Selling existing pets can temporarily raise the balance to 15.

The run ends at 10 victories or 0 lives. Turn and shop-action limits bound
stalling policies. Shopping rewards use the change in total strength; selling
removes the same strength credit that buying awarded. A buy/sell cycle therefore
loses reward through action penalties. Terminal transitions do not bootstrap;
time-limit truncations do. See [architecture and design notes](docs/architecture.md).

## Optional sapai battle drill

The [upstream sapai engine](https://github.com/manny405/sapai) provides pet-aware
combat. This integration has a **custom shop** and a fixed enemy; it is not a
complete, up-to-date SAP training environment. Install the tested revision:

```bash
python -m pip install "sapai @ git+https://github.com/manny405/sapai.git@3850d25b1646aa9f8696306c6b8df2a2f7b6aecd"
python train.py --episodes 1000 --seed 0 --output reports/qtable-sapai.json
```

This separate drill offers three shop slots, five low-tier species, gold costs,
rolls, and a bounded episode. Combat uses sapai; shop and buy abilities are not
simulated. Its observation and actions differ from the toy arena, so its saved
Q-tables cannot be used by `play.py`. The integration is tested with mocks, plus
an actual-engine test when sapai is installed. The upstream engine uses NumPy's
global RNG; the adapter saves/restores it around battles and is intended for
serial use.

## Experimental desktop bridge

Use the structured controller in [Desktop controller](docs/desktop.md) for
desktop development. It observes attack/health and occupancy, recognizes
calibrated phases, chooses actions, physically clicks, and checks their effects.
It controls only the foreground game window and stops on ambiguous observations,
unacknowledged actions, loss of focus, a stop file, or its runtime limit.

On this Windows checkout, open Super Auto Pets and restore its window, then run
these commands in PowerShell:

```powershell
cd C:\Users\nikhi\Downloads\SAP-AI
.\play.cmd
```

The launcher uses the project's `.venv`, checks the private native calibration
and OCR dependencies, brings the game forward once, and enables the bot's own
actions and arena transitions. It stops after a terminal result or a configured
bound (one hour, 600 actions, 12,000 observations by default). Changing apps later
stops control. Hold Escape until the bot stops, move the pointer to a screen corner, or create
`.local/desktop/STOP` to stop; the launcher never removes a stop file. It requires
the local verified profile, which is not included in Git. This convenient entry
point does not establish that an entire fresh arena has been validated.

For a shorter run, use `.\play.cmd --max-actions 40 --max-seconds 600`.
With an already verified profile, the separate preview interface remains:

```bash
python desktop.py run --profile .local/desktop/calibration-native.json
python desktop.py run --profile .local/desktop/calibration-native.json --execute --max-actions 40 --max-seconds 600
```

The first command previews without input; `--execute` enables the program's own
clicks. Create `.local/desktop/STOP` to request a stop. Profiles and captures are
private and are not included in Git. See the controller guide for calibration.

A recorded Windows run completed shop turns six through eight, including
purchases, rolls, battles and round continuation, with 31 verified actions and
no agent gameplay clicks. It then stopped at an unseen turn-nine overlay.
This is partial desktop validation: the stat-based policy lost those battles,
fresh-start naming is unvalidated, and recognition still has gaps.
[Desktop progress](docs/desktop-progress.md) records the evidence and limits.

### Legacy bridge

The older `play.py --live` bridge below is retained for compatibility. Its
limitations do not describe the structured `desktop.py` controller above.

`python play.py` runs offline. Desktop control requires explicit `--live`, an
installed Tesseract binary, optional Python dependencies, and a calibrated file:

```bash
python -m pip install -e ".[live]"
python autogui.py --write-template layout.json
# Edit regions/points and screen_size; verify them, then set calibrated to true.
python play.py --live --layout layout.json --preview
python play.py --live --layout layout.json
```

Preview reads one board and prints an action without clicking. Calibration is
checked against the actual screen resolution. OCR reads one screenshot, rejects
ambiguous numbers, and never assumes unreadable text means an empty slot. The
mouse failsafe remains enabled; a live session stops after one end-turn and
revokes control when it exits.

**Legacy perception is incomplete:** empty slots currently require an explicit OCR zero,
pet species and game phases are unrecognized, and the bridge cannot merge pets
or supply the metadata needed by 13-input DQN policies. It is a development
scaffold, not a ready-to-use autonomous SAP bot. Desktop actions are covered by
mocks; real-client behavior has not been validated.

## Validation

```bash
python -m pytest -q
python play.py --episodes 3 --quiet
python main.py --episodes 20 --eval-episodes 5 --no-save
python dqn.py --episodes 5 --eval-episodes 3 --no-save
```

Tests cover Gymnasium compliance, legal masks, observation bounds, lifecycle
limits, reward exploits, masked learning targets, truncation semantics, seeded
reproducibility, checkpoint restoration, reports, optional-engine contracts, and
mocked desktop control. PyTorch tests skip when that extra is absent. CI exercises
core tests on Windows/Linux and Python 3.10/3.12, plus a separate CPU PyTorch job.
Generated models, plots, reports, and local calibration files are Git-ignored.
