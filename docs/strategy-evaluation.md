# Economy strategy evaluation — November 6, 2026

The launcher now uses `--strategy economy`. It fills all five team slots before
merging and spends spare gold on another roll only when supported level-one sale
income could still fund a replacement. Each later sale or purchase is replanned
from a new Board and needs its own observed receipt. `--strategy conservative`
restores the previous policy. `--no-sale-funded-rolls` disables the extra search
without changing the fill-first rule. Generic `run` keeps the conservative default.

## Selection and held-out comparison

Exploration used seeds 4000–4249 and 6000–6299. Variants included merging at
three, four or five pets, sale-funded searches, and two balanced-stat scores.
The fixed fill-first/search candidate was chosen before running final evaluation
on seeds 20000–20499. Both sides use fresh policy instances and equal initial
seeds in each pair, with separate opponent instances.

| Opponent policy | Baseline trophies | Economy trophies | Paired difference | Approximate 95% interval | Baseline / economy victories |
| --- | ---: | ---: | ---: | --- | --- |
| Recorded baseline | 4.050 | 4.914 | +0.864 | [0.558, 1.170] | 118 / 155 |
| Conservative | 3.442 | 4.174 | +0.732 | [0.453, 1.011] | 89 / 115 |
| Economy | 3.120 | 3.598 | +0.478 | [0.217, 0.739] | 71 / 97 |

Each row contains 500 paired runs. Intervals are normal approximations for each
paired mean, not simultaneous intervals or guarantees about future runs. The
candidate was not changed after inspecting these held-out results. Future tuning
must use another development seed range and reserve a fresh final range.

The [machine-readable report](strategy-evaluation-20261106.json) records the
configuration and source fingerprints. Reproduce from the project root:

```bash
python -m sap_ai.sim.policy_study --runs 500 --seed 20000 --opponents recorded conservative economy --output reports/economy-validation.json
```

Reports refuse to overwrite existing files. Use `--include-episodes` to retain
individual outcomes. On a small sample, uncertainty may be large; a one-run
study does not report a confidence interval.

## Recorded-board and latency checks

The two policies were compared on 499 distinct Boards in the committed native
replay fixtures. Economy made 16 different proposals (End Turn became Roll),
with zero illegal proposals and no change to abstention on incomplete Boards.
This checks legality on historical evidence; it does not simulate the unseen
shops those extra rolls would produce.

Source names read in the shop are now kept pending across a dispatched purchase
or merge, and copied to the observed target only after acknowledgment. A failed
or conflicting receipt cannot establish the target's name. The cache uses the
observed target stats and tracked level so pre-input checks remain consistent.
Receipt updates use the raw Board, separately from inferred levels; otherwise
an old derived level could erase newly earned merge experience.
Regression tests exercise the full observe–dispatch–receipt–observe path and
verify that the purchased pet needs no extra hover. Shops with zero gold skip
identification, as the current policy cannot fund another purchase there.

## Limits

The simulator uses approximate pet stats and excludes abilities, food and
positioning effects. Its gains do not establish better live-game win rates.
This pass used offline simulation, recorded Boards and fake windows only; SAP
was not opened and no input was sent to the desktop. Live performance requires
a separate measured run.
