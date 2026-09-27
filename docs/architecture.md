# Architecture and design notes

## Runtime paths

```text
main.py / dqn.py       agent.train(env)
                            |
                     reset -> obs + mask
                            |
                     choose legal action
                            |
                     env.step(action)
                            |
                     learn from reward + next obs + next mask
                            |
                     stop on termination OR truncation

evaluation.py         seeded reset -> greedy policy -> EpisodeResult
benchmark.py          isolated training -> common evaluation reset seeds -> JSON
play.py               saved or heuristic policy -> simulator (default)
                                            -> calibrated OCR bridge (--live)
train.py              custom team-building shop -> optional sapai battle engine
```

The core depends only on NumPy and Gymnasium. Optional modules import PyTorch,
Matplotlib, and desktop libraries only when their functionality is needed.
Importing the desktop bridge does not capture the screen or initialize control.

## Environment contract

`reset(seed=...)` returns `(observation, info)`; `step(action)` returns
`(observation, reward, terminated, truncated, info)`.

| Indices | Meaning | Upper bound |
| --- | --- | --- |
| 0 | Gold | 15 |
| 1 | Completed turns | 30 |
| 2:5 | Shop strengths | 6 each |
| 5:10 | Team strengths | 50 each |
| 10 | Wins | 10 |
| 11 | Remaining lives | 5 |
| 12 | Shop actions this turn | Configured limit, normally 40 |

Only integer actions 0..5 are accepted. Valid action IDs that are illegal for
the current board produce a penalized no-op and consume the shop-action budget.
Stepping before reset or after an episode finishes raises `ResetNeeded`.

The mask has six entries aligned to the discrete action IDs. Terminal states
have no legal actions. Truncated states retain physically legal actions so a
value estimate can bootstrap at the time limit. Callers must still reset before
another step. Legacy 10-value observations remain accepted by mask helpers;
the extended observation preserves the original slot indices.

`info` exposes gold, turn, wins, lives, total team strength, action validity,
observation version, episode steps, and the legal-action mask. Fight steps add
the opponent/team strengths and outcome. Completed runs add total reward,
length, and `episode_reason` (`victory`, `defeat`, `turn_limit`, or
`shop_action_limit`).

## Rewards and learning

Battle rewards are +1 for a win, -1 for a loss, and 0 for a draw, with +5 for
winning a run and -2 for elimination. Shop actions cost 0.05 reward; rolling
also costs 0.02. Buy and sell shaping is 0.1 times the actual change in team
strength. Combining always adds one strength, regardless of the pet's tier.
This removes the former positive-reward sell/rebuy loop. The shaping is a
heuristic signal, not a discounted potential-shaping theorem.

Tabular updates maximize only over legal next actions. Double DQN selects a
legal next action using the online network and values it using the target
network. Terminal and empty-mask targets use zero continuation; truncations
keep their next-state estimate. Both trainers support classic four-value and
Gymnasium five-value step APIs, including classic `TimeLimit.truncated` info.

The compact tabular representation is intentionally lossy. It keeps gold,
bucketed battle pressure, occupied-slot count, weakest strength, and each shop
slot's strength/combine status. It omits detailed team composition and episode
metadata, which limits its ceiling. The neural network receives all 13 values,
normalized using environment observation bounds.

## Reproducibility and persistence

Agents own their exploration RNGs. A training seed initializes the environment
on the first reset; subsequent episodes advance that stream. Neural network
initialization preserves unrelated global Torch/NumPy RNG state. Greedy
evaluation resets every episode to an explicit seed and temporarily seeds and
restores tabular tie-breaking RNG state. It never inserts unseen Q-table rows.

Q-table JSON preserves key/action types, hyperparameters, encoder identity,
encoder version, completed episode count, and exploration RNG state. DQN saves
model/configuration, normalization, optimizer, target network, replay samples,
replay RNG, policy RNG, and counters. Saves use a temporary sibling file followed
by atomic replacement. DQN supports `save(path, include_replay=False)` when a
smaller deployment file is useful.

Neither checkpoint captures environment state. Resume continues learning from
saved training state but starts fresh episodes. Old Q-tables using the previous
encoder must be retrained; dimensions and action order are checked before
resuming built-in training or playback.

JSON reports include per-episode records, averages, truncation counts, and a
Wilson 95% interval for victory probability. They describe one trained policy
on a seed set; they do not estimate variation across independent training runs.

## Boundaries of the optional integrations

The sapai drill delegates only battle resolution to the upstream engine. It has
its own finite shop, gold accounting, and action set. It does not claim to
reproduce the real purchase phase, buy abilities, all packs, or current balance.

The desktop bridge models only scalar OCR. Equal numbers cannot identify equal
species, so it buys into empty slots only. Ambiguous OCR fails before input,
calibration must match resolution, preview disables control, and a live session
ends after end-turn rather than guessing whether a battle finished. Robust
species recognition, empty-slot detection, phase detection, and the real game's
action semantics remain necessary for autonomous real-client play.
