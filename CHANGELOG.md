# Changelog

Milestones for the desktop bot and the offline lab. Detailed run logs, captures
and calibration stay private in `.local/` (Git-ignored).

## 2026-10-20 — runs on its own

- Autopilot (`play.cmd` default): keeps playing arenas; a stalled session is
  followed by a fresh one, then a game restart; Game Over starts the next arena.
  Stops after 3 restarts in a row that change nothing, or 5 sessions without a
  gameplay action, instead of looping. Pauses when you use the computer.
- Any 16:9-capable screen: on a 1920x1200 laptop the game runs in a 1904x1071
  window and the 2560x1440 calibration is scaled (regions, points, references,
  tooltip layout), with a measured allowance for resampled phase references.
- Phase signs match on unseen background themes (consensus masks).
- Battle intros (VS screen, fades) no longer time out End Turn.
- HUD fallbacks: italic "1"s read glyph by glyph (gold 11 had deadlocked a run);
  trophies read from the whole "N/10" label.
- Pet levels tracked from the bot's own acknowledged buys/merges/sales, so it
  sells and merges at scaled sizes where level badges are unreadable.
- Upgrade threshold 3 (+0.35 mean trophies offline vs. 4 across three seed sets).
- Live: a fresh arena played start to Game Over in one session (72/72 actions
  acknowledged); hover identification named most pets.

## 2026-10-19 — playing to win, not just playing

- Hover-based species identification: unknown pets are hovered once and their
  tooltip title is read against a fixed pet-name list, so merges can fire live.
  Verified offline on recorded native hover frames (Tiger, Rhino).
- Strategy: merges are preferred once three pets are placed, unread shop levels
  count as level one, and the last 3 gold of a turn buys any strict upgrade.
- Offline arena (`python -m sap_ai.sim.arena`) on the bot's own Board/Action
  types, scored in trophies. Over 1,000 runs vs. ghost teams: new strategy 4.14
  mean trophies (234 wins), old strategy 3.92 (194), old strategy without
  species 1.77 (0 wins), in line with the real bot's 1.0 before identification.
- Batch summaries report trophies per arena and their mean.
- Code moved into `sap_ai/desktop` (the bot) and `sap_ai/sim` (the lab); the
  launcher runs `python -m sap_ai.desktop.cli`. Tools rely on the editable install.
- `DesktopSession.run` split into explicit run state and one method per stage.
- Removed the legacy `autogui.py` / `play.py --live` bridge.
- The agent work journal moved out of the public repo.

## 2026-10 — desktop bot plays whole arenas

- One-command launcher (`play.cmd`): opens the game through Steam, plays, and
  restarts after a verified Game Over for multi-arena batches.
- Six-arena soak (Oct 17): 490/490 actions acknowledged, five verified client
  restarts, trophies 1, 0, 1, 0, 4, 0. Control is reliable; strategy is weak.
- Single desktop-input lease per Windows session; held Escape, screen-corner
  and STOP-file stop controls.

## 2026-09 — desktop controller foundations

- Calibrated phase recognition, numeric OCR with glyph isolation, and empty-slot
  templates for a native 2560×1440 profile.
- Every action waits for action-specific evidence (gold delta, slot changes,
  phase transitions) and is never retried blindly.
- Offline replay fixtures from recorded native sessions; frame/species studies.

## 2026-09-19 — offline lab

- Gymnasium toy arena, tabular Q-learning, Double DQN, heuristic baseline,
  seeded evaluation and benchmarks. The toy arena models pets as scalar
  strengths and is not the real game.
