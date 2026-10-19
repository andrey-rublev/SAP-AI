# Changelog

Milestones for the desktop bot and the offline lab. Detailed run logs, captures
and calibration stay private in `.local/` (Git-ignored).

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
