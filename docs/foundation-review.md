# Foundation review — 2026-11-06

## Scope

Reviewed the repository history and the current desktop/controller, perception,
calibration, replay, packaging/CI and offline-learning architecture. The focused
code review covered the recent autonomy changes and their interaction with
existing receipts and cached evidence. This is not an independent proof of every
historical change. All verification in this pass is offline: no game launch,
window capture, mouse input or display-settings changes.

The project has a useful separation: typed Board/Action data and receipt-driven
sessions for the Steam client, plus offline simulators and learning experiments.
The toy DQN is still separate from real-game decision making. Existing replay
fixtures remain valuable controller evidence, but are not recognition or
strategy benchmarks.

## Findings fixed in this pass

| Area | Failure | Correction |
| --- | --- | --- |
| Arena accounting | Menu receipts could turn an old Game Over into a completed arena, including across recovery sessions. | Completion and recovery progress use gameplay receipts. |
| Startup/visibility | An already stopped autopilot could open a window; the progress file did not exist during startup or expose a focus wait promptly. | Check stop before startup; save the initial state and the focus-wait state. |
| Pet identity | Empty slots, contradictory species, or stats that later returned to an old value could revive cached names. | Contradictory evidence permanently removes the old cache entry. |
| Tracked levels | Raw unread levels and derived levels produced different species-cache keys; a visible contradiction could later revert to old tracked history. | Normalize levels for cache operations; discard contradictory tracking. A turn regression clears arena history. |
| Scaled calibration | Editing a reference image left the JSON unchanged and reused its obsolete scaled copy. Flattened filenames could also collide. | Cache identity includes reference contents and a format version; generated filenames distinguish original paths. |
| Profile validation | Scaling rewrote data before validating the original profile, allowing rejected reference paths/options to be hidden. | Validate first and reject malformed target dimensions before writing a cache. |
| Offline combat | Two zero-attack front pets caused an infinite battle. | Ability-free combat treats the stalemate as a draw; benchmark counts are validated. |

Each correction has regression coverage. Native IO is replaced by fake windows,
synthetic images or recorded typed observations.

## Offline diagnostics

```
python -m sap_ai.desktop.audit --run-dir .local/desktop/autopilot/RUN --output reports/audit.json
```

The new command streams saved JSONL, checks dispatch/receipt/count consistency,
retains incomplete sessions and pending actions, and reports missing HUD fields,
species/level coverage and gold left at End Turn. Repeated shop polls and distinct
Boards are reported separately. Output excludes screenshot paths, raw OCR and
exception messages; source logs and existing reports cannot be overwritten.
The command does not import the desktop runtime or OCR packages.

Applied to the saved run that previously looped:

- 32 recorded sessions; one terminal screen observed.
- 14 shop-reading timeouts, eight action timeouts and eight observation errors.
- 874 shop polls with unreadable gold; these are repeated observations, not 874
  independent recognition tests.
- No dispatch/receipt/count contradictions in the saved timeline.

These findings explain the old run; they do not establish new live performance.

## Remaining priorities

1. **Measure perception on a fixed held-out frame set.** The phase consensus masks,
   resampling allowances and narrow-glyph fallback need independently labeled
   positive and negative cases across themes, scales, animation and clipping.
   Passing synthetic tests alone cannot justify universal screen-size support.
2. **Strengthen identity and level evidence.** Tooltip appearance and boundary
   agreement are useful gates, but should be evaluated for wrong-target matches.
   Tracked merge experience also assumes supported level-one shop purchases;
   verified observations must continue to override history.
3. **Improve strategy with measured rules.** Food, freezing, team ordering and pet
   abilities remain incomplete. Compare changes on shared seeds and recorded
   boards, then use separately authorized native runs to measure actual trophies.
4. **Keep simulator claims narrow.** Scalar training results and the approximate
   ability-free arena do not predict real SAP wins. The historical README numbers
   predate the latest replacement threshold and are labeled accordingly.
5. **Retain evidence without publishing private data.** Logs, screenshots and
   calibration remain ignored. The removed agent journal is still in old Git
   history; this pass did not rewrite history.

## Commit dates

The arena-accounting fix has a November 5 author date at the user's request.
Its committer timestamp records the actual November 6 commit creation. Other
commits use their actual creation dates. Existing history was not rewritten.
