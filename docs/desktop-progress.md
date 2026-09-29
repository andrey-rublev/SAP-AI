# Desktop AI progress

## Working agreement

**Latest user instruction, 2026-09-27:** the user explicitly authorized screen
control again for this task and asked to remove the scheduled task. The
`advance-sap-desktop-ai` heartbeat was deleted through the app. Game calibration
and controlled live validation may resume; the earlier background-only
restriction has been superseded. Keep captures and calibration private.

**Next-demo requirement, explicitly clarified by the user:** SAP-AI itself
must both decide and physically execute game actions. The running program must
observe the desktop, choose an action, send its own mouse input, and verify
the result. Codex choosing or manually clicking gameplay moves through
Computer Use does not satisfy this requirement. Fix calibration and controller
blockers before the next demo; if the program stops, report and repair the
blocker instead of silently substituting agent-directed gameplay. Label any
manual setup or intervention separately from the bot's verified actions.

The user requests continued development toward actual Super Auto Pets desktop
play and immediate pushes for every small completed change, approximately five
meaningful commits per day. Work on `main`, pushing to `origin/main`; never
force-push. Preserve unrelated edits. Test each coherent change before its
commit, then push immediately. Do not manufacture commits to fill a quota.

There is no recurring development schedule. Continue work from this chat and
keep these notes current. Stop routine development once the requested live
objective is verified complete.

## Implemented, 2026-09-26

- Preserved and pushed the earlier 200-test RL overhaul (`edc969a`).
- Added actual observed desktop state and stat-based policy (`7c338a0`, 58 tests).
- Added calibrated image/phase/stat recognition (`0e766d6`, 58 tests).
- Added stable-frame session and action acknowledgments (`a734704`, 21 tests).
- Rejected empty classifications that compete with pet templates (`692077e`,
  vision suite now 64 tests).
- Excluded pre-input OCR latency from the acknowledgment timer (`bf7a04f`,
  session suite now 22 tests).
- Added Windows client-relative IO, offline frame inspection/template extraction,
  calibration overlays, preview, bounded execution, and local JSONL logging.
  Interrupted drags always release the button; focus and geometry are checked
  before input. Thirteen native-runtime tests use mocks only.

Final local validation for this batch: **357 tests passed**, including the
optional sapai integration. The installed `sap-desktop` command's help works.
No real shop purchase or autonomous turn has been claimed or verified.

The game is installed on this machine and was launched through Steam. The user
completed Steam sign-in and opened the game. The game menu was observed via
Computer Use. A real shop profile and successful purchase have not yet been
verified. Live testing stopped at the user's request. Do not infer live
readiness from synthetic tests.

## Progress, 2026-09-27

- Deleted the recurring task at the user's request and recorded renewed screen
  permission above. The initial launch reached Steam's sign-in window; the
  user subsequently opened the game before the live demo below.
- Bounded each Tesseract subprocess to three seconds. A timed-out crop now
  aborts observation immediately, producing a clear session error without
  further OCR retries or mouse input. Other OCR failures remain unknown readings.
- Validation: **361 tests passed**, including four new mocked OCR regression
  tests. No private captures or calibration were added to Git. At this point,
  a real purchase and an autonomous game turn were still unverified.

## Live validation resumed, 2026-09-27

The user requested visible desktop play. Using Computer Use, we opened the
installed game, started a free Turtle-pack Arena run, bought two fish and a
duck, rolled once, named the team, and observed a first-round victory. The
three purchases changed gold from 10 to 7 to 4 to 1, then the roll spent the
last gold. These were agent-directed UI actions, not an unattended Python run.
Private frames under `.local/desktop/` support calibration and offline replay.

The client ignored a drag purchase but accepted selecting a shop pet and then
clicking an empty team slot. The runtime now uses this click sequence for
purchases and merges, with focus/geometry checks at each click and no retry
after an error. **22 mocked runtime tests passed.** Shop compaction and OCR
calibration are also being checked against the recorded real frames.

Recorded purchases also showed that the shop packs its remaining pets left.
Acknowledgment now accepts the selected offer's removal with either preserved
gaps or left-compacted survivors, checking every survivor's stats and any known
species/level. Unknown species still cannot establish exact identity. A replay
of the observed fish/duck/fish purchase and negative cases pass: **39 session
tests**, and **119 combined runtime/session/state tests**. The first victory
screen required a click to return to the turn-two shop.

The visible demo continued through turn two: bought an otter and ant, rolled,
and merged a shop fish into a team fish (2/3 became 3/4). The second battle also
ended in victory. The game is left at the turn-three shop with **2 trophies,
5 lives, and 10 gold**. Both rounds were agent-directed; the standalone runner
has not completed an end-to-end live turn. The turn-three tier-unlock overlay
also needed dismissal before the new shop settled.

Real frames revealed OCR reading a health digit as 72 or 4. Numeric OCR now
isolates complete contrasting glyphs, removes frame lines, pads the image,
and uses one bounded raw-line recognition call. Blank/clipped crops return
unknown rather than a guessed or partial value. **393 tests passed** after
the click, compaction, and OCR changes. Offline inspection of the held-out
turn-three frame reads all occupied attack/health values correctly and
proposes a roll. Some zero-gold, wins, and level crops remain unreadable;
species coverage remains incomplete. Recorded day/night shops, naming,
battle, victory, and tier-unlock frames have been checked; later-turn coverage
still needs expansion.
The private profile stays `calibrated: false`; unattended play is not enabled.

## Overnight development, 2026-09-29

The user authorized eight hours of development and live program testing,
06:38–14:38 UTC. The target is the program's own observe/decide/click/verify
loop. Agent-directed gameplay is not a substitute for this validation.

Optional, field-specific numeric references now supplement OCR. Ambiguous
references and conflicting OCR remain unknown. Private replay checked 286
expected numeric/occupancy fields across eleven recorded shop frames with no
unknown or incorrect values. This set includes reference-source frames and
does not establish general recognition accuracy. A fresh turn-three frame
read all HUD values, occupancy, and attack/health values correctly without
new references; one hovered level badge remained unknown. The local profile
was enabled for a bounded roll test. No private calibration is committed.

Naming, naming-ready, round-result, and tier-unlock are now distinct phases;
the terminal result phase remains separate. Shop dice indicate tier, so they
are not used as pet-level observations. Species coverage remains incomplete.
Validation: 279 focused state/session/runtime/CLI/vision tests passed with
mocked IO. A live program action is the next check.

The controller now supports opt-in, phase-specific naming, round continuation,
and tier dismissal, with stable observations, expected next-phase checks,
repeat prevention, and bounded transition waits. The CLI adds a time budget,
stop file, last-frame diagnostic, and failure status for unacknowledged input.
The first live preview sent no inputs: native capture is 2560x1440, whereas
Computer Use supplied 2048x1152 images. The exact-size check caught this before
input. A separate native-resolution private profile is being validated.

A seeded offline controller evaluator now records reproducible failure cases,
action coverage, source hashes, and atomic resume checkpoints. Peer review
caught and fixed swallowed interrupts and missing independent acknowledgment
checks. Fifty harness tests and 1,600 additional cases passed after those
fixes; these cases do not exercise vision, native IO, or combat strategy.

The native-resolution live preview passed. The standalone program then chose
and physically executed one roll, independently observed gold 10 -> 9 with
the new shop, and stopped at its one-action limit: one attempt, one
acknowledgment, six observations, no pending action. This is the first verified
program-executed desktop action. It does not yet establish autonomous turns.
The full offline suite passed 529 tests before subsequent cache work.

Exact-pixel OCR memoization now avoids repeated subprocesses for unchanged
crops. The bounded, session-local cache stores raw OCR strings, retains range
and template checks, and never stores engine failures. A private recorded-frame
benchmark measured 2.80 seconds initially and 0.0044 seconds on the identical
frame; changing/animated frames still need OCR. Nineteen cache tests pass.

The program completed five further rolls and acknowledged each gold decrease,
then correctly stopped before a sale because no Sell point was calibrated.
Selecting a teammate for calibration exposed the Sell button; that inspection
did not sell a pet. Sales now use select-then-click, with calibration checked
before selection and focus/geometry checks before both clicks. Fifty-nine
mocked runtime tests pass, including six added sale regressions.

Live validation then confirmed a program-selected sale (gold 4 -> 5 and the
selected team slot became empty), followed by its chosen replacement purchase
(gold 5 -> 2, a 3/6 pet entered that slot, and the shop compacted). The chosen
sale was a 2/3 fish; the ant's level was unreadable, so the policy excluded it.
End Turn opened an excess-gold confirmation. The program stopped on that
unknown screen without clicking again. Dialog support is being added.

Optional `--record-dir` now saves changed observed boards with frame paths in
the event log, distinct directories per run, and a default 200-frame limit.
Repeated stable polls do not produce duplicate files. Eighteen mocked tests
cover limits, unique paths, evidence linkage, and stopping on recording errors.

## Remaining milestones

1. Finish the private 2048x1152 profile using the recorded turn-one through
   turn-three frames. Keep `calibrated` false until all readings and input
   points pass validation. Empty team and shop reference images are available.
2. Verify OCR, empty slots, species, and proposals across both day and night
   backgrounds. Include naming, battle, victory, and tier-unlock negatives.
3. Verify a bounded run of the standalone controller: purchase acknowledgment,
   roll, end-turn, and the following shop without duplicate inputs. The
   agent-directed demo does not establish that this pipeline works end to end.
4. Add calibrated naming/dialog/result handling and changing shop-slot counts,
   including victory continuation and tier-unlock dismissal. Record verified
   multi-turn controller playback before claiming autonomous runs.
5. Broaden species/level recognition and strategic play (food, abilities,
   ordering), using an observation representation compatible with the real game.

Do not run the toy DQN against the real board. Keep `.local/` captures and
profiles private. Use Computer Use skill for agent-directed UI interaction;
do not automate authentication. Native runtime tests must remain mocked unless
an explicit, observed game state permits a live test.
