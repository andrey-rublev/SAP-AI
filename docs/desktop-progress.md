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

Excess-gold confirmation now has its own opt-in phase/action pair. End Turn
can acknowledge this modal, and its confirmation must lead to naming, battle,
or a round result. It cannot acknowledge an unchanged dialog or unknown frame.
Stable recognition, repeat prevention, and transition limits also apply here.
Validation: 295 focused state/session/runtime/vision tests pass.

The program confirmed that modal, acknowledged battle, recognized victory,
then used a newly calibrated continuation point and observed the turn-four
shop: 10 gold, 3 trophies, 5 lives. These actions were program-executed across
bounded sessions, with calibration work between sessions. An uninterrupted
turn-four sequence is the next live check.

A paired shopping experiment compared eight policies on 50,000 seeds (400,000
episodes) with no illegal actions or unexpected stops. Combined threshold-two
changes increased this synthetic immediate-stat objective by 4.42%; ranking
upgrade gains at threshold four improved its gold-efficiency proxy by 2.75%.
The live policy is unchanged: abilities, food, combat, and experience strategy
are absent from this experiment. Twenty-six experiment tests pass and peer
review found no blocking issues. The installed historical `sapai` engine has
different species/stats from recorded current-game offers, so overnight work
does not train the incompatible toy DQN.

Turn four completed its shop inputs and confirmation, but night battle imagery
was unknown. Private battle references now use the PAUSE label on the player's
half, with both day/night examples; DRAW has an explicit round-result reference.
The bot continued to the turn-five tier overlay, then dismissed it after
calibration. It sold the ant at 10 gold but could not read the resulting 11,
so acknowledgment timed out without retrying. The game is stopped at turn five
with four teammates and 11 gold. Turn five also reveals a fourth shop offer.

Shop slots now support an explicit `available_from_turn` calibration schedule.
Unknown turn readings fail closed for gated layouts. The runtime retains the
same prefix indices for input. 115 vision tests pass. The private profile still
needs its fourth slot and the 11-gold crop validated before continuing play.

At 07:16 UTC the five-hour Plus usage window was exhausted; reset is 11:33:44
UTC. Further model work is deferred until reset to honor the user's usage
constraint. Bounded offline CPU jobs continue: two-hour controller soak,
four-hour long-sequence soak, and one-hour perception perturbation sweep.
The eight-hour work window still ends at 14:38 UTC.

## Validation results after the usage reset, 2026-09-29

At 11:35 UTC, local jobs had completed 287,444 thirty-round controller cases
in two hours and 183,473 hundred-round cases in four hours, with zero unexpected
failures. Source hashes identify the exact controller versions used. These are
synthetic IO contract checks, not game wins.

The private perception sweep completed 1,485 cases over eleven labeled native
frames in 1,388.6 seconds: 21,487 correct fields, 6,053 unknown, zero incorrect,
zero false-empty, and no case errors. Perturbed/reference-source frames are
not held-out gameplay accuracy. The reusable evaluator snapshots its inputs
and reports errors separately; an exception can never score as a correct
UNKNOWN observation. Thirty-three focused evaluator tests pass. The full
offline suite passed 688 tests before this final error-accounting revision.

Private calibration now recognizes 11 gold and the fourth shop slot from turn
five. A bot purchase spent 11 -> 8 gold and filled the empty teammate with the
6/3 badger. It stopped because the fourth shop slot had no verified empty
reference; that resulting native sample is now available for calibration.

The fourth-slot empty sample is validated against both empty and occupied
native frames. A longer bounded consecutive-turn run is underway.

A local-only Unity extractor recovered 34 named textures for nine candidate
species. Visual comparison identified matching modern Fish/Otter variants;
partial Ant crops still require care. The extractor preserves IDs/hashes,
rejects existing output and game-installation paths, bounds decoded data, and
reads external resources in bounded chunks. Thirty-three asset tests pass.
All artwork and manifests remain private; no classifier is enabled in play.

## Stop controls and the next live run, 2026-09-29

The bounded turn-five run completed seven actions and seven acknowledgments,
then stopped at an unrecognized defeat screen. The private defeat reference
now passes comparison against 69 recorded frames without changing unrelated
phase results. Fifth-slot geometry is staged for turn nine, with no invented
empty reference; naming and terminal screens remain unvalidated.

The CLI, session, runtime and native input now share the same stop-file and
deadline predicate. It is checked after fresh OCR, between selection clicks,
and after pointer travel before pressing. This closes the gap where a stop
request during OCR or mouse travel could still allow input. Drag interruption
still releases the button. All 79 runtime/CLI regression tests pass.

A separate, bounded 55-minute CPU species-recognition experiment is running
against private local artwork. It is not connected to the live controller;
synthetic accuracy alone will not justify enabling it.

## Uninterrupted desktop controller evidence, 2026-09-29

One standalone run observed all 327 frames and attempted 32 actions, with 31
verified acknowledgments. It sold and purchased a teammate, rolled, confirmed
end turns, observed battles, continued results, and dismissed a tier overlay
across shops six through eight without agent gameplay clicks. It stopped at
the previously unseen turn-nine tier overlay. The three battles were losses;
this validates control and sequencing, not competitive strength.

The sanitized board/action recording is now an offline regression fixture.
Seven tests reproduce exact action/acknowledgment polls, delayed observations,
shop compaction and safe stopping. Corrupted purchase evidence must time out
without retries. Screenshots, paths and account metadata are excluded. The
full offline suite passed 718 tests before these seven additional tests.

A follow-up dismissed the newly calibrated tier overlay, sold the fish and
bought a 6/5 rooster. It stopped because the newly empty fifth slot lacked a
reference. The actual empty sample is now calibrated against an occupied
negative. No input was retried. The private profile is still specific to this
client geometry; naming and terminal results remain unvalidated.

## Terminal validation and training result, 2026-09-29

The final bounded turn-nine session completed seven actions with seven
acknowledgments and stopped at an unrecognized game-over screen. A narrow
GAME OVER reference was then validated against 157 native recordings: only
the two terminal images changed from UNKNOWN to RESULT. A fresh read-only
native run returned `reason=result`, zero actions, and one poll. No further
gameplay or new arena was started.

The CPU recognition experiment ran 31,697 steps in 3,300.2 seconds. Its original
real-frame report accepted all 17 known-pet samples correctly but falsely
accepted one of seven unknown samples (24 samples, only 14 unique pixel crops).
The small same-session set is not general validation. The model remains
disconnected from production. Reevaluation with shared bilinear preprocessing
produced the same counts: the unfamiliar hedgehog was incorrectly accepted as
beaver at 0.99725 confidence. Raising a generic confidence threshold is not a
validated solution. Original weights, report, loaded source and provenance
remain private alongside the corrected evaluation.

The reusable experiment now bounds training with finite limits, separates
loaded training/evaluation hashes, writes atomic artifacts only under `.local`,
and rejects redirected or linked output targets. It has no desktop input or
live-model integration. Final full-suite validation: 739 passed, one skipped
because this Windows account cannot create test symlinks; the separate Windows
reparse-point regression passed.

## Moving-background phases and fresh-start work, 2026-10-03

Phase templates now support calibrated white-text masks, retaining RGB as the
default. The matcher uses symmetric union-normalized error, rejects inadequate
ink, preserves competing-phase margins and refuses conflicting evidence from
different modes. This addresses the fixed PAUSE label over a moving moon.
Across 167 private native recordings, two previously UNKNOWN battle frames
became BATTLE and all other phase results were unchanged. Visual inspection
confirms PAUSE is present in the recovered frames. Full suite: 786 passed,
one Windows symlink-permission skip.

The user requested work now and one continuation at 07:00 Eastern after the
reset; that single continuation is configured in this chat. Local jobs and
focused delegation should conserve usage, stopping before ordinary usage is
exhausted instead of using paid credits.

Menu navigation started a fresh Turtle-pack arena. The standalone bot then
bought three pets, verified each purchase, ended the turn and confirmed excess
gold. It stopped at the unrecognized naming screen without retrying. The
unselected naming footer and both choice points now have native calibration.
The program selected both name components and stopped at the initially
unrecognized Confirm state. After calibrating that state, a standalone run
performed 20 actions with 20 acknowledgments through naming confirmation,
three battles and the fourth shop. A continuation then performed 41 actions
with 41 acknowledgments over 375 polls in 197.1 seconds and stopped at the
turn-seven loss terminal. The arena finished with two wins. No agent gameplay
clicks were used; calibration pauses mean this is not yet an uninterrupted
fresh-start-to-terminal run.

The controller now has opt-in menu transitions, enabled by `--start-arena`.
Each requires its recognized source phase, calibrated button and observed
destination. Starting an arena requires a known turn-one shop; unknown frames,
stale shops, repeated menus, missing calibration and a terminal result cannot
authorize another start. Full suite: 852 passed, one Windows symlink-permission
skip. Native menu calibration remains pending: returning from the terminal
opened an optional account-registration form, which needs user dismissal under
the Computer Use authentication-dialog restriction. No fields were entered.

## Remaining milestones

1. Continue the private 2560x1440 native profile validation as new states appear.
   The older 2048x1152 Computer Use profile cannot drive native Windows input.
2. Verify OCR, empty slots, species, and proposals across both day and night
   backgrounds. Include naming, battle, victory, and tier-unlock negatives.
3. Extend the verified multi-turn controller run to a fresh start through a
   terminal result without calibration pauses. The turn-six-to-eight replay establishes bounded observed control;
   it does not establish general unattended runs or competitive play.
4. Validate naming and further variations of results and shop layouts. A loss
   terminal, victory/draw/defeat continuation, tier dismissal, and the fifth
   shop slot now have limited native evidence. Broaden this across new runs.
5. Broaden species/level recognition and strategic play (food, abilities,
   ordering), using an observation representation compatible with the real game.

Do not run the toy DQN against the real board. Keep `.local/` captures and
profiles private. Use Computer Use skill for agent-directed UI interaction;
do not automate authentication. Native runtime tests must remain mocked unless
an explicit, observed game state permits a live test.
