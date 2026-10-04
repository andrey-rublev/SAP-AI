# Desktop AI progress

## Working agreement

**Latest user instruction, 2026-10-04:** the user renewed full laptop control,
requested work now and a one-time continuation at 7:30 a.m. Indianapolis time,
and specified a PowerShell workflow: change to this folder and run one command
so the standalone bot takes over and plays. Controlled live validation may
resume; the October 3 background-only restriction is superseded. Check ordinary
Plus usage and stop before paid credits; do not use reset credits. Keep captures
and calibration private. The previous recurring development schedule was
removed through the app.

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

There is no recurring development schedule. The October 4, 7:30 a.m.
continuation is a single run in this chat. Continue work here and keep these
notes current. Stop routine development once the requested live objective is
verified complete.

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

The turn-four-to-terminal recording is now a sanitized offline fixture. Its
replay preserves all 41 action and acknowledgment poll numbers and stops at
RESULT even with menu observations queued afterward. A corrupted tier-dismissal
observation showing stale turn six instead of turn seven times out without
retrying. These tests replay recorded choices; they do not test OCR or strategy.
Full suite after adding the replay: 854 passed, one Windows permission skip.

A private main-menu Play reference was compared against 295 native captures:
only its source image changed from UNKNOWN to MAIN_MENU. The Play submenu and
selected free Turtle-pack setup still need native references and live testing.

A bounded CPU perception sweep completed all 2,160 cases over 16 labeled private
scenes and 135 image perturbations per scene in 1,369.5 seconds. The profile and
templates were frozen at launch. Its final report under
`.local/desktop/frame-evaluations-1003/` contains 21,944 correct field readings,
6,271 unknown fields, zero incorrect fields, zero false-empty readings and zero
observation errors. Team levels are frequently unknown under perturbations.
These are small same-session data, including calibration sources; they do not
establish unseen-client accuracy.

## Autonomous menus and life recovery, 2026-10-03

After the account form was dismissed, native Play and selected Turtle-pack
references were added privately. Across 299 native captures, only the three
new submenu/setup sources changed phase. A corrupted setup image without the
green selection check is UNKNOWN. A fresh main-menu preview proposed `open_play`
with zero inputs.

One standalone `--start-arena --execute` run then opened Play, opened Arena,
started the free Turtle pack, shopped, selected and confirmed its generated
name, and completed two battles without agent input or calibration pauses.
It performed 18 actions with 17 acknowledgments in 176 polls before stopping
at an unrecognized life-recovery overlay after dismissing the turn-three tier
screen. The pending tier dismissal was not retried. This establishes fresh
desktop startup through two battles, with a newly observed interstitial still
blocking an uninterrupted terminal run.

The new `life_reward` phase and `dismiss_life_reward` action now model the
observed tier → life → shop sequence. Recognized life recovery can acknowledge
only a legal tier dismissal; its own dismissal needs a known matching/new shop
turn. Missing calibration, unknown observations, repeated overlays and stale
shops stop without retries. Native comparison over 332 captures recovered only
the two life-overlay images; removing the label remains UNKNOWN. Full suite:
946 passed, one Windows permission skip. A bounded native run then performed
one life dismissal with one acknowledgment in six polls, reaching the turn-three
shop with five lives (the overlay showed four). The program supplied the click.

A read-only level audit found complete level-one digits alongside white UI
borders that trigger the OCR clipping guard. Numeric templates already read
the 23 original team-zero-through-three samples correctly. Border-free private
candidate regions improved a small 161-label perturbation probe from 92 to 134
correct readings, with zero incorrect readings. Team four also has a dim/tinted
digit problem. No live geometry or global threshold was changed: actual level
two/three and clipped-digit negatives remain necessary before adopting these
candidates. Proposed regions: x=474/666/858/1050/1242, y=368, width=20, height=44.

That private border-free level candidate completed the same bounded 2,160-case
sweep in 1,424.9 seconds under `.local/desktop/level-region-evaluations-1003/`.
Its totals are 23,722 correct fields, 4,493 unknown, zero incorrect, zero
false-empty and zero observation errors. The live profile retains the original
regions: improved level-one readings alone cannot validate levels two/three.

The post-reward continuation performed 22 actions with 21 acknowledgments and
stopped after selling a level-one Pig in turn five: the selected slot became
empty and gold increased from ten to twelve. The previous one-gold rule rejected
that legitimate result without retrying. Direct tooltip inspection confirms
the Pig's extra one-gold sale ability and the base Sell(1) button.

Sale acknowledgment now accepts exactly two gold for an identified level-one
Pig. Unknown/other species retain the exact one-gold rule; an identified Pig
with an unreadable or higher level is unsupported and abstains. Same turn,
known gold and selected-slot removal remain required. Private Pig portrait
references for team slots zero/one matched 82 crops across 233 shop images;
inspection of their ten distinct pixel crops found only Pigs. The recorded
ten-to-twelve sale now acknowledges with this calibration. Full suite: 961
passed, one Windows permission skip. Unseen species and artwork remain unvalidated.

The next native continuation performed 33 actions with 33 acknowledgments over
288 polls in 162.1 seconds and stopped at the turn-seven loss terminal. It
identified the remaining level-one Pig, sold it and verified gold six → eight.
Seven consecutive observations around that sale are now a sanitized replay
fixture: temporary identity/occupancy loss does not acknowledge the sale, and
the stable empty slot plus exact two-gold receipt does. Missing identity or
bonus evidence times out without retries. Final full suite: 964 passed, one
Windows permission skip.

Returning to the menu opened the guest registration form again. No fields were
entered; user dismissal is required before the next uninterrupted fresh arena.
This arena had zero wins, so stronger strategy and species/ability observation
remain necessary. At 07:00, read the experimental level-sweep result, then
continue fresh-start validation if the normal menu is available. Do not claim
the calibration-paused arena was an uninterrupted start-to-terminal run.

## 07:00 continuation, 2026-10-03

The single requested continuation fired; no recurring schedule was created.
The working tree was clean and `origin/main` synchronized. Ordinary Plus usage
was available after reset. A fresh read-only game observation still showed the
guest registration form, so live input remains paused pending the existing
user-dismissal handoff. No authentication input was sent.

The frame evaluator now accepts independently labeled species on known occupied
slots and reports correct, unknown and wrong identities separately, including
species confusion per slot. Canonical names and occupied=true are mandatory;
missing identities are omitted from labels. Synthetic Perceptor tests include
an occupied non-Pig that falsely matches a Pig reference, ensuring this is
reported as incorrect rather than unknown.

A bounded private audit of four recorded shop scenes and seven photometric
variants completed 28 cases: 261 correct fields, 68 unknown, zero incorrect,
zero false-empty and zero observation errors. Its 147 species readings were
79 correct and 68 unknown. This combines earlier independent same-session
labels with a newly inspected Pig/Rat/Fish/Beaver scene; it is not general
held-out accuracy. Captures, labels and frozen profile remain private. No
species model or candidate level geometry was enabled in the live bot.

Completed perception studies can now be compared case by case with
`tools/compare_desktop_frames.py`. The comparison verifies identical recorded
data/labels, variants and runtime environment, checks all case counts against
observations, and rejects partial or duplicate reports. It records implementation
and profile provenance while distinguishing gains from regressions, new wrong
readings, false-empty readings and observation errors. Aggregate improvements
cannot cancel a loss in another field or slot.

The paired overnight level comparison covers all 2,160 cases: 21,944 readings
remain correct, 4,493 remain unknown and 1,778 change unknown → correct, all in
team levels. There are zero lost correct readings, new incorrect readings,
new false-empty readings or new observation errors. These are same-session
level-one samples, so real level-two/three and clipped-digit negatives remain
the next adoption requirements. Comparison suite: 40 passed. Combined full
suite: 1,020 passed, one Windows symlink-permission skip. Live fresh-start
validation still needs the registration form dismissed by the user.

## Alternate species references and account handoff, 2026-10-03

Species templates now accept a legacy image path or a nonempty list of paths.
Matching takes the best distance per canonical species before comparing rival
species, retaining the original thresholds, occupancy/stat guards and empty-slot
competition. Same-species references do not create false ambiguity. Legacy
string profiles serialize unchanged. Offline studies freeze every alternate
image; sprite geometry fitting retains the first reference per species.
The full suite passes 1,049 tests with one Windows symlink-permission skip.

A private candidate adds an independently inspected native night-shop Fish
portrait for team slot zero, retaining its earlier reference. Its source scene
is not one of the four species-audit scenes. Paired evaluation of the same
28 cases recovers six unknown Fish readings without losing correct readings or
adding wrong/false-empty readings or observation errors: 267 correct fields and
62 unknown. The candidate remains separate from the live profile pending broader
occupied/empty negatives; this small same-session gain is not general accuracy.

The user completed account setup themselves. A fresh read-only observation
confirmed the normal main menu, and the standalone preview proposed `open_play`
with zero inputs in two polls. No account fields, credentials or settings were
handled by the agent. Fresh native execution can now resume with the existing
private profile; the experimental level regions and species model remain
disconnected from live control.

## Dessert-theme calibration and funded replacements, 2026-10-03

The registered account uses a dessert background and pet hats. The first native
menu run made two actions with one acknowledgment, then stopped UNKNOWN after
opening Arena. A private reference for the already selected free Turtle card
recovered only its two new setup captures among 418 compared images; removing
the check still yields UNKNOWN. No thresholds or input points changed.

Two agent-directed Back clicks reset the menus as test setup. The next
standalone run opened Play, opened Arena and started the selected Turtle pack:
three actions, two acknowledgments, 33 polls. It stopped UNKNOWN at the new
turn-one shop before any purchase. An added dessert Shop-sign reference
recovered only that new shop image among 420 compared images; removing the sign
remains UNKNOWN. The existing HUD/stat and empty-slot calibration read the
new shop correctly without relaxing thresholds. A bounded standalone shop
session then bought all three opening pets: three actions, three acknowledgments,
11 polls, gold ten → one. The program supplied every gameplay input. These
calibration-paused segments are not an uninterrupted fresh-to-terminal run.

The policy now considers a sale-funded level-one replacement with two gold
remaining, or one gold for an identified level-one Pig. It filters candidates
by the supported exact receipt before choosing the weakest affordable pet,
retains the four-stat minimum gain and existing buy/merge priority, and never
sells to fund a purchase when the team already has an empty slot. Policy,
acknowledgment and offline accounting share the existing one-/two-gold receipt
rule; unsupported Pig levels still abstain. Wrong or unstable receipts cannot
trigger a purchase or retry.

The recorded turn-six fixture has a full level-one team, gold two and an offer
with strength nine versus the weakest teammate's five. Previously the policy
ended the turn; it now proposes selling that teammate, then buying only after
the exact one-gold receipt and empty-slot evidence. This is an observed
affordability improvement, not a measured combat advantage. Both simulators
now include the verified Pig bonus; the economy harness retains its level-one
sale restriction. Full suite: 1,102 passed, one Windows permission skip.

## Background purchase validation, 2026-10-03

The user stopped Computer Use with Escape, then requested continued development
without desktop control. Work in this continuation used source edits, mocked
IO and existing typed recordings only. No apps were opened, focused or captured,
and no desktop inputs were sent. Live validation remains paused under this
latest instruction.

An offline corruption of the turn-six purchase exposed a false acknowledgment:
the recorded shop offer was 3/6, but an observed destination of 1/6 or 3/1 was
accepted when the three-gold receipt and source removal were correct. The
controller could then send its next action. Fresh-purchase acknowledgment now
requires complete observed source stats and destination attack/health at least
as high in each field. Missing or lower readings remain pending and stop without
a retry or follow-up input. Higher stats remain compatible; species uncertainty
is retained, and this does not establish that a buff or specific identity was
observed.

The prior native opening purchases are now a sanitized 11-poll replay fixture.
The real policy preserves all three buys at polls 2/5/8 and their acknowledgments
at 4/7/10, including left compaction and a known shop Ant becoming unidentified
on the team. Only typed Boards/actions and relative polls are public; captures,
calibration, timestamps and account data remain private. Recorded corruptions
of each destination stat now time out after one purchase attempt with no later
action. These replays verify controller behavior, not OCR or combat strength.

Full suite: 1,128 passed, one Windows symlink-permission skip. A bounded offline
controller soak after the fix passed 10,000 seeded cases across eight simulated
shop rounds per healthy case, with zero failures. The fresh uninterrupted
desktop arena remains unverified. Next live work, only after renewed user
authorization, is to observe the current game state and continue the new-theme
validation through naming, battle transitions and terminal results.

## Merge receipt consistency, 2026-10-03

Mocked legal merges showed that an increase in one target field could previously
hide a decrease in another: a 5/6 level-two teammate was accepted as 6/5 or 4/7,
and a falling level was accepted alongside rising stats. Acknowledgment now
rejects any known target attack, health or level decrease and known identity
conflicts while retaining the required observed gain. Stable corrected evidence
can still recover within the original budget; missing species/level remain
unknown. Contradictions time out after one attempt with no retry or follow-up.
The focused controller/evaluation/recorded replay suites pass 315 tests. This
is synthetic contract evidence, not an observed native merge or combat result.

## Preserve eligible template conflicts, 2026-10-03

Synthetic scenes exposed two paths that hid eligible evidence while grouping
reference variants. Numeric matching could discard a closer strict reference,
miss an eligible farther reference and accept contradictory OCR. Phase matching
could hide an eligible RGB/white-text reference behind a stricter same-phase
variant, allowing a conflicting mode to authorize a phase. Eligibility is now
tracked before grouping, per numeric field and phase matching mode. An eligible
mode with an ineligible best reference abstains; it cannot silently enable OCR
fallback or another phase. Existing distances, margins and thresholds remain.

Twenty-five new cases cover hidden rivals and same-value variants, both phase
modes and reference orders, valid all-ineligible fallback, agreement and OCR
conflicts. Perception/frame-study/comparison suites pass 309 tests. The same
private four-scene, seven-variant audit against its frozen candidate profile
completed 28 cases in 44.1 seconds: all 267 correct readings remain correct,
all 62 unknown remain unknown, and there are no new wrong/false-empty readings
or observation errors. This is same-session regression evidence; no candidate
profile or trained model was enabled for live play.

## Reusable offline replay export, 2026-10-03

`tools/export_desktop_replay.py` replaces manual log copying with a typed
fixture exporter. It handles appended runs with explicit selection, rejects
incomplete newest evidence, enforces observation/action ordering and final
counts/state, and refuses overwrites or failed IO that Boards alone cannot
reproduce. It drops image paths, timestamps, arbitrary account fields and error
text; it never loads referenced captures or desktop dependencies. Independent
review caught and fixed an impossible same-poll acknowledgment → new dispatch
timeline. Thirty-one focused tests pass, including metadata rejection and
source/output preservation. A smoke export of the actual opening log matches
every Board, action and relative poll of the sanitized three-purchase fixture.
Replay bounds and choices still require explicit test configuration; this tool
does not reconstruct wall-clock OCR/input behavior or establish live play.

## Experimental ranking uses supported sale costs, 2026-10-03

The offline gain-ranking candidate previously charged every replacement two
net gold, even for an identified level-one Pig whose supported receipt makes
the sale → purchase cost one. Ranking now uses purchase cost minus that exact
supported receipt. A synthetic counterexample compares four replacement stats
for one gold against eight merge stats for three; ordinary/unknown species
retain the two-gold replacement cost and select the merge instead. The complete
Pig sale/purchase trace independently confirms receipt two, spend three, net
one and four added stats. Production `DesktopPolicy` is unchanged.

All 49 economy tests pass. A bounded paired study completed 10,000 seeds across
eight experimental policies without illegal actions, truncations or unexpected
stops. It models supported sale receipts and synthetic stat utility; other
abilities, food, equipment, combat and experience value remain excluded. These
results do not establish a competitive strategy or justify enabling a candidate
for live play. The study and its provenance remain private.

## Independent receipt fault soak, 2026-10-03

The seeded controller harness now includes purchase stat mismatches and merges
that raise one stat while lowering another. Both preserve price, source removal,
target occupancy and identity, isolating the contradictory receipt evidence.
Seed parity covers attack and health decreases. Each stops after one attempt
without acknowledgment, retry or follow-up. The independent event oracle checks
nondecreasing target fields and a visible merge gain separately from the applied
fixture board. Deliberately bypassing the production acknowledgment guard now
fails this oracle for both faults and an unchanged capped merge.

All 70 evaluator tests pass. A bounded final-code soak completed 50,000 seeded
cases across 18 scenarios, with eight simulated rounds per healthy case:
50,000 passed, zero failed. The full suite passes 1,231 tests with one Windows
symlink-permission skip. This verifies synthetic controller contracts; it does
not evaluate OCR, combat or native desktop play. Reports remain private.

## OCR engine failures and private portrait audit, 2026-10-03

An offline portrait study launched with the global Python environment exposed
a swallowed missing `pytesseract` dependency. That first attempt is invalid
perception evidence and is excluded from the results below. OCR engine,
dependency and process exceptions now abort the observation at the first failing
crop, rather than silently counting unreadable fields. A missing binding reports
the requirement in the current Python environment. Blank or invalid returned
text still becomes unknown; deliberate reference-only observation remains
supported. Numeric references cannot conceal an invoked OCR engine failure.

Mocked sessions verify zero actions before an initial failure, or one pending
purchase without acknowledgment, retry or follow-up when the engine fails after
dispatch. Frame studies persist observation errors and return a failing CLI
status. The focused perception/runtime/frame suites pass 393 tests; the full
suite passes 1,231 with one Windows symlink-permission skip.

Two previously recorded dessert/clockwork-hat scenes were manually checked as
Ant, Cricket, Cricket. A private six-reference candidate, tested with the project
environment on the existing four-scene/seven-variant audit, completed 28 cases:
268 correct, 61 unknown, zero wrong, false-empty or observation-error readings.
Compared with that audit's matching baseline (261 correct, 68 unknown), all seven
gains were shop slot two becoming correctly identified as Cricket, with no
regressions. A separate source-scene study reached its 120-second bound after
only three of fourteen candidate cases (69 correct labels); it is incomplete
and is not a complete paired result. These source scenes are calibration data,
not held-out evidence. The live profile, thresholds and model selection remain
unchanged; candidate references, images and reports stay private. No desktop
apps were launched, captured or controlled during this continuation.

## Explicit startup handoff and preflight, 2026-10-04

Launching a controller from PowerShell previously left the terminal foreground,
so capture correctly stopped with game-lost-focus. The native adapter now has
an explicit one-time startup activation API. It verifies a visible client and
matching calibrated dimensions before requesting focus, checks stop conditions
before the handoff, and verifies actual foreground ownership and geometry after
it. A denied request, minimized/closed window or changed size fails without mouse
input. Capture and actions still stop after later focus loss; they do not steal
focus back. The separate preflight checks optional package availability without
importing desktop IO and runs a hidden, five-second Tesseract version query.
Preflight and actual OCR share executable selection.

All 142 runtime tests pass with mocked native IO; independent review found no
blocking defect. Live inspection found SAP at its main menu. A fresh private
native capture is recognized as main_menu by the existing 2560x1440 profile.
This establishes current startup recognition only, not a complete arena run.

## One-command PowerShell entry point, 2026-10-04

`cd C:\Users\nikhi\Downloads\SAP-AI` then `.\play.cmd` now starts the
standalone desktop controller with the project's virtual environment. The new
`play` command defaults to the private native profile, enables execution and
arena transitions, and uses finite 600-action/12,000-poll/one-hour limits. It
checks STOP before loading the profile, loads all calibrated reference images,
checks dependencies, opens its local log and requests game focus once. An
existing STOP file is preserved and produces a typed zero-action stopped
receipt. Missing calibration/dependencies and disabled required transitions
fail clearly; `run` remains preview-first. The launcher forwards arguments and
preserves exit status, including from a folder with spaces, without changing
PowerShell execution policy.

All 35 CLI tests pass, and the full suite passes 1,285 with one Windows
symlink-permission skip. Independent review found no actionable defect.
`play.cmd --help` works with the local environment. A bounded native run is the
next check; the command's existence does not establish end-to-end arena play.

## Resume an existing arena, 2026-10-04

The first bounded `play.cmd` test sent its own Open Play and Open Arena inputs.
Arena resumed the previous turn-one shop directly, bypassing pack setup. The
old controller expected ARENA_SETUP and stopped after two attempts, one
acknowledgment and 55 polls. Open Arena now accepts a recognized SHOP only when
turn is positive and gold readable (including zero). Fresh Start still requires
turn one; incomplete readings cannot authorize a follow-up or reset the timeout.

The actual failed run is retained as a sanitized typed fixture. Replaying its
same observations with a two-action bound now acknowledges both menu actions at
polls four/eleven and stops at poll twelve without a duplicate click. This is
offline re-evaluation of native observations, not a newly successful native
resume. Corrupting either shop counter preserves one pending Arena attempt and
blocks further policy input. All 284 session/startup replay tests pass.

A separate native continuation from that observed shop sent End Turn and the
confirmation: two attempts, one acknowledgment, 30 polls, then UNKNOWN at the
new dessert-themed naming screen. Every gameplay input was from the standalone
program; agent inspection supplied no gameplay clicks. The missing naming
reference is the next calibration blocker. Private captures/logs stay local.

## Held-Escape cancellation, 2026-10-04

The one-command launcher now combines its stop file/deadline with a per-run
held-Escape latch, checked before startup activation, observations and dispatch,
and between the native input sequence's moves/clicks. It reads only the current
Escape high bit through typed Win32 IO; no key hook, key log, synthetic key or
additional dependency is used. Hold Escape until the bot stops: a tap between
checks may be missed and in-flight OCR is allowed to finish. Detected Escape
remains latched after release and resets for a new run. Existing pointer-corner,
focus and stop-file guards remain.

All 152 runtime tests pass. CLI tests cover cancellation before activation,
before dispatch and after a pending action, where the receipt remains unresolved
and no retry/follow-up is sent. Together with session checks, 470 focused tests
pass. The full suite passes 1,342 with one Windows symlink-permission skip.
Native Escape cancellation itself has not been exercised with injected
keys; the program never simulates the user's stop input.

## Native naming and battle handoff, 2026-10-04

The dessert naming screen differed from the existing unselected-footer RGB
reference. A private white-text reference over the existing footer region uses
the same .02 distance/.015 margin limits and unchanged input points. An exact
466-frame phase-only regression completed in 37.3 seconds: only the new source
changed UNKNOWN to NAMING, with no other changes and all four NAMING_READY
negatives preserved. Removing ellipses, removing "The", removing all glyphs or
blanking the footer yields UNKNOWN; the old RGB reference stays ineligible on
these corruptions. The broad RGB candidate was discarded. The robust reference
was promoted locally after preserving the original private profile.

`play.cmd` then selected both name options and confirmed using its own native
input. Choose Name was acknowledged at poll four; Confirm Name at poll 29 on
entering battle. It observed the round result and sent Continue, then stopped
UNKNOWN at the new night-shop theme: three attempts, two acknowledgments,
71 polls. Fresh inspection showed the resulting turn-two shop with one win,
five lives and ten gold. This is one observed battle outcome, not competitive
performance. There were no agent gameplay clicks.

The sanitized naming replay preserves the actual later UNKNOWN failure while
checking the first two actions and their recorded acknowledgments under an
explicit two-action bound. Corrupting name-ready observations blocks Confirm
and all retries. Only typed phases/actions/polls are public; generated name
text, images, accounts and calibration remain private. All 287 session/startup/
naming replay tests pass. The night shop is the next live calibration blocker;
this calibration-paused sequence is not an uninterrupted fresh arena run.

## Native night shop and tier-overlay calibration, 2026-10-04

The new night shop exceeded every existing SHOP RGB distance limit. A private
reference for the tight Shop word uses the existing mask matcher with a
low-spread ink selector; the .045 distance/.015 margin and input points remain
unchanged. All four individual letter removals and full-word/solid replacements
abstain. A 467-frame phase-only regression completed in 39.98 seconds with only
the new source changing UNKNOWN to SHOP. All seven preceding naming-run frames
were separately checked and preserved. This is source calibration and regression
evidence, not held-out theme accuracy. The profile was backed up before local
promotion; images and calibration remain private.

The standalone program then bought two pets, rolled, ended turn and confirmed
using its own input. These five actions were acknowledged at polls 7/10/14/17/42.
It observed battle/result and sent Continue at poll 71, then stopped at an unseen
tier-two overlay: six attempts, five acknowledgments, 96 polls, UNKNOWN timeout
with Continue unresolved. No agent gameplay clicks were used. The sanitized
night-shop fixture preserves this actual failure. Real-policy replay with a
five-action bound preserves native action/acknowledgment timing and stops in
BATTLE at poll 43. A purchase destination with contradictory health, despite
matching gold/source removal/occupancy, blocks acknowledgment and all follow-up
input. All 42 night/naming/startup/export replay tests pass.

A private tier reference now matches only the constant "pets unlocked!" footer
suffix, excluding the tier digit and animated die. Its inherited .015 distance
and .01 margin remain unchanged. A 483-frame regression, including all fifteen
night-run frames, completed in 42.76 seconds: only the final source changed
UNKNOWN to TIER_UNLOCK. Removing either word, all glyphs or the exclamation mark,
or replacing the suffix with solid colors, yields UNKNOWN. Header-only and
tier-prefix erasure deliberately still match; this phase cue does not read tier
or turn. The original profile was backed up before adding exactly one reference,
and recorded-image inspection recognizes TIER_UNLOCK without inferred counters.
The subsequent standalone continuation dispatched Dismiss Tier at poll two and
acknowledged the turn-three shop at poll five, then acknowledged a roll at poll
eight. This verifies one native tier dismissal; the bounded continuation is
still running and does not yet establish a finished uninterrupted arena run.

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
