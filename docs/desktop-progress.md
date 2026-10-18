# Desktop AI progress

## Working agreement

**Current continuation, 2026-10-18:** work in the background using source edits,
mocked IO, recorded development frames and bounded CPU jobs. The October 17 live
control window expired at 5 p.m.; live inspection/control needs renewed human
authorization for today in this chat. Do not launch, focus, capture or control
desktop apps until then. The requested deliverable remains a PowerShell command
that runs the standalone bot, which itself observes, decides, acts and verifies.
Check ordinary Plus usage first and reserve enough to test, review, push and
write a handoff. Do not spend paid/reset credits or change account settings.
Keep captures, account data, calibration, artwork and trained models private.

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

There is no recurring development schedule. Today's 8:30 a.m. and 1:15 p.m.
one-shots were both deleted after firing. Create no further wakeups. Read the
latest private October 18 handoff before resuming. Stop routine development once
the requested live objective is verified complete.

Normal start is `.\play.cmd`; hold Escape, move to a screen corner or create
`.local/desktop/STOP` to stop. Eleven earlier uninterrupted fresh native runs
are evidence for the earlier conservative baseline; no October 18 native run
has occurred. Current blockers include tooltip/target binding, readable shop
levels and directly validated food effects. No learned recognizer is enabled;
the consumed duplicate-refused final species study must not be rescored or
described as unseen data. Keep the toy DQN separate from real-game decisions.

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
eight. That native continuation finished with 21 attempts, 20 acknowledgments
and 202 polls across turn-three/four shops. Each shop included a supported
sale-funded replacement with observed +1 sale/-3 purchase receipts and a visible
four-to-nine stat gain. It stopped UNKNOWN on the tier-three overlay after
Continue at poll 177. The typed replay preserves the original failure and all
twenty acknowledged action timings; a separately labeled synthetic splice of
recorded sequences provides prior-turn context and rejects a stale shop after
tier dismissal without retries. This is bounded observed control, not a finished
uninterrupted fresh arena run.

## Optional tolerance for rasterized text edges, 2026-10-04

The tier-two and tier-three footer suffixes have identical glyph identities and
bounds, but their thresholded masks differ by .050235, above the .015 limit.
Whole-pixel translations through three pixels and brightness-threshold probes
do not solve it. Inspection shows thin opposing edges from fractional text
rasterization. White-text references can now opt into `edge_tolerance: 1`,
counting unmatched ink farther than one pixel from the opposing mask in both
directions and dividing by the original ink union. This does not wrap at crop
edges or dilute the denominator with expanded background. Blank/sparse/filled
ink gates, reference limits, cross-mode conflicts and competitor margins remain
unchanged. The default is strict zero; only integer zero/one is accepted.

All 530 focused vision/session/night/tier replay tests pass. Coverage includes
one-pixel variations, missing letters, unrelated strokes, larger shifts, ink
limits, symmetry, no wrapping and round-trip validation. Independent review
found no concrete defect. The full suite passes 1,368 tests with one Windows
symlink-permission skip. Production phase/scorer verification covered 538 stored
native images in 46.47 seconds, including all 69 recorded frames from five new
logs. Only the new tier-three source changed UNKNOWN to TIER_UNLOCK; the
remaining classifications were preserved. All 758 source-mask differences are
boundary pixels within one pixel of opposing ink; the tolerant score is zero.
All twelve damaged-footer controls across tier-two/three still abstain. The
private profile was backed up before changing only this reference's tolerance
to one.

## Native continuation through Game Over, 2026-10-04

With the optional edge tolerance enabled only for that suffix, `play.cmd`
dismissed tier three, controlled the turn-five/six/seven/eight shops, dismissed
the later tier-four overlay and reached RESULT: 42 actions, 42 acknowledgments,
377 polls, no pending action or error. Fresh independent inspection shows Game
Over on turn eight with two trophies. This is a completed native continuation
from a tier overlay; it is not a fresh startup or competitive-performance claim.
Every action in the run was chosen and physically executed by the standalone
controller. No agent input occurred during the run.

The sanitized successful fixture reproduces the full native action and receipt
timeline with real `DesktopPolicy`. Trailing calibrated menu phases remain
unread after RESULT, so it cannot start another arena accidentally. A separately
labeled synthetic early terminal observation preserves an unresolved final
confirmation and stops immediately without follow-up. All 42 focused edge/tier/
night/terminal/export replay tests pass. Images, names and calibration remain
private. For the next fresh-start test, the agent used only post-game menu
navigation to restore MAIN_MENU; that setup is outside the recorded bot run.

## Replan after a proven no-input board change, 2026-10-04

A fresh startup test dispatched its own Open Play/Open Arena/Start Arena,
opening purchases, naming and battle transitions. On turn two it stopped after
13 tentative attempts/12 acknowledgments/88 polls: the pre-input capture changed
one teammate from readable 3/2 stats to unknown during an animation. Later
inspection showed 5/2. No input was sent for the rejected thirteenth attempt;
the original failed log remains private. This still does not establish an
uninterrupted fresh game.

The runtime now signals only that exact pre-input board mismatch with
`BoardChangedBeforeInput`, whose contract guarantees no pointer or click input.
The session removes the tentative attempt/pending marker, restores its previous
transition and duplicate-action state, requires two fresh stable observations
and asks policy again. It does not reuse the stale proposal. Poll, transition,
deadline and stop budgets remain in force. Other exceptions, including failures
after a first click, retain the pending attempt and stop without retries.

Typed logs record the deferral separately. The exporter validates its matching
proposal, poll and fixed reason, preserves sanitized optional deferrals, and
rejects a forged new proposal before two fresh observations. Legacy fixtures
keep their original schema. Mocked runtime/session checks verify that a changed
destination receives no stale click and policy selects a newly legal destination.
The full suite passes 1,399 tests with one Windows symlink-permission skip.
Independent review found the exporter freshness gap, which was repaired and
covered by forged same-poll/one-observation follow-up cases; no runtime/session
defect was found. Ordinary five-hour usage reached 98%, with the paid/reset
credits untouched during this continuation, so further native testing is
deferred to the requested one-time 7:30 AM run. No local job remains running;
SAP is open at the turn-two shop of the fresh attempt. First validate the
no-input deferral recovery there, then prepare MAIN_MENU after terminal and
attempt an uninterrupted fresh-start-to-result run. Do not treat the previous
calibration/resumption sequence as that evidence.

## Victory recognition and recorded-image deferral validation, 2026-10-04

The requested 7:30 AM continuation began with ordinary five-hour usage reset,
46% weekly usage and unchanged paid/reset credits. The native recovery from the
turn-two shop completed 25 actions and 25 acknowledgments in 277 polls, then
stopped UNKNOWN at a nonterminal Victory screen on turn four. Independent fresh
inspection confirmed the word and one trophy; no gameplay input came from the
agent. This run did not exercise a pre-input deferral.

A private strict white-text reference now covers the complete Victory word,
excluding the character and background. The previous RGB reference exceeded
its unchanged distance limit and clipped the bottom of the current word. A
701-frame production-perceptor comparison completed in 58.14 seconds: only the
new source changed UNKNOWN to ROUND_RESULT, with no known-phase changes. All
eleven damaged-word/solid-color controls remained UNKNOWN, including removal
of each of the seven letters. The original native profile was backed up before
adding exactly one reference; thresholds, existing references and species
recognition were unchanged. Terminal victory layouts remain unobserved.

With that private reference, the standalone continuation reached RESULT with
54 actions, 54 acknowledgments and 466 polls, without pending input or error.
Fresh independent inspection confirmed Game Over on turn nine with one trophy.
The agent then used only post-game menu navigation to prepare a separate fresh
MAIN_MENU run; these setup clicks are outside the bot's recorded gameplay.

A separate offline study exercised real Perceptor/OCR/runtime/session/policy
against a synthetic splice of stored native images, with window capture and
mouse IO mocked. A changing buff animation rejected the poll-two purchase
before any fake input. Two fresh settled boards led to a new poll-four purchase,
acknowledged at poll six. The study used nine captures and two fake clicks and
finished in 4.02 seconds with one action, one acknowledgment and seven polls.
This is synthetic recovery evidence, not an uninterrupted native deferral.
The sanitized typed fixture preserves the timeline without promoting images,
paths or names. Its real-policy regression also proves that a three-poll bound
stops after the first fresh board with zero actions, acknowledgments or clicks.
All 342 focused replay/export/session tests pass.

A private Horse/Beaver portrait candidate was rejected. It recovered two
highlighted source Horses across 21 labeled scenes, but recovered no species
in an independently inspected, settled, unhighlighted holdout. Horse distances
were .040887/.185903 and Beaver .043945 against the unchanged .04 limit. The
candidate was not promoted and no limits were loosened. Collect separate
settled training references and independent holdouts before adding species.

## Uninterrupted fresh native arena, 2026-10-04

The separate fresh run began with an independently observed MAIN_MENU and the
same native profile, with no intervening calibration edits or agent input.
`play.cmd` dispatched Open Play/Open Arena/Start Arena at polls 2/5/8, then
made opening purchases and selected/confirmed its own name at polls 30/33.
It controlled every shop turn from one through nine, including day/night
backgrounds, four tier dismissals and round continuations. It reached RESULT
in 466.38 seconds: 94 actions, 94 acknowledgments, 874 observations, zero
deferrals, no pending action and no error. Fresh independent final inspection
confirmed Game Over on turn nine with two trophies. Terminal counters in the
typed Board remain unknown; the trophy count comes from that final inspection.
Every recorded gameplay input came from the standalone program.

This is the first uninterrupted fresh-menu-to-terminal validation on the local
2560x1440 setup. It establishes one completed arena, not general unattended
reliability or competitive strength. The program stops at terminal and does
not restart another arena. The current stat-based strategy still has incomplete
species recognition and does not use food, ordering or most abilities.

Open and restore Super Auto Pets at its main menu, then use PowerShell:

```powershell
cd C:\Users\nikhi\Downloads\SAP-AI
.\play.cmd
```

The launcher uses `.venv\Scripts\python.exe` and the private native profile.
Hold Escape until it reports a stop, move the pointer to a screen corner, or
create `.local/desktop/STOP` to stop before further input. The launcher preserves
an existing stop file. Return the completed game to its main menu before a new
arena. Native validation captures, calibration and trained species models remain
private.

The sanitized fresh-arena fixture reproduces all native action and receipt
polls with real DesktopPolicy, including startup and naming. Enabled menu
observations after RESULT remain unread, preventing a second arena. A
counterfactual unselected-name receipt stops with Choose Name pending, without
retrying or confirming it. All 57 focused fresh/edge/deferral/naming/export
replay tests pass. The full suite passes 1,404 tests in 14.42 seconds with one
Windows symlink-permission skip. Independent review confirmed exact source
timeline agreement, terminal stopping and the sanitized fixture's privacy.

## Typed pre-input diagnostics, 2026-10-04

The no-input mismatch exception now optionally carries the typed preflight
Board. Native runtime deferrals retain that board in the event instead of
losing the transient reading after recovery. It remains diagnostic evidence:
the session still restores its action state and requires two fresh stable
observations for policy. The exporter keeps only validated typed evidence,
rejects unchanged fingerprints and malformed/nested metadata, and excludes
capture paths and exception text. Message-only executors and old fixtures keep
their original schema. Independent review found no concrete defects.

Repeating the actual recorded-image splice through the updated production
perceptor/runtime/session took 4.17 seconds, retained the transient unknown
teammate in its preflight diagnostic and preserved the original seven-poll
timeline. All three comparisons against recorded stable Boards matched;
there were still nine fake captures, two fake clicks, one action and one
acknowledgment, with the private profile unchanged. This remains a synthetic
offline splice with mocked input. The typed fixture now carries the diagnostic
without turning it into an observation.
The full suite passes 1,417 tests in 14.37 seconds with one Windows
symlink-permission skip; legacy replay timelines remain unchanged.

The later settled Horse/Beaver study also remains private and unpromoted.
Across 26 production observations, readings improved from 200 correct/36
unknown to 210 correct/26 unknown, with no wrong labels, false empties or
regressions. Three gains were on training and seven on later holds; five held
portrait crops were byte-identical to training and two were nearly identical.
A separate fresh-arena Horse still abstained at .155770 against .04. Disjoint
full images did not establish independent pose coverage. Collect materially
different, settled and independently labeled portraits before promotion.

## Second fresh native arena with observed deferral recovery, 2026-10-04

A second uninterrupted MAIN_MENU-to-RESULT run under the same private profile
completed 90 actions and 90 acknowledgments in 818 observations/439.42 seconds,
with one proven no-input deferral and no pending action or error. The native
process had loaded the earlier code before typed preflight diagnostics were
added, so its original deferral deliberately has no reconstructed preflight
Board. Fresh independent inspection confirmed Game Over on turn nine with two
trophies. All gameplay input in this run came from the standalone bot.

At poll 87 it rejected Buy shop zero into team four before input. Poll 88
observed teammate three as unknown, poll 89 read changed 4/2 stats with unknown
level, and polls 90/91 agreed on level one. The policy then proposed again and
dispatched at poll 91, acknowledging the purchase at poll 93. The previous
readable teammate had been 3/2. The native controller continued through all
shops one through nine and terminal without a calibration pause or gameplay
intervention. This validates a real deferral recovery as part of a complete
arena, beyond the earlier synthetic image splice.

The sanitized second-run fixture preserves every native action, acknowledgment
and deferral poll with real policy. A poll-88 bound stops with the twelve prior
actions acknowledged, no follow-up input and no pending/proposed action; the
unknown slot cannot drive a purchase. Terminal replay leaves appended menu
observations unread. Both fresh runs together have 184 acknowledged actions
and 1,692 observations; both earned two trophies, so strategic strength remains
limited. Recognition coverage and long-term unattended reliability still need
broader evidence.

The one-time 7:30 AM wakeup was deleted after it fired. After both terminal
tests, the agent used only post-game menu navigation to leave SAP at MAIN_MENU
for the next `play.cmd` launch. These setup actions are outside bot gameplay.
Paid and reset credits remained untouched during this continuation.
The final full suite passes 1,420 tests in 14.14 seconds with one Windows
symlink-permission skip. Ordinary five-hour usage was 58%, weekly usage 55%,
and no local game/test job remains running. The documented command workflow
has two complete native arenas behind it; further changes should target the
remaining recognition and strategy gaps instead of producing quota commits.

## Cold game startup and loading grace, 2026-10-17

The renewed work window began from a clean main branch with ordinary five-hour
usage at 1% and no paid/reset credits used. The current installed client was
closed. A fresh baseline from its main menu completed 81 actions and 81
acknowledgments in 738 observations/390.43 seconds, with one typed pre-input
deferral, no pending action and no error. Independent final inspection showed
Game Over on turn eight with two trophies. All gameplay input came from the
standalone program; the agent only opened the installed client for setup.

`play.cmd` can now open the installed Steam game when its standard window is
missing. Only the specific missing-window error permits one launch; arbitrary
window/focus/dependency failures cannot relaunch it. The original native Escape
latch works even before a client exists. Stop-file/runtime limits remain live
through the bounded 90-second wait. `--no-launch` requires an existing game and
`--launch-timeout` adjusts that bound. No authentication input is implemented.

The first cold test revealed that Unity exposes the final title before applying
saved fullscreen geometry. It stopped before capture/input. Newly launched
clients now wait for the calibrated size using native reads without focus or
capture. The second test reached full size but stopped after 25 unknown
observations/8.04 seconds on the game's automatic login/loading screen, again
with zero actions. Initial unknown grace is now separately bounded: `play`
allows 300 initial observations, then restores the normal limit of 25 after
the first recognized phase. Other run/session defaults remain unchanged.

After confirming the game was closed, the third command-only test launched it,
waited through 30 initial unknown observations and dispatched Open Play/Open
Arena/Start Arena at polls 32/35/38. Its first three purchases were acknowledged
and it reached the turn-one shop without agent input. It later stopped safely
at turn six with 53 dispatched actions and 52 acknowledgments: an unidentified
level-one Pig sold for two gold, but the receipt expected one. Private profiles,
capture regions and thresholds are unchanged.
Independent review approved the startup guards and loading grace; mocked
startup/runtime/CLI/session tests cover stop conditions, late windows, geometry
transitions and restoring the original unknown limit.

## Visible-shop planning study, 2026-10-17

A sandbox candidate plans several supported level-one replacements using the
visible offers and observed sale-income lower bounds. For example, two gold
can fund two identified Pig replacements instead of one ordinary replacement.
It preserves unknown-state abstention, empty-slot fills, identified merges and
baseline ordering when the candidate adds no modeled utility. Every actual
proposal would still need a fresh board and an observed receipt; the plan is
not an input macro. The candidate is separate from the live policy.

On disjoint 25,000 development/25,000 held-out synthetic shops, the fixed
candidate improved held-out attack-plus-health by .02572 per shop (95% paired
mean interval .02255 to .02889): 278 better, 24,722 equal and zero worse on that
stat proxy. It spent .0184 more gold per shop. Both splits passed 64,640 modeled
production receipt checks. No weights or parameters were trained. Combat,
abilities, food, experience and future-roll value are excluded, so this is not
a win-rate or general no-regression result.

The historical proposal audit covered 393 distinct recorded shop Boards:
290 legal proposals, 103 abstentions and zero changed choices. There is no
demonstrated native benefit, so the candidate remains a sandbox. All 85 focused
candidate/economy/Pig-sale/opening replay tests pass. The report remains private.

## Unidentified level-one sale receipt, 2026-10-17

The selected native 4/1 pet became transiently unreadable, then visibly empty,
while gold rose from eight to ten. Sale acknowledgment now accepts one or two
gold for an unidentified level-one seller after the selected slot becomes empty.
Planning still budgets only one gold. Identified pets retain their exact receipt
rules, unsupported levels remain protected, and the controller still requires
stable observations and never retries a pending sale.

A sanitized seven-observation fixture replays this recorded window through the
real policy/session with mocked input. It dispatches one sale at relative poll
two and acknowledges it at poll six. This is offline evidence; the original
native timeout remains recorded and the new receipt still needs live validation.
Counterfactual gold, occupancy, target, species and level cases reject invalid
receipts. All 309 focused session/Pig-sale/replay tests pass.

Earlier today a separate fresh native arena completed with 81 dispatched actions,
81 acknowledgments and no pending action, reaching the turn-eight loss terminal
with two trophies. This establishes another complete bounded run, not competitive
performance or general unattended reliability.

## Explicit food observation schema, 2026-10-17

`Board` can now carry optional typed food slots with separately observed identity,
price, effect and stat bonuses. No values are inferred from an identity. An
uncalibrated food row remains distinct from an unknown or empty calibrated slot;
legacy board JSON and fingerprints remain exact when food is not configured.
The legality check requires an allowlisted flat-stat food contract, enough gold
and a readable occupied target. The default policy, perceptor and native runtime
do not yet buy food; this is a foundation for separately validated observation,
execution and receipts. No current Apple balance or stat-cap rules are claimed.
Independent review found no compatibility blocker; 164 targeted tests passed.

## Real-image species training and rejection study, 2026-10-17

The offline training tool now supports labeled native crops with separate
training, calibration and final-test roles. It records source/reference hashes,
rejects split aliases and near-duplicates (including resized/mirrored inputs),
distinguishes unfamiliar occupied pets from empty slots, and freezes weights and
rejection thresholds before final evaluation. Final-test preflight rejects
changed declarations, resolved source paths, invalid weights or unsafe thresholds
before opening test images. Models, images and audit reports remain private;
production promotion is explicitly disabled. Independently reviewed species and
prototype tests pass: 74 passed, one expected Windows symlink skip.

The October 4 first fresh arena was explicitly reassigned to training and the
second to calibration. Manually checked head-inclusive crops produced 149 unique
training inputs and 111 calibration inputs across 19 known species plus unknown
and empty classes. Artwork recurs and samples remain correlated; these are
appearance/theme/slot tests, not independent novel-art generalization. Badger is
missing from calibration. October 17 images remain unopened final-test data.

A bounded CPU run completed 20,000 steps in 317.55 seconds. Although it fit all
124 known training inputs, the original .95 threshold accepted six wrong known
identities and 18 of 24 unfamiliar pets on calibration. Tightening the confidence
threshold retained only four of 83 known inputs with zero observed unsafe accepts.
A separate fixed RGB prototype baseline retained eight of 83 at distance .04 and
margin .015 with zero observed unsafe accepts. Neither recognized the four
calibration empty slots. Both remain disabled: further training alone would not
resolve this demonstrated recognition gap. Calibration metrics are not final-test
accuracy or evidence of safe native species recognition.

Reproduction entry points, using a reviewed private manifest:

```powershell
.\.venv\Scripts\python.exe tools/train_desktop_species.py train-real --manifest .local/desktop/real-species-heads-traincal-1017.json --output .local/desktop/new-species-run --seconds 900 --steps 20000 --threads 3
.\.venv\Scripts\python.exe tools/eval_desktop_species_prototypes.py --manifest .local/desktop/real-species-heads-traincal-1017.json --output .local/desktop/new-prototype-run
```

## Bounded arena collection, 2026-10-17

`play --arenas N` adds optional multi-arena collection without changing the
single-arena default. A batch shares all action, observation, time and stop
budgets. Only an arena with acknowledged gameplay and verified RESULT can
restart. The controller observes the terminal again, requests normal WM_CLOSE
once, verifies disappearance and uses the existing bounded Steam startup path.
It does not force-kill or retry shutdown. Zero-action terminal clients, pending
actions, errors, lost focus and exhausted budgets prevent further restarts.
Restart attempts are separately logged and capped at N minus one.

Unique private batch directories preserve existing logs and separate each
arena's receipts/recordings. `first_recognized_phase` distinguishes resumed
shops from menu starts. Root review and 601 focused tests covered the lifecycle,
then 58 batch/lifecycle tests passed after final diagnostic refinements. The
combined suite passed 1,686 tests with one expected skip. The first two-arena
native attempt resumed the shop and stopped after 40 actions/40 acknowledgments
at turn ten, without completing or restarting an arena. A hover tooltip covered
three team stat regions. No unattended batch reliability claim is made yet.

## Keep hover tooltips out of observations, 2026-10-17

An optional calibrated `pointer_rest` now moves the pointer to a neutral place
before the first executing observation and once after each dispatched input.
Preview and unconfigured profiles do not move. The movement-only native primitive
checks focus, bounds and stop conditions before and after travel. Failures retain
pending actions and cannot trigger gameplay retries or substitute known stats.

Mocked tooltip-camera tests verify startup recovery, post-purchase acknowledgment,
preview preservation and failure cleanup; 87 focused parking/batch tests pass.
On the stopped native turn-ten shop, startup parking removed the tooltip and all
five team stats became readable before the bot resumed. A separate later stop at
turn eleven exposed a blank OCR result for the turn counter after tier dismissal;
that receipt remains recorded as unacknowledged and is being diagnosed separately.
The pointer-rest calibration is private and no recognition limits were loosened.

## New hover/title diagnostics and turn-eleven calibration, 2026-10-17

The movement-only diagnostic probe requires an existing foreground client, a
calibrated pointer-rest point and two matching fresh SHOP observations before
hovering each occupied profile slot. It caps targets at five and time at 30
seconds, parks between targets and records typed before/after observations into
a new private directory. Its manifest declares development use before capture;
these new diagnostic frames cannot enter held-out species evaluation. It never
clicks, launches, activates or interprets an identity. The final 66 probe/parking
tests passed, including OCR time reservation and path confinement.

A native two-target probe completed in 8.51 seconds with five pointer moves, six
captures and the pointer restored. Independently inspected tooltip titles showed
Tiger and Rhino. The offline exact-title reader returned both correctly using
two agreeing preparations on manually selected title-only crops. Its 47
synthetic/injected-OCR tests cover ambiguity, missing color evidence, malformed
names, dependency failures and bounded engine errors. This establishes two
diagnostic readings, not automated title location or production recognition.

Separately, the parked run stopped with 11 attempted actions/10 acknowledgments
after tier dismissal because numeric OCR returned blank for the visible turn 11.
An independently inspected private turn-eleven reference now uses the existing
numeric-template mechanism at distance .005. None of 117 distinct historical
other-turn crops matched; the closest distance was .07417 (observed turns 1–10,
not independent manual labels). No global threshold or receipt rule changed.
The original timeout is preserved; a new resumed batch is being tested.

## Calibrated tooltip title location, 2026-10-17

A pure offline layout helper can now locate a single connected white panel and
return its title crop with geometric evidence. Search bounds, panel dimensions,
title offset and color/density checks are explicit calibration inputs; no game
coordinates are built in. Clipped, fragmented, sparse or ambiguous panels abstain.
The helper does not capture, hover, OCR or enrich a Board. Combined synthetic
title/layout tests pass: 81 tests.

Development calibration on the two new hover frames located Tiger and Rhino in
different screen positions, and both exact OCR preparations read each title.
The no-tooltip control produced no panel. Body text occupied more than the
default density allowance; private calibration used white fraction .75 with
edge fraction .8. The optional one-pixel dark-perimeter check rejected these
antialiased panels and remains disabled in that private experiment. These are
calibration samples, not independent accuracy measurements or native integration.
Further layouts, names and negative screens need validation before a runtime
observer can use this evidence for gameplay decisions.
The combined repository suite now passes 1,834 tests with one expected skip.

## Verified restart through a fresh arena result, 2026-10-17

After the private turn-eleven calibration, the resumed arena reached RESULT with
11 actions/11 acknowledgments in 77 observations. The batch observed the terminal
again, requested normal shutdown, verified disappearance and cold-launched the
next game. That fresh session waited through 30 unknown loading observations,
opened Play/Arena, selected and confirmed its name and completed nine shop turns.
It stopped at RESULT with 90 actions/90 acknowledgments in 854 observations and
416.91 seconds. Independent final inspection confirmed turn nine and three
trophies. No gameplay input came from the agent.

The full batch completed two arenas with 101 actions/101 acknowledgments, 932
observations (including one fresh restart check), one shutdown attempt and no
pending action/error. Only the second arena began fresh; this does not rewrite
earlier interrupted sessions as uninterrupted runs. Together with the earlier
October 17 baseline and two October 4 runs, four complete fresh runs are verified.

The sanitized fresh-arena fixture preserves all 90 policy-generated actions,
acknowledgments and 854 polls under real policy/session replay with mocked IO.
An unreadable late-roll gold receipt instead stops at 88 attempted/87 acknowledged
actions, retaining the pending roll without reading a later terminal. Export
checks exclude capture paths, chosen names and account metadata. Both new tests
pass. The subsequent six-arena native soak completed, as recorded below.
The offline wheel also builds without network
dependencies and contains the required modules without private assets.

## Six consecutive native arenas, 2026-10-17

One bounded standalone invocation completed six fresh main-menu-to-RESULT arenas
and five verified client restarts: 490 actions, 490 acknowledgments and 4,392
observations, including five fresh restart checks. There were no pending actions,
errors or agent gameplay inputs. Session elapsed time summed to 2,154.82 seconds;
client startup/shutdown time is additional. All six began at the main menu.

| Arena | Actions acknowledged | Final turn | Trophies |
| --- | ---: | ---: | ---: |
| 1 | 81 | 8 | 1 |
| 2 | 81 | 8 | 0 |
| 3 | 85 | 8 | 1 |
| 4 | 70 | 7 | 0 |
| 5 | 91 | 9 | 4 |
| 6 | 82 | 8 | 0 |

Independent final inspection confirmed the last turn-eight loss with zero
trophies. Four unidentified level-one sales returned two gold and were
acknowledged live, validating the bounded receipt change beyond offline replay.
The run also covered the life-reward phase. Its 856 recorded changed-board frames
and detailed logs remain private. Ten complete fresh runs are now verified across
October 4 and 17, but the six losses and low trophy counts show the stat baseline
is still a weak strategy. This is bounded control evidence, not competitive play
or reliability across arbitrary layouts, packs or unseen states.

## Frozen foreground recognition evaluation, 2026-10-17

A separate foreground-shape/color experiment suppressed background and shared
hat regions. Its fixed metric ranked 70 of 83 calibration identities correctly,
but accepted four unfamiliar pets; calibrated rejection retained 62 of 83 with
zero observed unsafe accepts. These are development metrics, not test accuracy.
The candidate does not recognize emptiness and remains disabled in production.

After independent source/provenance review, the complete executable closure and
policy were frozen before final annotation. Root visually annotated 24 slots in
three previously reserved panels: 15 known pets, three unfamiliar Whale/Kangaroo
examples and six empties. Previously used Sheep/Worm negative identities were
excluded before scoring. Metadata preflight passed, but the unchanged dataset
gate refused final evaluation because resized inputs were near-duplicates across
training/calibration/test splits. No predictions or final accuracy were produced.
No rows, crops, features or thresholds were changed to bypass the refusal. New
independent appearance data is required before this candidate can be assessed.
The subsequent duplicate-only audit found overlap in 10 of 15 known crops and
all six empties, including seven and four exact copies respectively. None of
the three unfamiliar crops overlapped. The audit computed no classifier scores;
there was no alternate-subset evaluation.

## Start another arena from an existing result, 2026-10-17

The PowerShell launcher now enables `--restart-completed`. One initial clean
zero-action RESULT can receive a fresh terminal check and normal close/relaunch
before the actual arena. The old terminal remains a separate exportable receipt;
it does not count as gameplay. Both observations consume the shared poll budget,
and time, Escape, stop-file and focus guards remain active. A second empty
terminal fails without another bootstrap. Batch summaries separate this one
bootstrap allowance from their N-minus-one normal restarts. Direct CLI behavior
without the flag remains unchanged, and `--no-restart-completed` overrides the
launcher default.

Root review caught an exception-accounting issue: an unexpected actual-arena
exception must propagate with unknown counts, rather than fabricate a zero-action
bootstrap error. That fix and a dispatch-then-error regression are included.
All 1,857 repository tests pass with one expected skip. The final native test
started from the completed soak result, verified it twice, closed it once and
launched a fresh arena using the standalone program. A first activation attempt
was denied focus and stopped before input; after setup restored game focus, the
test resumed as a new invocation. That invocation completed a fresh arena with
92 actions/92 acknowledgments in 811 observations and 403.07 session seconds.
The two bootstrap observations bring its total to 813. It requested one normal
close, retained the old zero-action/one-poll result receipt, and stopped with no
pending action or error. Independent inspection confirmed Game Over at turn nine
with one trophy. All gameplay and the completed-client restart came from the bot.

Eleven complete fresh native runs are now verified across October 4 and 17.
The final wheel builds offline, includes the updated bootstrap implementation,
and contains no private images, calibration or trained models. All project
controller/training jobs were stopped before the 5 p.m. control deadline; the
game remains open at its completed result. Both requested October 17 continuation
wakeups were deleted after firing, with no further schedule created.

The delivered command is `cd C:\Users\nikhi\Downloads\SAP-AI`, then
`.\play.cmd`. Hold Escape until it stops, use a screen corner, or create the
documented STOP file. This is a working calibrated desktop baseline on this
laptop, not a finished competitive player: food, ordering and most abilities
remain unsupported, species recognition has gaps, and learned recognizers remain
disabled. Further work should validate tooltip/target association and readable
shop levels before enabling identity-dependent decisions, then implement food
observation, execution and receipts with current-game evidence.

## October 18: distinguish clear tooltip evidence from detector uncertainty

The pure calibrated panel detector now exposes separate `absent`, `unique`,
`ambiguous` and `unresolved` results. A fragmented search that exceeds its work
bound, white evidence clipped by the search boundary, and a plausible panel with
insufficient edges, perimeter or title geometry cannot masquerade as a cleared
tooltip. Only a unique resolved panel returns a detached title crop. The existing
title-only helper remains available and abstains for every other result.

This supports the next hover association diagnostic: parked-pointer clearance
must be established before accepting a subsequent tooltip appearance. Absence
is relative to the caller's calibrated geometry/color search, not a claim about
the whole screen or the identity of a hovered pet. Search edges must be clear;
the stricter unresolved result may reduce recognition coverage. Synthetic tests
cover the four outcomes, all four clipped edges, work limits and detached crops;
95 tooltip/layout tests pass. No live screen or input was used for this change,
and native identity recognition and policy remain unchanged.

## October 18: optional calibrated food observations

Profiles can now declare separate food portrait/price regions, explicit empty
references and food identity image variants. A known food template must beat
empty evidence to establish occupancy; a price or empty-template mismatch alone
cannot establish an unfamiliar item. Competing food identities retain known
occupancy while leaving identity unknown. Prices come only from bounded numeric
OCR or an explicitly verified `food.N.cost` reference, never from identity.

Food observations remain optional and legacy profiles/boards keep their old
serialization when no food is configured. Calibration overlays show the new
regions, and every food image variant and price reference is checked before
window activation. No effect, stat bonus, game balance or native geometry is
inferred, so these observations alone cannot authorize a food purchase. The
default desktop policy is unchanged. All 404 focused food/vision/state/CLI tests
pass using synthetic images; no private calibration or artwork was committed
and no live food validation has occurred.

## October 18: explicit food input and acknowledgment contract

The runtime can dispatch a `buy_food` action supplied by an explicit adapter:
select its calibrated food source, then its selected teammate. Both rectangles
are revalidated before the first click, and each click keeps the normal stop,
focus and geometry guards. A partial failure stays pending without retry. Food
legality now also requires a known positive turn and enough food occupancy/
identity evidence to distinguish the selected removal from surviving offers.

Acknowledgment requires the explicitly observed price, source removal (gap or
left compaction), exact configured target stat gains, and unchanged teammates,
pet offers, surviving food and other counters. It abstains on capped/extra
ability effects rather than guessing their rules. Review found a mutable-profile
geometry gap; malformed or out-of-bounds destinations now reject before source
selection. All 587 focused food/state/runtime/session tests pass, including
full mocked sessions, failed receipts, preflight price changes, stop-between-
clicks and partial input failures.

This is a tested adapter protocol, not native food gameplay evidence. Prices and
bonuses in tests are deliberately arbitrary. The native perceptor still provides
no food effect/bonus contract and the default policy never selects food, so
normal `play.cmd` behavior remains unchanged. Current food mechanics, stat caps,
ability interactions, calibrated regions and the physical native sequence still
need direct validation before enabling food in normal play.

## October 18: bounded tooltip appearance sequence diagnostic

The hover development probe now requires explicit private tooltip calibration
matching the profile size and records the exact layout hash before capture.
Each target uses six bounded frames: two parked clear SHOP observations, two
settled tooltip frames, and two parked clear SHOP observations again. The panel
and title pixels/positions must agree between hover samples; full typed Boards
must agree across both clear pairs and the target must remain known occupied.
Tooltip-covered frames never enter Board perception, avoiding OCR over occluded
stats. Captures are copied so reused camera buffers cannot hide a change.

Stale, ambiguous, unresolved, disappearing or changing panels and changed clear
Boards reject the sequence. The report calls success `appearance_sequence_verified`,
keeps unknown fields unknown, and explicitly limits this to development evidence
within the calibrated search; it does not prove target identity or causation.
No species text is interpreted or promoted into gameplay. All 166 focused
probe/association/title/layout tests pass with injected IO, including stop and
private-output checks. New captures remain development diagnostics; the revised
sequence has not yet been run live.

## October 18: one input controller and recorded compatibility check

`play.cmd`, `run --execute` and the hover probe now share a Windows session
input lease. A competing or nested controller fails immediately before window
construction, launch, focus, capture or input. One outer lease covers startup,
completed-client bootstrap, every arena and restart. Normal exit, exceptions
and process exit release ownership without PID files or deleting stale locks.
Passive preview, snapshot capture and offline inspection need no input lease.

Review verified the full lease scope and clarified the passive-capture exception.
Mocked tests cover ownership, pointer-sized handles, conflicts, thread/nested
calls and cleanup. A separate background Win32 synchronization check confirmed
real cross-process rejection and reacquisition after normal release and owner
process exit. That check loaded no game IO and sent no capture or input.
All 2,045 repository tests pass with one expected Windows symlink skip. An
offline wheel build includes the new module and no private assets.

A bounded CPU regression reread all 856 saved changed frames from the six-arena
October 17 soak through the current observer and unchanged private profile.
All 856 typed Boards exactly match their original recorded observations across
all six arenas, in 168.11 seconds with no errors. These are compatibility checks
against previous observer outputs, not independent labels or new accuracy,
strategy or live-gameplay evidence. The private report preserves source/profile
hashes; no frames or calibration were committed. No game window was inspected,
activated or controlled during this work window.

## October 18: validate complete input sequences before selection

The earlier food destination review exposed the same partial-selection risk in
pet purchases, merges, sales and naming. The runtime now resolves and validates
every portrait/button destination and freezes its coordinates before sending the
first click. A missing later slot, malformed noun or out-of-bounds Sell point
rejects the whole sequence without selecting anything. Portraits require typed
in-bounds rectangles; points require two non-boolean integer client coordinates.

The actual click loop keeps its previous order, pointer cleanup bookkeeping,
per-click stop/focus/geometry guards and pending/no-retry behavior. Fresh changed
Boards still defer before point validation. All 305 focused input/runtime/food
execution tests pass, including a mutable button changed after selection and
stop requests between clicks. This correction was tested entirely with mocked
IO; native gameplay has not been repeated during this window.

## October 18: recorded tooltip calibration limitation

The stricter panel inspection was checked against the two previously designated
Tiger/Rhino development hover frames and their parked controls. The previous
wide search returns unresolved boundary evidence on all of them, so that old
layout cannot establish the new probe's clearance requirement. This is a real
coverage limitation; the revised sequence is not yet demonstrated on native
frames. No held-out evaluation data was opened or rescored.

Review rejected a tempting shortcut that ignores small boundary components:
a beveled panel satisfying the existing fill/edge gates can leave a narrow
visible strip when clipped. Discarding that strip by minimum full-panel width
would falsely establish absence. A synthetic regression now preserves this
counterexample; all 49 layout tests pass. Keep unresolved evidence unknown.
Further work needs justified calibration or a separately validated appearance
method, rather than relaxing clearance merely to make these samples pass.

## October 18: opt-in sale-funded replacement searches

`play.cmd --sale-funded-rolls` enables an experimental use of spare gold. After
all existing buy/merge/sell choices, a full team can roll when a supported known
level-one sale would still fund a three-gold replacement after the roll cost.
Ordinary supported pets permit a final roll from three gold; an identified
level-one Pig can fund a roll from two. Unknown species use only the existing
one-gold minimum, and unknown/higher levels cannot fund this option. Only the
roll is proposed; the policy must see a useful new offer before proposing any
sale. Unhelpful results end normally. The option is off by default, preserves
the current launcher behavior, and records its enabled configuration in the
private session-start event for later comparisons.

All 229 focused policy/CLI/replay tests pass, including a complete mocked
roll-sale-purchase sequence with three acknowledgments and a disappointing roll
that never triggers a sale. On 504 distinct typed shop states from the six-arena
development logs, this option changes 38 end-turn proposals into legal roll
opportunities. These are counterfactual proposals, not executed actions or win
rate evidence. The previous visible-replacement allocation candidate changes
zero proposals on the same states, so it remains outside the live launcher.

The same recorded audit finds only 50 identified species among 1,522 occupied
shop-slot observations and no known shop levels; team readings identify 245 of
2,253 occupied observations with 2,071 known levels. These repeated states are
correlated and not an independent accuracy study. They show why more training
of the existing allocation candidate would not resolve the native recognition
blocker. No model was promoted and the frozen failed final study was unchanged.
Native validation of the new optional search remains outstanding.

A bounded CPU development study then compared the fixed option with the baseline
on 25,000 paired synthetic shops, masking shop species/levels and team species
from both policies. It completed in 9.33 seconds with no illegal proposals,
truncations or unexpected stops. The option increased immediate attack+health
utility on 780 seeds, tied on 24,220 and decreased it on none: mean stat gain
rose from 7.58380 to 7.73468. It also spent more gold (mean leftover fell from
1.68896 to 1.28772), with lower stat gain per net gold. This synthetic objective
excludes food, abilities, combat and real opponents; it does not justify a win
rate claim or changing the default. Private reports retain the observation mask,
fixed source hashes and runner provenance. No weights were trained or promoted.

Final integration passes all 2,156 tests with one expected Windows symlink skip
in 30.99 seconds. The final wheel builds offline, includes the controller lease
and optional policy, and contains no private assets. No project CPU/controller
job remains running. Work stayed in the background; renewed desktop permission
was requested because the previous October 17 control window expired.

## October 18: freeze food references in recorded-frame evaluation

Final integration review found that recorded-frame studies rebased the optional
food profile without copying its food images. The evaluator now freezes every
food identity variant and empty reference alongside its existing references.
It accepts only explicit `food.N.occupied`, `food_id` and `cost` labels, with
canonical identity, known occupancy for metadata and prices from one through
99. Unknown, wrong and false-empty outcomes remain distinct. Paired comparison
uses the same validated labels and keeps its existing report shape. Synthetic
tests verify evaluation still uses frozen bytes after originals change or
disappear. All 128 focused evaluation/comparison tests pass.

A separate bounded study manually labeled 93 HUD/occupancy/stat/level fields
across three previously recorded development frames: opening, night and late
shops. Seven fixed brightness/contrast/JPEG variants produced 21 cases and 651
field checks in 48.31 seconds: 508 correct, 143 unknown, zero wrong readings,
zero false-empty readings and zero observation errors. Original opening/night
frames supplied all 53 labeled readings; the original late frame left 12 of
40 readings unknown. One perturbed phase also became unknown. These limitations
need follow-up; the results do not establish unseen-scene accuracy. Species
and native food actions were not evaluated. No held-out final data was reused.

The final repository suite now passes 2,190 tests with one expected Windows
symlink skip in 36.00 seconds. No live game or new model training was involved.

## October 18, morning: bounded tooltip appearance evidence

The strict tooltip inspector still refuses every white component touching the
search boundary, including small HUD glyphs: size alone cannot rule out a
clipped panel. A separate pure candidate inventory now retains all bounded
panels, boundary rectangles and unresolved reasons without weakening that
clearance contract.

A new offline six-frame verifier accepts only supplied clear/clear/hover/hover/
clear/clear snapshots with four matching typed SHOP Boards and a known occupied
target. It requires no bounded panel in clear frames, one stable hover panel,
exact clear restoration including its perimeter, and caller-declared substantial
RGB change plus newly white area. Boundary components may remain unresolved
only when their rectangles and RGB patches are identical across all six frames
and outside the panel/perimeter. Interior ambiguity and work limits refuse.
Shared capture storage refuses; accepted frames are copied before verification.

This is detached development evidence, with no capture, input, OCR, identity
enrichment or gameplay integration. A synchronized unrelated overlay and an
unchanged clipped panel elsewhere remain possible. Existing Tiger/Rhino records
do not contain six independent frames and cannot validate this protocol.
Synthetic adversarial coverage and existing hover diagnostics pass 221 focused
tests. The combined repository suite passes 2,294 tests with one expected Windows
symlink skip in 43.02 seconds; a further synthetic case independently verifies
that a large RGB change with little newly white area still refuses.
Work remained in the background; no October 18 native game validation occurred.

## October 18, morning: explain animated stat abstentions

The original late-shop development frame's 12 unknown fields were traced to a
tier-dismissal animation, not calibration drift. Six stat crops contain clipped
numerals or tall bright badge-rim fragments; partial slot stats are deliberately
withheld together. With identical source/profile hashes and no intervening
action, the next saved frame 1.341 seconds later reads all ten shop stats. A
further stable observation acknowledges the tier dismissal. Private diagnostics
retain frame/crop hashes, masks and event provenance. This explains that selected
development failure; it does not alter the frozen perturbation-study scores.

Six synthetic integration regressions exercise the real clipping preprocessing,
cached mocked OCR, perception, policy and preview session. Top/bottom clipping
and separate edge fragments leave occupancy unknown and authorize no purchase;
fresh stable pixels recover the proposal with zero native input. No numeric
thresholds, calibration or production stat behavior were changed.

## October 18, morning: private tooltip replay command

`tools/check_desktop_tooltip_reveal.py` checks an explicitly declared private
six-frame development manifest, with four caller-annotated Boards tied to clear
frame hashes. It validates all declarations before frame reads, checks all file
hashes before decoding, bounds PNG bytes/pixels and refuses existing outputs.
Reports retain limits and provenance, mark caller Boards unverified, and make
no clearance or identity claim. Only success saves a title crop. No window,
capture, perception, OCR or input call is involved. The pure verifier now also
bounds full-frame copying and conservatively refuses potentially shared array
storage without an expensive exact-overlap search.

A separate static development check reproduced the old strict status/reason
on all seven supplied frames and found bounded candidates only in the two
known hover images. Both before/hover pairs fail boundary invariance: one has
changed pixels and the other changed geometry. They are not complete six-frame
sequences. A conservative search expansion fitted strict absent/unique results
to one pair, while the other expanded to an image boundary and stayed unresolved.
This is calibration exploration on development images, not unseen validation;
no native profile, threshold, species model or gameplay default was changed.
All 264 focused tooltip/inventory/replay/native-probe mock tests pass; runtime
package versions and private IO helper hashes are retained in replay reports.

## October 18, morning: bounded recovery for incomplete shops

The standard CLI now waits through a bounded episode of incomplete SHOP readings
instead of immediately ending when the stat policy cannot decide. It requires
a positive readable turn and the baseline's existing gold/occupancy/stat
requirements. Missing fields remain unknown; fresh complete observations must
earn stability again. The default limit is 30 nonpending incomplete observations,
configurable with `--max-incomplete-shop-polls`; persistent failure stops with
`shop_observation_timeout` and a failing CLI result. Pending acknowledgment,
transition, unknown-phase, overall poll/time and stop limits keep precedence.
Direct generic sessions remain unchanged unless explicitly configured.

Typed and mocked CLI regressions cover changing incomplete fingerprints,
readable-turn recovery, permanent failure, fresh stability, original pending
timeouts, callback errors and stop controls. The 534 focused state/session/CLI/
export tests pass. An audit of the six earlier soak logs found 100 incomplete
SHOP polls, all with a pending action; none demonstrates the new predecision
recovery. A bounded replay then reproduced all six recorded timelines exactly:
4,387 polls, 490 actions and 490 acknowledgments in 0.239 seconds, with the new
readiness guard enabled. The initial replay attempt used the generic 25-poll
startup limit and stopped before recognition; the successful report explicitly
uses the CLI's 300-poll startup allowance. Private reports preserve both.
These are compatibility and preventive tests, not new native gameplay evidence.
The full repository suite passes 2,381 tests with one expected Windows symlink
skip in 38.69 seconds. An offline wheel builds successfully, contains the new
tooltip module and controller lease, and includes no private image/model/log
assets. No native application was opened or controlled.

## October 18, morning: background launcher preflight

`.\play.cmd --check` now validates the calibrated profile, every configured
reference image, Python dependencies and the bounded hidden Tesseract version
query, then reports expected client size and STOP-file status. It returns before
input ownership, window/controller construction, Steam startup, capture or input.
It preserves STOP, session receipts, last-frame images and recording directories.
Passing checks is explicitly separate from live validation. Mocked failure and
preservation cases pass with the existing launcher tests (110 focused tests).

The actual PowerShell wrapper passed on this laptop using the private native
profile and installed OCR executable. This was a background resource check;
no game screen was inspected and no native gameplay was performed. The working
agreement above now reflects today's background-only continuation and the
expired October 17 live-control deadline.

## October 18, morning: stress incomplete-shop recovery

The CPU controller harness now includes transient and persistent incomplete-shop
scenarios. Seeded missing gold, stats or occupancy persist for two to four frames
before recovery, or change every poll until an eight-poll refusal. Only these
new scenarios enable the readiness guard. The fixture independently rejects
premature callbacks against unreadable or queued observations, including when
production readiness is deliberately bypassed. Existing scenario order and
explicit replay behavior are retained; default assignment changes with the
expanded 20-scenario list, and old reports cannot resume with changed sources.

A bounded 120-second CPU run passed all 44,304 synthetic cases, including 2,215
cases of each new scenario, with no unexpected outcomes or invariant failures.
The 108 explicit legacy scenario/seed comparisons reproduced identical full
traces under the current session implementation. These checks exercise controller
contracts, not OCR, mouse IO, combat, training or native win rates.

Final integration passes 2,412 tests with one expected Windows symlink skip in
39.06 seconds. The final offline wheel contains 26 files, imports the desktop
and tooltip modules in isolation, accepts the new preflight flag, and contains
no private image/model/log assets. Background resource preflight passed; no
October 18 screen inspection, gameplay input or new model training occurred.

## October 18, afternoon: reject discarded calibration options

Pet-slot and numeric-reference parsing now reject unsupported keys and nonobject
entries, matching the existing phase/food/profile schemas. Previously a typo
such as `available_from_tun`, `level_region` or `max_distnce` was silently
discarded: the active-turn gate or matching threshold fell back to its default,
and resource preflight could still succeed. Invalid options now fail before
perception, reference loading or native dependencies. Valid version 1 defaults,
optional null regions and serialized fields are unchanged. All 288 focused
schema/preflight/vision tests pass. The native profile still passes the actual
background `play.cmd --check`; no calibration changed.

The afternoon continuation waited for the reported ordinary allowance reset
before substantial work. Both requested one-shot wakeups are now removed. Work
continues in the background; no new live-control authorization has been received.

## October 18, afternoon: optional colored numeric ink

An offline examination of seven designated development images found 30 explicit
team `Lvl 1` badges and five empty-opening controls. The generic numeric path
rejects their level crops because separate tall white badge borders touch the
crop edges, despite complete central numerals. Existing verified RGB references
already recover many level-1 observations; neither shop dice nor ability-body
phrases independently establish a pet's level. No real level 2/3 examples were
available.

Profiles can now opt into strict per-field `numeric_colors` records, specifying
inclusive RGB ink bounds. Only the OCR crop is masked; original RGB reference
matching, ambiguity, disagreement and fallback semantics are preserved. Blank
or dense masks decline rather than invert into dark holes, and the existing
whole-glyph clipping checks remain. The absent configuration preserves legacy
serialization/readings; the native profile and game defaults remain unchanged.
All 535 focused color/vision/runtime/preflight tests pass with synthetic pixels
and mocked OCR processes.

A fixed development-only comparison removed level references from both observers
and evaluated 35 manually labeled crops across seven fixed perturbations. Of
245 readings, the baseline retained 35 correct absent controls and 210 unknowns;
the color candidate returned 245 correct readings with no unknowns or errors.
Original unperturbed cases changed from five correct/30 unknown to 35 correct.
The bounded run used 47 OCR subprocesses in 8.96 seconds. These correlated,
level-1-only development results do not measure unseen accuracy or improve the
reference-assisted live baseline by the same amount. No filter/model was
promoted; real higher-level and broader negative validation remain necessary.

## October 18, afternoon: unambiguous profile JSON

Profile loading now rejects repeated decoded JSON keys at every nesting level.
Previously the last value silently won, allowing a second `calibrated` flag,
slot availability, level region or reference threshold to replace the first.
Equal-valued and escaped duplicate keys also fail before profile conversion;
the same field names in separate objects remain valid. UTF-8 BOM profiles made
by PowerShell are now accepted alongside ordinary UTF-8, with Unicode paths,
legacy defaults and save/roundtrip behavior preserved. All 379 focused JSON,
schema, numeric-color, vision and preflight tests pass without native IO.

## Remaining milestones

1. Continue the private 2560x1440 native profile validation as new states appear.
   The older 2048x1152 Computer Use profile cannot drive native Windows input.
2. Verify OCR, empty slots, species, and proposals across both day and night
   backgrounds. Include naming, battle, victory, and tier-unlock negatives.
3. Repeat fresh-start-through-terminal validation across independent arenas
   and newly observed layouts. Eleven uninterrupted fresh runs are now verified; broader
   unattended reliability and competitive performance remain unmeasured.
4. Validate naming and further variations of results and shop layouts. A loss
   terminal, victory/draw/defeat continuation, tier dismissal, and the fifth
   shop slot now have limited native evidence. Broaden this across new runs.
5. Broaden species/level recognition and strategic play (food, abilities,
   ordering), using an observation representation compatible with the real game.

Do not run the toy DQN against the real board. Keep `.local/` captures and
profiles private. Use Computer Use skill for agent-directed UI interaction;
do not automate authentication. Native runtime tests must remain mocked unless
an explicit, observed game state permits a live test.
