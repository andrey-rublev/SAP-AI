# Desktop AI progress

## Working agreement

**Latest user instruction, 2026-09-27:** the user explicitly authorized screen
control again for this task and asked to remove the scheduled task. The
`advance-sap-desktop-ai` heartbeat was deleted through the app. Game calibration
and controlled live validation may resume; the earlier background-only
restriction has been superseded. Keep captures and calibration private.

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
  permission above. The next launch reached Steam's sign-in window; live
  calibration is waiting for the user to sign in and open the game.
- Bounded each Tesseract subprocess to three seconds. A timed-out crop now
  aborts observation immediately, producing a clear session error without
  further OCR retries or mouse input. Other OCR failures remain unknown readings.
- Validation: **361 tests passed**, including four new mocked OCR regression
  tests. No private captures or calibration were added to Git. No real purchase
  or autonomous game turn has been verified.

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

## Next useful milestones

1. Capture a stable shop image and establish a private local profile using the
   actual client resolution and geometry. Obtain empty team/empty shop samples.
2. Verify every OCR reading and proposed action against recorded real frames,
   including negative frames (menus/battle) and post-purchase/roll frames.
3. Observe one real purchase and confirm its expected gold and slot changes.
   Then verify roll, end-turn, and the following shop without duplicate inputs.
4. Add calibrated naming/dialog/result handling and changing shop-slot counts.
   Record verified multi-turn playback before claiming autonomous runs.
5. Broaden species/level recognition and strategic play (food, abilities,
   ordering), using an observation representation compatible with the real game.

Do not run the toy DQN against the real board. Keep `.local/` captures and
profiles private. Use Computer Use skill for agent-directed UI interaction;
do not automate authentication. Native runtime tests must remain mocked unless
an explicit, observed game state permits a live test.
