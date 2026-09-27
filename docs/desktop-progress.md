# Desktop AI progress

## Working agreement

**Latest user constraint: background development only.** The user is watching a
TV show and explicitly asked us not to control their screen. Do not launch,
focus, capture, or operate desktop apps, and do not run live-control commands.
Continue code, recorded/synthetic-frame tests, mocks, commits, and pushes.
Resume live validation only after the user explicitly authorizes screen use.
The recurring automation has been updated with this restriction.

The user requests continued development toward actual Super Auto Pets desktop
play and immediate pushes for every small completed change, approximately five
meaningful commits per day. Work on `main`, pushing to `origin/main`; never
force-push. Preserve unrelated edits. Test each coherent change before its
commit, then push immediately. Do not manufacture commits to fill a quota.

A thread heartbeat is scheduled five times daily: 09:00, 12:00, 15:00, 18:00,
21:00 in the user's America/Indianapolis timezone. Its prompt resumes from this
file and the chat. Keep these notes current; stop routine development once the
requested live objective is verified complete.

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
