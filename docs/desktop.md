# Desktop controller

The new desktop pipeline is separate from the toy simulator and its DQN. It
reads real attack/health, explicit slot occupancy, optional species/levels, and
the game phase. Its first policy is a conservative stat-based baseline. It does
not yet model food, pet abilities, frozen offers, or team positioning.

```text
game client image -> calibrated Perceptor -> typed Board
   -> two matching observations -> DesktopPolicy -> legal Action
   -> re-read board and verify focus -> mouse input
   -> wait for expected gold/slot/phase changes -> next decision
```

## Install and capture

```powershell
python -m pip install -e ".[live,dev]"
python desktop.py capture --output .local/desktop/shop.png
```

The capture command requires the Super Auto Pets window to be in the foreground.
Live IO is Windows-only. Tesseract must be installed separately; its standard
Windows install location is detected if it is absent from PATH. Local captures,
templates, profiles, and event logs should stay under `.local/desktop/`, which
Git ignores. Do not publish account information or private screenshots.

## Calibrate from actual game frames

A profile is JSON version 1. `image_size` is the exact game **client** size, not
the monitor size. Rectangles use `[x, y, width, height]`; button points use
`[x, y]`. Coordinates are relative to the top-left of the game client.

Fields:

| Field | Contents |
| --- | --- |
| `version` | `1` |
| `image_size` | `[width, height]` from the captured image |
| `calibrated` | `false` until every region and action point is verified |
| `hud` | Rectangles for `gold`, `turn`, and optionally `wins`, `lives` |
| `phase_templates` | Objects containing `phase`, `region`, and a relative `template` image path |
| `shop` | One configuration for each currently visible pet shop slot (up to five) |
| `team` | Exactly five team slot configurations |
| `buttons` | Calibrated `roll`, `end_turn`, optional `sell` points |

Each slot contains `portrait`, `attack`, `health`, and optional `level`
rectangles. `empty_template` references an image of that exact portrait region
when empty. `species_templates` optionally maps species names to matching
portrait images. Reference paths must stay inside the profile directory.

Extract template crops from a screenshot using observed coordinates:

```powershell
python desktop.py template --image .local/desktop/shop.png --region 100 200 80 60 --output .local/desktop/templates/empty-slot.png
```

Those coordinates are an example, not a shipped calibration. An empty shop
slot normally needs a reference captured after a purchase. Empty team slots
need references from an empty board. Include a stable, distinctive shop-only
region as a `shop` phase template. Add `battle` and `result` references from
their respective screens. Avoid animations, pet sprites, hover highlights,
numeric counters, and broad background regions for phase identification.

Matching uses normalized mean pixel distance; `max_distance` defaults to 0.06
and an ambiguity `margin` to 0.015. Verify these against both positive and
negative frames. A template match is not a probability. Profiles are specific
to game resolution and appearance; changes require recalibration.

```powershell
python desktop.py inspect --profile .local/desktop/profile.json --image .local/desktop/shop.png --overlay .local/desktop/overlay.png
```

This command is entirely offline. Inspect its board readings and proposed
action, then view the overlay to confirm every region. Blank OCR is unknown,
not empty. Occupied slots require both attack and health. Merging additionally
requires identified species and known compatible levels.

Numeric regions must include a clear margin around every digit, including
two-digit values. OCR isolates contrasting glyphs, removes frame lines, and
pads the result before one recognition call. Blank or substantially clipped
glyphs return unknown; they are not repaired by guessing a number. Validate
counter zeroes, levels, and larger stats as well as the opening shop.

Optional `numeric_templates` entries provide verified references for individual
fields, for example `{"field": "gold", "value": 0, "template": "gold-zero.png",
"max_distance": 0.006, "margin": 0.01}`. Fields include HUD names and
`team.0.level`/`shop.0.attack` style slot paths. They use the field's existing
region. A match can resolve blank OCR, but conflicting OCR or ambiguous
references remain unknown. Keep reference images beside the private profile.
Shop dice show tier, not pet level.

## Preview and controlled execution

After setting `calibrated` to true and keeping the game foreground:

```powershell
python desktop.py run --profile .local/desktop/profile.json
python desktop.py run --profile .local/desktop/profile.json --execute --max-actions 3
```

The first command proposes one stable action without sending input. The second
executes a bounded session and waits for each action to be acknowledged. Every
step is logged to `.local/desktop/session.jsonl`. `--action-timeout` defaults
to 30 seconds and can be increased for slower OCR. Ctrl+C or moving the mouse to
a screen corner stops control. Losing window focus stops control. The program
does not activate or navigate other applications.

`--max-seconds` defaults to 1800; the deadline is checked before observations
and action dispatch, without interrupting an in-flight OCR call. Creating the
`--stop-file` (default `.local/desktop/STOP`) stops further dispatch at the same
checks. Remove that file explicitly before restarting. The final captured
frame is saved to `--last-frame` for local diagnosis. Pending, unacknowledged
input produces a failing exit status even when the poll budget expires.

Purchases and merges select the shop pet with one click, then click its team
destination. Both clicks check window focus and geometry. If either fails,
the session stops without repeating the input sequence.
Sales select the teammate, then click the calibrated `sell` point. That
button appears after selection; its point must be present before either click.

Purchase acknowledgment checks gold, the destination pet, and removal of the
selected shop offer. Remaining offers may stay in place or shift left, but
their order and observed stats must match. Known species and levels must also
agree; equal stats alone cannot identify an unrecognized species.

Each Tesseract call has a three-second subprocess timeout. If a numeric crop
times out, observation stops immediately and the session reports an error
without issuing another action. This bound is separate from the action
acknowledgment timeout; increasing `--action-timeout` does not extend OCR calls.
An in-memory 128-entry cache reuses raw OCR strings only when dtype, shape, and
every crop byte match. Numeric bounds and reference conflicts are still checked
on every observation. Errors are not cached, and each invocation starts fresh.

Interrupted drags release the mouse button before the session exits. The
session will not repeat a purchase just because its result is slow. It
stops on unreadable boards, mismatched image size, missing calibration, action
timeouts, or terminal result screens. End-turn requires an observed turn
number; after a battle it waits for a stable shop showing a higher turn.
Interstitial control is opt-in: both the recognized phase and every required
button must be in the profile. `naming` uses `name_adjective` and `name_noun`;
`naming_ready` uses `confirm_name`; `round_result` uses `continue_round`; and
`tier_unlock` uses `dismiss_tier`. Each action needs stable recognition and
its expected following phase. Unknown frames never confirm success, repeated
dialogs stop, and transition polling has a finite budget. An unconfigured
dialog stops without clicking. Initial game setup still requires manual
handling. The legacy `play.py --live` bridge remains available, but this
structured pipeline is the path for ongoing desktop development.

## Validation boundaries

The state model, recognition logic, and session are tested with synthetic
images and replayed observations. Native input is tested with mocks. These
tests do not establish correct coordinates, template quality, OCR accuracy, or
successful real-client play. Track observed game evidence and remaining work
in [desktop-progress.md](desktop-progress.md).

For sustained controller stress tests without desktop access:

```powershell
python tools/eval_desktop.py --seconds 7200 --output .local/desktop/soak.json --fail-fast
python tools/eval_desktop.py --replay 17 --scenario wrong_price
```

The seeded harness exercises purchases, shop compaction, merges, sales, rolls,
delayed frames, missing observations, and input/OCR failures. An independent
fixture checks inputs and acknowledged effects. Reports checkpoint atomically;
`--resume` requires unchanged configuration and source hashes. A stop file or
Ctrl+C retains the next unfinished seed. This is a controller contract test,
not a combat simulator, OCR evaluation, model training, or measured win rate.
