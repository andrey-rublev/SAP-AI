# Level recovery — November 6, 2026

The runtime can now retain a consistently read team level while its badge is
briefly unreadable. This helps resumed games: the policy can still distinguish
an eligible level-one replacement from an upgraded teammate.

## Evidence rules

- Confirmation requires two consecutive raw observations with the same readable
  level, stats and compatible known species. An unreadable badge breaks an
  unconfirmed streak.
- Confirmed evidence only fills a missing level. A directly read level wins.
- Changed stats, incompatible species, empty or uncertain occupancy, or missing
  team geometry invalidate evidence. A later readable species narrows an
  initially anonymous entry, so it cannot pass to a different named pet.
- Buy, merge, food and sell inputs invalidate their target before any later
  receipt. A turn regression or a new arena resets the cache.
- A preflight lookup never counts as an additional confirmation.

The cache works without tooltip calibration and survives recovery sessions
inside one autopilot process. It is not a disk checkpoint: closing the process
loses this memory. It also does not infer merge experience for an untracked pet.

## Historical coverage check

Replayed six saved October 17 arenas as typed observations, clearing memory on
each recorded input. No images were opened and no OCR or desktop input ran.

| Measure | Count |
| --- | ---: |
| Arenas / shop polls | 6 / 1,212 |
| Occupied team observations | 5,527 |
| Directly readable team levels | 5,305 |
| Team levels available with memory | 5,501 |
| Missing levels recovered | 196 |
| Changed policy proposals | 0 |

Every directly known level was preserved. These are repeated historical
observations, not independently labeled recognition samples. The unchanged
policy proposals mean this check demonstrates continuity rather than a measured
trophy improvement.

Mocked integration tests additionally start with a resumed team whose levels
become unreadable. The policy still selects an eligible replacement, and the
pre-input check accepts the consistent cached evidence. An autopilot test
verifies that recovery reuses the memory while a new arena clears it, including
when tooltip identification is disabled.

The cache cannot detect a replacement that is indistinguishable in every
observed field. It does not improve OCR itself or establish new live-game
reliability. All validation in this pass was offline.
