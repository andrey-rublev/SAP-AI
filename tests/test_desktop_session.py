"""Replay real control sequencing without importing desktop IO or sleeping."""
from dataclasses import replace
from itertools import chain, repeat
import json

import pytest

from desktop_state import Action, Board, PetSlot, Phase
from desktop_session import DesktopSession, action_acknowledged


EMPTY = PetSlot(False)
ANT = PetSlot(True, "ant", 2, 1, 1)
FISH = PetSlot(True, "fish", 2, 3, 1)
DUCK = PetSlot(True, "duck", 2, 2, 1)


def shop(**changes):
    board = Board(phase=Phase.SHOP, gold=10, turn=1, wins=0, lives=5,
                  shop=(ANT, ANT, ANT), team=(EMPTY,) * 5)
    return replace(board, **changes)


class Policy:
    def __init__(self, *actions):
        self.actions = iter(actions)

    def choose_action(self, board):
        return next(self.actions, None)


def session(frames, *actions, **options):
    frames = list(frames)
    replay = iter(chain(frames, repeat(frames[-1])))
    clicks, events = [], []
    options.setdefault("event_callback", events.append)
    options.setdefault("max_polls", 30)
    options.setdefault("sleep", lambda _: None)
    options.setdefault("clock", lambda: 0.0)
    runner = DesktopSession(lambda: next(replay), clicks.append, Policy(*actions), **options)
    return runner, clicks, events


def test_replay_waits_for_stable_shop_and_acknowledges_actions_across_battle():
    before = shop()
    bought = shop(gold=7, shop=(EMPTY, ANT, ANT), team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    battle = replace(bought, phase=Phase.BATTLE)
    next_shop = replace(bought, gold=10, turn=2, shop=(ANT,) * 3)
    rolled = replace(next_shop, gold=9)
    buy, end, roll = Action("buy", 0, 0), Action("end_turn"), Action("roll")
    runner, clicks, events = session(
        [Board(Phase.UNKNOWN), before, before, bought, bought, bought,
         battle, battle, next_shop, next_shop, rolled, rolled, rolled],
        buy, end, roll, max_actions=3,
    )
    result = runner.run()
    assert clicks == [buy, end, roll]
    assert (result.actions, result.acknowledgments, result.reason) == (3, 3, "max_actions")
    assert [e["poll"] for e in events if e["event"] == "acted"] == [3, 6, 10]
    json.dumps(events)
    json.dumps(result.to_dict())


def test_buy_timeout_never_reclicks_or_accepts_unrelated_changes():
    before = shop()
    unrelated = replace(before, gold=7, wins=1)
    runner, clicks, _ = session([before, before, unrelated], Action("buy", 0, 0),
                                action_max_polls=3)
    result = runner.run()
    assert len(clicks) == 1
    assert result.reason == "action_timeout"
    assert result.acknowledgments == 0
    assert result.pending_action == clicks[0]


def test_buy_requires_gold_source_and_target_evidence():
    before = shop()
    correct = shop(gold=7, shop=(EMPTY, ANT, ANT), team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    action = Action("buy", 0, 0)
    assert action_acknowledged(before, correct, action)
    assert not action_acknowledged(before, replace(correct, gold=10), action)
    assert not action_acknowledged(before, replace(correct, shop=before.shop), action)
    assert not action_acknowledged(before, replace(correct, team=before.team), action)
    assert not action_acknowledged(before, replace(correct, turn=2), action)


def test_live_purchase_replay_acknowledges_left_compaction():
    before = shop(shop=(FISH, DUCK, FISH))
    after = shop(gold=7, shop=(DUCK, FISH, EMPTY), team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    action = Action("buy", 0, 0)
    runner, clicks, _ = session([before, before, after, after, after], action, max_actions=1)
    result = runner.run()
    assert clicks == [action]
    assert (result.reason, result.actions, result.acknowledgments) == ("max_actions", 1, 1)


@pytest.mark.parametrize("remaining", [(FISH, FISH, EMPTY), (FISH, EMPTY, FISH)])
def test_duplicate_offers_support_compaction_and_preserved_gaps(remaining):
    before = shop(shop=(FISH, FISH, FISH))
    after = shop(gold=7, shop=remaining, team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    assert action_acknowledged(before, after, Action("buy", 1, 0))


@pytest.mark.parametrize("remaining", [
    (FISH, FISH, EMPTY),  # The duck, rather than the selected fish, disappeared.
    (DUCK, EMPTY, EMPTY),  # Two offers disappeared.
    (FISH, DUCK, EMPTY),  # Remaining offers were reordered.
    (DUCK, replace(FISH, attack=3), EMPTY),
    (DUCK, replace(FISH, species="beaver"), EMPTY),
    (DUCK, replace(FISH, level=2), EMPTY),
    (DUCK, replace(FISH, attack=None), EMPTY),
    (DUCK, replace(FISH, health=None), EMPTY),
    (DUCK, FISH, PetSlot(None)),
    (DUCK, FISH),  # A different calibration geometry is not compaction.
])
def test_compaction_rejects_inconsistent_or_incomplete_survivors(remaining):
    before = shop(shop=(FISH, DUCK, FISH))
    after = shop(gold=7, shop=remaining, team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    assert not action_acknowledged(before, after, Action("buy", 0, 0))


def test_gap_preserving_purchase_also_requires_unchanged_survivors():
    before = shop(shop=(FISH, DUCK, FISH))
    after = shop(gold=7, shop=(EMPTY, ANT, FISH), team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    assert not action_acknowledged(before, after, Action("buy", 0, 0))


def test_compaction_with_unknown_species_uses_only_observed_stats():
    fish, duck = replace(FISH, species=None, level=None), replace(DUCK, species=None, level=None)
    before = shop(shop=(fish, duck, fish))
    after = shop(gold=7, shop=(duck, fish, EMPTY), team=(fish, EMPTY, EMPTY, EMPTY, EMPTY))
    assert action_acknowledged(before, after, Action("buy", 0, 0))
    # Complete numeric evidence remains necessary when species are unavailable.
    missing = replace(after, shop=(replace(duck, health=None), fish, EMPTY))
    assert not action_acknowledged(before, missing, Action("buy", 0, 0))


def test_compaction_still_requires_price_and_matching_target_species():
    before = shop(shop=(FISH, DUCK, FISH))
    after = shop(gold=7, shop=(DUCK, FISH, EMPTY), team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    assert not action_acknowledged(before, replace(after, gold=8), Action("buy", 0, 0))
    assert not action_acknowledged(before, replace(after, team=(DUCK, EMPTY, EMPTY, EMPTY, EMPTY)),
                                   Action("buy", 0, 0))


def test_merge_purchase_can_compact_the_shop():
    before = shop(shop=(FISH, DUCK, FISH), team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    after = replace(before, gold=7, shop=(DUCK, FISH, EMPTY),
                    team=(replace(FISH, attack=3, health=4), EMPTY, EMPTY, EMPTY, EMPTY))
    assert action_acknowledged(before, after, Action("merge", 0, 0))


def test_sell_requires_emptied_slot_and_one_gold():
    before = shop(team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    after = replace(before, gold=11, team=(EMPTY,) * 5)
    assert action_acknowledged(before, after, Action("sell", 0))
    assert not action_acknowledged(before, replace(after, gold=10), Action("sell", 0))
    assert not action_acknowledged(before, replace(after, team=before.team), Action("sell", 0))


def test_merge_requires_target_improvement_as_well_as_purchase_evidence():
    before = shop(team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    purchased = replace(before, gold=7, shop=(EMPTY, ANT, ANT))
    improved = replace(purchased, team=(replace(ANT, attack=3), EMPTY, EMPTY, EMPTY, EMPTY))
    assert not action_acknowledged(before, purchased, Action("merge", 0, 0))
    assert action_acknowledged(before, improved, Action("merge", 0, 0))


def test_roll_can_acknowledge_identical_random_shop():
    before = shop()
    assert action_acknowledged(before, replace(before, gold=9), Action("roll"))
    assert not action_acknowledged(before, replace(before, wins=1), Action("roll"))


def test_end_turn_does_not_resume_on_stale_shop_after_battle():
    before = shop(team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    battle = replace(before, phase=Phase.BATTLE)
    runner, clicks, _ = session([before, before, battle, before],
                                Action("end_turn"), Action("roll"), max_polls=8)
    result = runner.run()
    assert clicks == [Action("end_turn")]
    assert result.reason == "next_shop_timeout"
    assert result.acknowledgments == 1


def test_end_turn_needs_observed_turn_number():
    board = shop(turn=None, team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    runner, clicks, _ = session([board], Action("end_turn"))
    result = runner.run()
    assert result.reason == "turn_unreadable"
    assert not clicks


def test_result_terminates_without_clicking_continue():
    runner, clicks, _ = session([Board(Phase.RESULT)], Action("continue"))
    result = runner.run()
    assert result.reason == "result"
    assert not clicks


def test_unknown_screen_is_bounded():
    runner, clicks, _ = session([Board(Phase.UNKNOWN)], Action("roll"), max_unknown_polls=3)
    result = runner.run()
    assert (result.reason, result.polls) == ("unknown_timeout", 3)
    assert not clicks


def test_preview_proposes_without_acting_after_two_stable_frames():
    runner, clicks, events = session([shop()], Action("buy", 0, 0), preview=True)
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("preview", 2, 0)
    assert result.proposed_action == Action("buy", 0, 0)
    assert not clicks
    assert not any(e["event"] == "acted" for e in events)


def test_unstable_boards_never_trigger_a_click():
    first, second = shop(), shop(gold=9)
    runner, clicks, _ = session([first, second] * 4, Action("roll"), max_polls=8)
    assert runner.run().reason == "max_polls"
    assert not clicks


@pytest.mark.parametrize("stage", ["observation", "action", "policy", "event"])
def test_dependency_errors_stop_without_retries(stage):
    runner, clicks, _ = session([shop()], Action("buy", 0, 0))

    def broken(*args):
        raise RuntimeError("disconnected")

    if stage == "observation":
        runner.observe = broken
    elif stage == "action":
        runner.act = broken
    elif stage == "policy":
        runner.policy.choose_action = broken
    else:
        runner.event_callback = broken
    result = runner.run()
    assert result.reason == stage + "_error"
    assert result.error == "RuntimeError: disconnected"
    assert result.actions == (1 if stage == "action" else 0)
    if stage == "action":
        assert result.pending_action == Action("buy", 0, 0)
    assert not clicks


def test_external_stop_leaves_pending_action_unretried():
    runner, clicks, _ = session([shop()], Action("buy", 0, 0))
    runner.should_stop = lambda: bool(clicks)
    result = runner.run()
    assert result.reason == "stopped"
    assert (result.actions, result.acknowledgments) == (1, 0)


def test_wall_clock_action_timeout_without_poll_budget_exhaustion():
    runner, clicks, _ = session([shop()], Action("buy", 0, 0), action_timeout=0.5)
    now = [0.0]
    runner.clock = lambda: now[0]
    runner.sleep = lambda duration: now.__setitem__(0, now[0] + 0.3)
    result = runner.run()
    assert result.reason == "action_timeout"
    assert result.polls == 4
    assert len(clicks) == 1


def test_slow_action_preflight_does_not_consume_acknowledgment_timeout():
    before = shop()
    bought = shop(gold=7, shop=(EMPTY, ANT, ANT), team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    action = Action("buy", 0, 0)
    runner, clicks, _ = session([before, before, bought, bought, bought], action,
                                action_timeout=0.5, max_actions=1)
    now = [0.0]
    runner.clock = lambda: now[0]
    runner.sleep = lambda duration: now.__setitem__(0, now[0] + 0.1)

    def slow_act(proposal):
        # Simulate expensive OCR revalidation before the actual click.
        now[0] += 5.0
        clicks.append(proposal)

    runner.act = slow_act
    result = runner.run()
    assert clicks == [action]
    assert (result.reason, result.actions, result.acknowledgments) == ("max_actions", 1, 1)
    assert result.polls == 5


def test_rejects_single_frame_control():
    with pytest.raises(ValueError, match="at least 2"):
        DesktopSession(lambda: shop(), lambda action: None, Policy(), stable_frames=1)


@pytest.mark.parametrize("options", [{"poll_interval": float("nan")},
                                    {"action_timeout": float("inf")}])
def test_rejects_unbounded_clock_configuration(options):
    with pytest.raises(ValueError, match="finite"):
        DesktopSession(lambda: shop(), lambda action: None, Policy(), **options)
