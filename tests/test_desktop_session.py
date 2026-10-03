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


def phase_actions():
    return {Phase.NAMING: Action("choose_name"),
            Phase.NAMING_READY: Action("confirm_name"),
            Phase.ROUND_RESULT: Action("continue_round"),
            Phase.TIER_UNLOCK: Action("dismiss_tier"),
            Phase.END_TURN_CONFIRM: Action("confirm_end_turn")}


def test_recorded_round_flow_requires_each_calibrated_ready_phase():
    """Replay the observed naming -> battle -> victory -> tier -> shop flow."""
    before = shop(gold=0, team=(FISH, FISH, DUCK, EMPTY, EMPTY))
    naming, ready = Board(Phase.NAMING), Board(Phase.NAMING_READY)
    battle, outcome, tier = Board(Phase.BATTLE), Board(Phase.ROUND_RESULT), Board(Phase.TIER_UNLOCK)
    next_shop = replace(before, turn=2, gold=10)
    runner, clicks, events = session(
        [before, before, naming, naming, naming, ready, ready, ready,
         battle, outcome, outcome, tier, tier, tier, next_shop, next_shop, next_shop],
        Action("end_turn"), phase_actions=phase_actions(),
    )
    result = runner.run()
    assert [action.kind for action in clicks] == [
        "end_turn", "choose_name", "confirm_name", "continue_round", "dismiss_tier",
    ]
    assert (result.reason, result.actions, result.acknowledgments) == ("policy_stopped", 5, 5)
    assert [event["poll"] for event in events if event["event"] == "acted"] == [2, 5, 8, 11, 14]


@pytest.mark.parametrize("phase", [Phase.NAMING, Phase.NAMING_READY, Phase.ROUND_RESULT, Phase.TIER_UNLOCK,
                                  Phase.END_TURN_CONFIRM, Phase.MAIN_MENU, Phase.PLAY_MENU,
                                  Phase.ARENA_SETUP])
def test_interstitial_actions_are_disabled_unless_explicitly_configured(phase):
    runner, clicks, _ = session([Board(phase)])
    result = runner.run()
    assert (result.reason, result.polls) == ("transition_disabled", 2)
    assert not clicks


def test_phase_preview_proposes_only_after_two_ready_frames():
    runner, clicks, _ = session([Board(Phase.ROUND_RESULT)], phase_actions=phase_actions(), preview=True)
    result = runner.run()
    assert (result.reason, result.polls) == ("preview", 2)
    assert result.proposed_action == Action("continue_round")
    assert not clicks


def test_unstable_naming_readiness_never_clicks_confirmation():
    naming, ready = Board(Phase.NAMING), Board(Phase.NAMING_READY)
    runner, clicks, _ = session([naming, naming, ready, naming, ready, naming],
                                phase_actions=phase_actions(), action_max_polls=4)
    result = runner.run()
    assert clicks == [Action("choose_name")]
    assert result.reason == "action_timeout"
    assert result.acknowledgments == 0


def test_unchanged_outcome_is_never_reclicked_even_when_hud_changes():
    outcome = Board(Phase.ROUND_RESULT)
    runner, clicks, _ = session([outcome, outcome, replace(outcome, wins=1),
                                 replace(outcome, wins=2)],
                                phase_actions=phase_actions(), action_max_polls=4)
    result = runner.run()
    assert clicks == [Action("continue_round")]
    assert (result.reason, result.acknowledgments) == ("action_timeout", 0)


def test_outcome_reappearing_before_new_shop_does_not_trigger_second_continue():
    outcome, tier = Board(Phase.ROUND_RESULT), Board(Phase.TIER_UNLOCK)
    runner, clicks, _ = session([outcome, outcome, tier, tier, outcome, outcome],
                                phase_actions=phase_actions())
    result = runner.run()
    assert clicks == [Action("continue_round")]
    assert (result.reason, result.acknowledgments) == ("repeated_transition", 1)


def test_new_stable_shop_resets_transition_budget_and_allows_next_round_continue():
    first = shop(gold=0, team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    second, third = replace(first, turn=2), replace(first, turn=3)
    battle, outcome = Board(Phase.BATTLE), Board(Phase.ROUND_RESULT)
    runner, clicks, _ = session(
        [first, first, battle, outcome, outcome, second, second, second,
         battle, outcome, outcome, third, third, third],
        Action("end_turn"), Action("end_turn"), phase_actions=phase_actions(),
        max_transition_polls=5,
    )
    result = runner.run()
    assert [action.kind for action in clicks] == ["end_turn", "continue_round"] * 2
    assert (result.reason, result.acknowledgments) == ("policy_stopped", 4)


def test_stale_shop_cannot_acknowledge_round_continuation():
    before = shop(gold=0, team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    battle, outcome = Board(Phase.BATTLE), Board(Phase.ROUND_RESULT)
    runner, clicks, _ = session([before, before, battle, outcome, outcome, before],
                                Action("end_turn"), Action("roll"),
                                phase_actions=phase_actions(), action_max_polls=4)
    result = runner.run()
    assert clicks == [Action("end_turn"), Action("continue_round")]
    assert (result.reason, result.acknowledgments) == ("action_timeout", 1)


@pytest.mark.parametrize("phase", [Phase.BATTLE, Phase.UNKNOWN, Phase.NAMING, Phase.ROUND_RESULT])
def test_naming_click_cannot_be_acknowledged_by_unrelated_phase(phase):
    assert not action_acknowledged(Board(Phase.NAMING), Board(phase), Action("choose_name"))


def test_unknown_frame_never_acknowledges_end_turn_even_with_different_counter():
    assert not action_acknowledged(shop(turn=1), Board(Phase.UNKNOWN, turn=2), Action("end_turn"))


def test_tier_dismissal_requires_matching_or_positive_observed_turn():
    action = Action("dismiss_tier")
    assert action_acknowledged(Board(Phase.TIER_UNLOCK, turn=3), shop(turn=3), action)
    assert not action_acknowledged(Board(Phase.TIER_UNLOCK, turn=3), shop(turn=4), action)
    assert not action_acknowledged(Board(Phase.TIER_UNLOCK), shop(turn=None), action)
    assert not action_acknowledged(Board(Phase.TIER_UNLOCK), shop(turn=2), action, previous_shop_turn=2)
    assert action_acknowledged(Board(Phase.TIER_UNLOCK), shop(turn=3), action, previous_shop_turn=2)


def test_transition_budget_does_not_reset_when_phases_flap():
    runner, clicks, _ = session([Board(Phase.BATTLE), Board(Phase.UNKNOWN)] * 4,
                                phase_actions=phase_actions(), max_transition_polls=5)
    result = runner.run()
    assert (result.reason, result.polls) == ("transition_timeout", 5)
    assert not clicks


def test_terminal_result_never_uses_round_continuation():
    runner, clicks, _ = session([Board(Phase.RESULT)], phase_actions=phase_actions())
    assert runner.run().reason == "result"
    assert not clicks


@pytest.mark.parametrize("mapping", [
    {Phase.RESULT: Action("continue_round")},
    {Phase.NAMING: Action("confirm_name")},
    {Phase.SHOP: Action("roll")},
])
def test_phase_plan_rejects_unsupported_or_mismatched_actions(mapping):
    with pytest.raises(ValueError, match="matching action"):
        session([shop()], phase_actions=mapping)


def test_excess_gold_modal_requires_stable_confirmation_before_battle():
    before = shop(gold=2, team=(FISH, EMPTY, EMPTY, EMPTY, EMPTY))
    confirm, battle = Board(Phase.END_TURN_CONFIRM), Board(Phase.BATTLE)
    runner, clicks, events = session([before, before, confirm, confirm, confirm, battle, battle],
                                     Action("end_turn"), phase_actions=phase_actions(), max_actions=2)
    result = runner.run()
    assert clicks == [Action("end_turn"), Action("confirm_end_turn")]
    assert [event["poll"] for event in events if event["event"] == "acted"] == [2, 5]
    assert (result.reason, result.acknowledgments) == ("max_actions", 2)


@pytest.mark.parametrize("destination", list(Phase))
def test_confirm_end_turn_acknowledges_only_expected_forward_phases(destination):
    result = action_acknowledged(Board(Phase.END_TURN_CONFIRM), Board(destination, turn=2),
                                 Action("confirm_end_turn"))
    assert result is (destination in (Phase.NAMING, Phase.BATTLE, Phase.ROUND_RESULT))


def test_unchanged_excess_gold_modal_does_not_retry_confirmation():
    runner, clicks, _ = session([Board(Phase.END_TURN_CONFIRM)], phase_actions=phase_actions(),
                                action_max_polls=3)
    result = runner.run()
    assert clicks == [Action("confirm_end_turn")]
    assert (result.reason, result.acknowledgments) == ("action_timeout", 0)


def test_excess_gold_modal_cannot_be_confirmed_again_before_new_shop():
    confirm, naming = Board(Phase.END_TURN_CONFIRM), Board(Phase.NAMING)
    runner, clicks, _ = session([confirm, confirm, naming, naming, confirm, confirm],
                                phase_actions=phase_actions())
    result = runner.run()
    assert clicks == [Action("confirm_end_turn")]
    assert (result.reason, result.acknowledgments) == ("repeated_transition", 1)


def arena_phase_actions():
    return {Phase.MAIN_MENU: Action("open_play"),
            Phase.PLAY_MENU: Action("open_arena"),
            Phase.ARENA_SETUP: Action("start_arena")}


def test_arena_start_waits_for_each_menu_acknowledgment_and_observed_first_shop():
    menu, play, setup = Board(Phase.MAIN_MENU), Board(Phase.PLAY_MENU), Board(Phase.ARENA_SETUP)
    first = shop()
    bought = replace(first, gold=7, shop=(EMPTY, ANT, ANT),
                     team=(ANT, EMPTY, EMPTY, EMPTY, EMPTY))
    runner, clicks, events = session(
        [menu, menu, play, play, play, setup, setup, setup,
         Board(Phase.UNKNOWN, turn=1), shop(turn=2), shop(turn=2),
         first, first, first, bought, bought, bought],
        Action("buy", 0, 0), phase_actions=arena_phase_actions(), max_actions=4,
    )
    result = runner.run()
    assert clicks == [Action("open_play"), Action("open_arena"),
                      Action("start_arena"), Action("buy", 0, 0)]
    assert (result.reason, result.actions, result.acknowledgments) == ("max_actions", 4, 4)
    assert [event["poll"] for event in events if event["event"] == "acted"] == [2, 5, 8, 14]
    assert [event["poll"] for event in events if event["event"] == "acknowledged"] == [4, 7, 13, 16]


@pytest.mark.parametrize("phase,kind,expected", [
    (Phase.MAIN_MENU, "open_play", Phase.PLAY_MENU),
    (Phase.PLAY_MENU, "open_arena", Phase.ARENA_SETUP),
])
def test_menu_acknowledgment_requires_exact_source_and_next_phase(phase, kind, expected):
    for candidate in Phase:
        assert action_acknowledged(Board(phase), Board(candidate), Action(kind)) is (candidate == expected)
        assert action_acknowledged(Board(candidate), Board(expected), Action(kind)) is (candidate == phase)


@pytest.mark.parametrize("after", [
    Board(Phase.UNKNOWN, turn=1), Board(Phase.SHOP), shop(turn=0), shop(turn=2), shop(turn=9),
    Board(Phase.NAMING), Board(Phase.BATTLE),
])
def test_start_arena_rejects_unknown_and_stale_shop_frames(after):
    assert not action_acknowledged(Board(Phase.ARENA_SETUP), after, Action("start_arena"))


@pytest.mark.parametrize("phase", list(Phase))
def test_start_arena_acknowledges_first_shop_only_from_arena_setup(phase):
    assert action_acknowledged(Board(phase), shop(), Action("start_arena")) is (phase == Phase.ARENA_SETUP)


@pytest.mark.parametrize("phase,action", list(arena_phase_actions().items()))
def test_arena_menu_preview_proposes_without_input(phase, action):
    runner, clicks, _ = session([Board(phase)], phase_actions=arena_phase_actions(), preview=True)
    result = runner.run()
    assert (result.reason, result.polls, result.actions) == ("preview", 2, 0)
    assert result.proposed_action == action
    assert not clicks


@pytest.mark.parametrize("before,after,action", [
    (Board(Phase.MAIN_MENU), Board(Phase.MAIN_MENU, wins=1), Action("open_play")),
    (Board(Phase.MAIN_MENU), Board(Phase.ARENA_SETUP), Action("open_play")),
    (Board(Phase.PLAY_MENU), shop(), Action("open_arena")),
    (Board(Phase.ARENA_SETUP), shop(turn=2), Action("start_arena")),
])
def test_unacknowledged_menu_click_is_never_retried(before, after, action):
    runner, clicks, _ = session([before, before, after],
                                phase_actions=arena_phase_actions(), action_max_polls=3)
    result = runner.run()
    assert clicks == [action]
    assert (result.reason, result.actions, result.acknowledgments) == ("action_timeout", 1, 0)
    assert result.pending_action == action


def test_menu_reappearing_after_acknowledgment_cannot_be_clicked_again():
    menu, play = Board(Phase.MAIN_MENU), Board(Phase.PLAY_MENU)
    runner, clicks, _ = session([menu, menu, play, play, menu, menu],
                                phase_actions=arena_phase_actions())
    result = runner.run()
    assert clicks == [Action("open_play")]
    assert (result.reason, result.acknowledgments) == ("repeated_transition", 1)


def test_arena_menu_actions_share_session_action_budget():
    menu, play = Board(Phase.MAIN_MENU), Board(Phase.PLAY_MENU)
    runner, clicks, _ = session([menu, menu, play, play, play],
                                phase_actions=arena_phase_actions(), max_actions=1)
    result = runner.run()
    assert clicks == [Action("open_play")]
    assert (result.reason, result.actions, result.acknowledgments) == ("max_actions", 1, 1)


def test_stop_after_menu_input_preserves_unacknowledged_action_without_retry():
    runner, clicks, _ = session([Board(Phase.MAIN_MENU)], phase_actions=arena_phase_actions())
    runner.should_stop = lambda: bool(clicks)
    result = runner.run()
    assert clicks == [Action("open_play")]
    assert (result.reason, result.actions, result.acknowledgments) == ("stopped", 1, 0)
    assert result.pending_action == Action("open_play")


def test_terminal_result_stops_pending_arena_start():
    menu = Board(Phase.MAIN_MENU)
    runner, clicks, _ = session([menu, menu, Board(Phase.RESULT)], phase_actions=arena_phase_actions())
    result = runner.run()
    assert clicks == [Action("open_play")]
    assert (result.reason, result.actions, result.acknowledgments) == ("result", 1, 0)
    assert result.pending_action == Action("open_play")
