"""Loading grace only before recognition; no input on unknown startup screens."""
from dataclasses import replace

import pytest

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase


def replay(frames, **kwargs):
    frames = iter(frames)
    clicks, events = [], []
    result = DesktopSession(lambda: next(frames), clicks.append, DesktopPolicy(),
        phase_actions={Phase.MAIN_MENU: Action("open_play")}, clock=lambda: 0.,
        sleep=lambda _: None, event_callback=events.append, **kwargs).run()
    return result, clicks, events


def test_cold_loading_can_outlast_normal_unknown_limit_before_first_recognition():
    unknown, menu = Board(Phase.UNKNOWN), Board(Phase.MAIN_MENU)
    frames = [unknown] * 30 + [menu, menu, Board(Phase.PLAY_MENU), Board(Phase.PLAY_MENU)]
    result, clicks, events = replay(frames, max_initial_unknown_polls=40, max_actions=1, max_polls=len(frames))
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == ("max_polls", 1, 1, 34)
    assert clicks == [Action("open_play")]
    assert [row["poll"] for row in events if row["event"] == "acted"] == [32]
    assert result.pending_action is None


def test_normal_unknown_limit_returns_after_any_known_phase_without_more_input():
    frames = [Board(Phase.BATTLE)] + [Board(Phase.UNKNOWN)] * 25
    result, clicks, _ = replay(frames, max_initial_unknown_polls=300, max_polls=300)
    assert (result.reason, result.polls, result.actions) == ("unknown_timeout", 26, 0)
    assert not clicks


def test_unrecognized_startup_is_still_bounded_without_actions_or_retries():
    result, clicks, _ = replay([Board(Phase.UNKNOWN)] * 8, max_initial_unknown_polls=8)
    assert (result.reason, result.polls, result.actions, result.acknowledgments) == ("unknown_timeout", 8, 0, 0)
    assert result.pending_action is None and not clicks


def test_startup_grace_never_expands_total_poll_or_stop_budget():
    frames = [Board(Phase.UNKNOWN)] * 10
    result, clicks, _ = replay(frames, max_initial_unknown_polls=300, max_polls=4)
    assert (result.reason, result.polls, result.actions) == ("max_polls", 4, 0)
    stopped, clicks, _ = replay(frames, max_initial_unknown_polls=300, should_stop=lambda: True)
    assert (stopped.reason, stopped.polls, stopped.actions) == ("stopped", 0, 0)
    assert not clicks


def test_initial_unknown_defaults_preserve_prior_timeout():
    result, clicks, _ = replay([Board(Phase.UNKNOWN)] * 25)
    assert (result.reason, result.polls) == ("unknown_timeout", 25)
    assert not clicks


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "3"])
def test_startup_poll_limit_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="max_initial_unknown_polls"):
        replay([], max_initial_unknown_polls=value)


def test_uncalibrated_food_stays_absent_from_session_events_and_final_receipt():
    empty = PetSlot(False)
    before = Board(Phase.SHOP, gold=3, turn=1, shop=(PetSlot(True, "fish", 2, 3),), team=(empty,) * 5)
    after = replace(before, gold=0, shop=(empty,), team=(before.shop[0], *before.team[1:]))
    result, _, events = replay([before, before, after, after, after], max_actions=1)
    assert (result.actions, result.acknowledgments) == (1, 1)
    assert "food" not in result.to_dict()["last_board"]
    assert all("food" not in row["board"] for row in events if "board" in row)
