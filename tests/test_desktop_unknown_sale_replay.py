"""Recorded native sale window replayed offline; no new live acknowledgment."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from desktop_session import DesktopSession, action_acknowledged
from desktop_state import Action, Board, DesktopPolicy, PetSlot, observed_sale_income
from tools.export_desktop_replay import export_session


@pytest.fixture
def recording():
    data = json.loads((Path(__file__).parent / "fixtures/desktop_unknown_pig_sale.json").read_text())
    boards = [Board.from_dict(row) for row in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"] for _ in range(count)]
    return data, frames


def replay(frames, *, max_actions=1, action_max_polls=10):
    observations, sent, events = iter(frames), [], []
    result = DesktopSession(lambda: next(observations), sent.append, DesktopPolicy(),
                            max_actions=max_actions, max_polls=len(frames),
                            action_max_polls=action_max_polls,
                            clock=lambda: 0., sleep=lambda _: None,
                            event_callback=events.append).run()
    return result, sent, events


def test_recorded_unknown_level_one_sale_replays_current_receipt_without_rewriting_native_failure(recording):
    data, frames = recording
    before, transient, after = frames[0], frames[3], frames[-1]
    assert len(frames) == 7 and data["observation_runs"] == [[0, 3], [1, 1], [2, 3]]
    assert (before.turn, before.gold, after.gold) == (6, 8, 10)
    assert before.team[3] == PetSlot(True, None, 4, 1, 1)
    assert transient.team[3] == PetSlot(None)
    assert after.team[3] == PetSlot(False)
    assert before.shop == transient.shop == after.shop
    assert [p for i, p in enumerate(before.team) if i != 3] == [p for i, p in enumerate(after.team) if i != 3]
    assert "replayed offline" in data["description"]
    assert "not a new live acknowledgment" in data["description"]
    assert "53 actions and 52 acknowledgments" in data["description"]
    assert DesktopPolicy().choose_action(before) == Action("sell", 3)
    result, sent, events = replay(frames)
    assert sent == [Action("sell", 3)]
    assert {key: getattr(result, key) for key in data["expected"] if key != "pending_action"} == {
        key: value for key, value in data["expected"].items() if key != "pending_action"}
    assert (result.actions, result.acknowledgments, result.polls, result.reason) == (1, 1, 7, "max_actions")
    assert result.pending_action is None and result.error is None
    assert result.proposed_action == Action("sell", 3)
    assert [row["poll"] for row in events if row["event"] == "proposed"] == [2]
    assert [row["poll"] for row in events if row["event"] == "acted"] == [2]
    assert [row["poll"] for row in events if row["event"] == "acknowledged"] == [6]
    assert data["actions"] == [{"poll": 2, "action": Action("sell", 3).to_dict(), "acknowledged_poll": 6}]


def test_fixture_is_exact_export_of_the_new_offline_replay_events(recording):
    data, frames = recording
    _, _, events = replay(frames)
    exported = export_session("\n".join(json.dumps(row) for row in events))
    assert {key: value for key, value in exported.items() if key != "description"} == {
        key: value for key, value in data.items() if key != "description"}
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    for board in data["boards"]:
        assert set(board) == {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
        assert all(set(pet) == {"occupied", "species", "attack", "health", "level"}
                   for group in ("shop", "team") for pet in board[group])


@pytest.mark.parametrize("damage", ["gold_zero", "gold_three", "still_occupied", "still_unknown",
                                  "wrong_destination", "known_fish", "unknown_level",
                                  "level_two", "level_three"])
def test_malformed_or_unsupported_sale_receipts_stay_pending_without_retry(recording, damage):
    _, frames = recording
    before, after = frames[0], frames[-1]
    if damage.startswith("gold_"):
        after = replace(after, gold=before.gold + (0 if damage == "gold_zero" else 3))
    elif damage in {"still_occupied", "still_unknown"}:
        target = before.team[3] if damage == "still_occupied" else PetSlot(None)
        after = replace(after, team=(*after.team[:3], target, after.team[4]))
    elif damage == "wrong_destination":
        after = replace(after, team=(PetSlot(False), *after.team[1:3], before.team[3], after.team[4]))
    else:
        sold = before.team[3]
        sold = (replace(sold, species="fish") if damage == "known_fish"
                else replace(sold, level={"unknown_level": None, "level_two": 2, "level_three": 3}[damage]))
        before = replace(before, team=(*before.team[:3], sold, before.team[4]))
    action = Action("sell", 3)
    assert not action_acknowledged(before, after, action)
    if damage in {"unknown_level", "level_two", "level_three"}:
        # Real policy protects these levels, so it cannot dispatch this sale.
        # Receipt rejection above still directly checks an already pending one.
        assert DesktopPolicy().choose_action(before) != action
        return
    result, sent, events = replay([before, before, *([after] * 4)], action_max_polls=4)
    assert sent == [action]
    assert (result.reason, result.actions, result.acknowledgments) == ("action_timeout", 1, 0)
    assert result.pending_action == action and result.error is None
    assert not any(row["event"] == "acknowledged" for row in events)


@pytest.mark.parametrize("delta", [1, 2])
def test_unknown_level_one_receipts_require_stable_supported_gold_and_selected_empty_slot(recording, delta):
    _, frames = recording
    before = frames[0]
    after = replace(frames[-1], gold=before.gold + delta)
    assert action_acknowledged(before, after, Action("sell", 3))
    result, sent, events = replay([before, before, frames[2], frames[3], after, after, after])
    assert sent == [Action("sell", 3)]
    assert result.reason == "max_actions" and result.acknowledgments == 1 and result.polls == 7
    assert [row["poll"] for row in events if row["event"] == "acknowledged"] == [6]


def test_observed_bonus_receipt_does_not_raise_unknown_sale_funding_assumption(recording):
    _, frames = recording
    before = frames[0]
    assert observed_sale_income(before.team[3]) == 1
    assert DesktopPolicy().choose_action(replace(before, gold=1)) == Action("end_turn")
    assert DesktopPolicy().choose_action(replace(before, gold=2)) == Action("sell", 3)


def test_single_receipt_frame_does_not_acknowledge_or_start_followup_purchase(recording):
    _, frames = recording
    before, after = frames[0], frames[-1]
    result, sent, events = replay([before, before, after])
    assert result.reason == "max_polls" and result.acknowledgments == 0
    assert result.pending_action == Action("sell", 3) and sent == [Action("sell", 3)]
    assert not any(row["event"] == "acknowledged" for row in events)
