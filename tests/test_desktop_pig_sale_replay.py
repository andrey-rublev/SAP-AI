"""Actual sale receipt replay; no screenshots, OCR, or native input."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from desktop_session import DesktopSession
from desktop_state import Action, Board


@pytest.fixture
def recording():
    data = json.loads((Path(__file__).parent / "fixtures" / "desktop_pig_sale.json").read_text())
    boards = [Board.from_dict(row) for row in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"] for _ in range(count)]
    return data, frames


def replay(action, frames, **options):
    class SalePolicy:
        def choose_action(self, board):
            return action

    observations, clicks, events = iter(frames), [], []
    session = DesktopSession(lambda: next(observations), clicks.append, SalePolicy(),
                             max_actions=1, max_polls=len(frames),
                             clock=lambda: 0., sleep=lambda _: None,
                             event_callback=events.append, **options)
    return session.run(), clicks, events


def test_recorded_pig_sale_waits_for_empty_slot_and_stable_two_gold_receipt(recording):
    data, frames = recording
    action = Action.from_dict(data["action"])
    assert action == Action("sell", 1)
    assert (frames[0].gold, frames[-1].gold) == (6, 8)
    assert frames[0].team[1].species == "pig" and frames[0].team[1].level == 1
    assert frames[3].team[1].occupied is None
    assert frames[4].team[1].occupied is False
    result, clicks, events = replay(action, frames)
    expected = data["expected"]
    assert {key: getattr(result, key) for key in ("actions", "acknowledgments", "polls", "reason")} == {
        key: expected[key] for key in ("actions", "acknowledgments", "polls", "reason")
    }
    assert clicks == [action] and result.pending_action is None and result.error is None
    assert [row["poll"] for row in events if row["event"] == "acted"] == [expected["acted_poll"]]
    assert [row["poll"] for row in events if row["event"] == "acknowledged"] == [expected["acknowledged_poll"]]
    assert set(data) == {"description", "boards", "observation_runs", "action", "expected"}
    assert all(set(board) == {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
               and all(set(pet) == {"occupied", "species", "attack", "health", "level"}
                       for kind in ("shop", "team") for pet in board[kind])
               for board in data["boards"])


@pytest.mark.parametrize("missing_evidence", ["bonus", "identity"])
def test_recorded_pig_sale_with_incomplete_receipt_stops_without_retry(recording, missing_evidence):
    data, frames = recording
    before, after = frames[0], frames[-1]
    if missing_evidence == "bonus":
        after = replace(after, gold=before.gold + 1)
    else:
        pets = list(before.team)
        pets[1] = replace(pets[1], species=None)
        before = replace(before, team=tuple(pets))
    action = Action.from_dict(data["action"])
    result, clicks, events = replay(action, [before, before] + [after] * 5, action_max_polls=5)
    assert (result.reason, result.actions, result.acknowledgments) == ("action_timeout", 1, 0)
    assert result.pending_action == action and result.error is None
    assert clicks == [action]
    assert not any(row["event"] == "acknowledged" for row in events)
