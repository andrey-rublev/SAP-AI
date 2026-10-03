"""Sanitized terminal-run controller evidence, without OCR or desktop IO.

The policy repeats recorded choices. These tests verify observation-driven
sequencing and stopping, not perception accuracy or the strength of those choices.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from desktop_session import DesktopSession, PHASE_ACTION_KINDS, action_acknowledged
from desktop_state import Action, Board, Phase


@pytest.fixture
def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_turns_4_to_terminal.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(board) for board in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(data, frames):
    shop_actions = iter(Action.from_dict(row["action"]) for row in data["actions"]
                        if row["action"]["kind"] not in PHASE_ACTION_KINDS.values())

    class RecordedPolicy:
        def choose_action(self, board):
            return next(shop_actions, None)

    observations, clicks, events = iter(frames), [], []
    runner = DesktopSession(
        lambda: next(observations), clicks.append, RecordedPolicy(),
        event_callback=events.append, clock=lambda: 0., sleep=lambda _: None,
        max_actions=100, max_polls=len(frames),
        phase_actions={phase: Action(kind) for phase, kind in PHASE_ACTION_KINDS.items()},
    )
    return runner.run(), clicks, events


def test_recorded_terminal_run_preserves_all_action_and_ack_polls_without_restart(recording):
    data, frames = recording
    # Leave calibrated menu phases available after Result. They must stay unread.
    trailing_menu = [Board(phase) for phase in
                     (Phase.MAIN_MENU, Phase.MAIN_MENU, Phase.PLAY_MENU,
                      Phase.PLAY_MENU, Phase.ARENA_SETUP, Phase.ARENA_SETUP)]
    result, clicks, events = replay(data, frames + trailing_menu)
    expected = data["expected"]
    assert {field: getattr(result, field) for field in
            ("actions", "acknowledgments", "polls", "reason")} == {
        "actions": 41, "acknowledgments": 41, "polls": 375, "reason": "result",
    } == {field: expected[field] for field in
          ("actions", "acknowledgments", "polls", "reason")}
    assert result.pending_action is expected["pending_action"] is None
    assert result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert [(event["poll"], Action.from_dict(event["action"])) for event in events
            if event["event"] == "acted"] == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    acknowledged = [event for event in events if event["event"] == "acknowledged"]
    assert [(event["poll"], Action.from_dict(event["action"])) for event in acknowledged] == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert all(Board.from_dict(event["board"]) == frames[event["poll"] - 1]
               for event in acknowledged)
    assert result.last_board == frames[-1] == Board(Phase.RESULT)
    assert {frame.turn for frame in frames if frame.phase == Phase.SHOP} == {4, 5, 6, 7}
    # Strict schema checks keep paths, timings, screenshots, and account metadata out.
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    assert all(set(board) <= {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
               for board in data["boards"])
    assert all(set(pet) <= {"occupied", "species", "attack", "health", "level"}
               for board in data["boards"] for row in ("shop", "team")
               for pet in board.get(row, []))
    assert all(set(row) == {"poll", "action", "acknowledged_poll"}
               and set(row["action"]) <= {"kind", "slot", "target"}
               for row in data["actions"])
    assert set(expected) == {"actions", "acknowledgments", "polls", "reason", "pending_action"}


def test_recorded_tier_dismissal_rejects_stale_turn_without_retry(recording):
    data, frames = recording
    row = data["actions"][31]
    action = Action.from_dict(row["action"])
    assert (action, row["poll"], row["acknowledged_poll"]) == (Action("dismiss_tier"), 281, 284)
    before, after = frames[row["poll"] - 1], frames[row["acknowledged_poll"] - 1]
    assert before.phase == Phase.TIER_UNLOCK and before.turn is None
    assert (after.phase, after.turn, after.gold) == (Phase.SHOP, 7, 10)
    previous_turn = next(frame.turn for frame in reversed(frames[:row["poll"]])
                         if frame.phase == Phase.SHOP)
    assert previous_turn == 6
    assert action_acknowledged(before, after, action, previous_shop_turn=previous_turn)
    stale = replace(after, turn=previous_turn)
    assert not action_acknowledged(before, stale, action, previous_shop_turn=previous_turn)
    # Preserve all earlier recorded delays and clicks; only the new turn evidence fails.
    result, clicks, events = replay(data, frames[:row["poll"]] + [stale] * 50)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "action_timeout", 32, 31, 331,
    )
    assert result.pending_action == action
    assert result.error is None
    assert clicks == [Action.from_dict(item["action"]) for item in data["actions"][:32]]
    assert [event["poll"] for event in events if event["event"] == "acted"] == [
        item["poll"] for item in data["actions"][:32]
    ]
    assert [event["poll"] for event in events if event["event"] == "acknowledged"] == [
        item["acknowledged_poll"] for item in data["actions"][:31]
    ]
