"""Replay a native menu-to-existing-shop transition without desktop IO."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, Phase


def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_arena_resume.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(frames, **bounds):
    observations, clicks, events = iter(frames), [], []
    result = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy(),
        phase_actions={Phase.MAIN_MENU: Action("open_play"),
                       Phase.PLAY_MENU: Action("open_arena")},
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
        max_polls=len(frames), **bounds,
    ).run()
    return result, clicks, events


def test_native_arena_resume_can_now_acknowledge_and_stop_at_its_action_bound():
    data, frames = recording()
    # Retain the actual pre-fix failure rather than rewriting its history.
    assert data["expected"] == {
        "actions": 2, "acknowledgments": 1, "polls": 55,
        "reason": "action_timeout", "pending_action": Action("open_arena").to_dict(),
    }
    result, clicks, events = replay(frames, max_actions=2)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_actions", 2, 2, 12,
    )
    assert result.error is None and result.pending_action is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert [(row["poll"], row["action"]["kind"]) for row in events if row["event"] == "acted"] == [
        (2, "open_play"), (5, "open_arena"),
    ]
    assert [(row["poll"], row["action"]["kind"]) for row in events if row["event"] == "acknowledged"] == [
        (4, "open_play"), (11, "open_arena"),
    ]
    assert result.last_board == frames[-1]
    assert result.last_board.turn == 1 and result.last_board.gold == 1


@pytest.mark.parametrize("field", ["turn", "gold"])
def test_native_resume_with_corrupted_shop_counter_never_sends_a_followup(field):
    _, frames = recording()
    frames = [replace(board, **{field: None}) if board.phase == Phase.SHOP else board
              for board in frames]
    result, clicks, events = replay(frames, action_max_polls=8)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "action_timeout", 2, 1, 13,
    )
    assert result.pending_action == Action("open_arena") and result.error is None
    assert clicks == [Action("open_play"), Action("open_arena")]
    assert [row["poll"] for row in events if row["event"] == "proposed"] == [2, 5]


def test_native_startup_fixture_contains_only_typed_relative_evidence():
    data, frames = recording()
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    assert len(frames) == data["expected"]["polls"]
    assert all(Board.from_dict(row).to_dict() == row for row in data["boards"])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"} for row in data["actions"])
    assert all(Action.from_dict(row["action"]).to_dict() == row["action"] for row in data["actions"])
