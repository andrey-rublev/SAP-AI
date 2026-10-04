"""Replay observed native name selection and battle handoff without desktop IO."""
import json
from pathlib import Path

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, Phase


def recording():
    data = json.loads((Path(__file__).parent / "fixtures" / "desktop_naming_to_battle.json").read_text())
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(frames, **bounds):
    observations, clicks, events = iter(frames), [], []
    result = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy(),
        phase_actions={Phase.NAMING: Action("choose_name"),
                       Phase.NAMING_READY: Action("confirm_name"),
                       Phase.ROUND_RESULT: Action("continue_round")},
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
        max_polls=len(frames), **bounds,
    ).run()
    return result, clicks, events


def test_native_name_selection_and_confirmation_preserve_recorded_acknowledgments():
    data, frames = recording()
    # The native run later stopped at an unrecognized night shop after Continue.
    assert data["expected"]["reason"] == "unknown_timeout"
    assert data["expected"]["pending_action"] == Action("continue_round").to_dict()
    result, clicks, events = replay(frames, max_actions=2)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_actions", 2, 2, 30,
    )
    assert result.last_board.phase == Phase.BATTLE
    assert result.pending_action is None and result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"][:2]]
    assert [(row["poll"], row["action"]["kind"]) for row in events if row["event"] == "acted"] == [
        (2, "choose_name"), (5, "confirm_name"),
    ]
    assert [(row["poll"], row["action"]["kind"]) for row in events if row["event"] == "acknowledged"] == [
        (row["acknowledged_poll"], row["action"]["kind"]) for row in data["actions"][:2]
    ] == [(4, "choose_name"), (29, "confirm_name")]


def test_native_unrecognized_name_selection_never_confirms_or_retries():
    _, frames = recording()
    frames = [Board(Phase.UNKNOWN) if board.phase == Phase.NAMING_READY else board
              for board in frames]
    result, clicks, events = replay(frames, action_max_polls=6)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "action_timeout", 1, 0, 8,
    )
    assert result.pending_action == Action("choose_name") and result.error is None
    assert clicks == [Action("choose_name")]
    assert [row["poll"] for row in events if row["event"] == "proposed"] == [2]


def test_native_naming_fixture_has_no_name_text_or_private_metadata():
    data, frames = recording()
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    assert len(frames) == data["expected"]["polls"] == 71
    assert all(Board.from_dict(row).to_dict() == row for row in data["boards"])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"} for row in data["actions"])
    assert all(Action.from_dict(row["action"]).to_dict() == row["action"] for row in data["actions"])
