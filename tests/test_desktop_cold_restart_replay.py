"""One native cold restart through terminal, with real policy and mocked IO."""
from dataclasses import replace
import json
from pathlib import Path

from sap_ai.desktop.session import DesktopSession, PHASE_ACTION_KINDS
from sap_ai.desktop.state import Action, Board, DesktopPolicy, Phase
from tools.export_desktop_replay import export_session


def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_cold_restart_arena_terminal.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"] for _ in range(count)]
    return data, frames


def replay(frames):
    observations, clicks, events = iter(frames), [], []
    result = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy.recorded_baseline(),
        phase_actions={phase: Action(kind) for phase, kind in PHASE_ACTION_KINDS.items()},
        max_actions=300, max_polls=len(frames), max_initial_unknown_polls=300,
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
    ).run()
    return result, clicks, events


def timeline(events, kind):
    return [(row["poll"], Action.from_dict(row["action"])) for row in events if row["event"] == kind]


def test_cold_restart_preserves_all_90_policy_receipts_and_sanitized_export():
    data, frames = recording()
    result, clicks, events = replay(frames + [Board(Phase.MAIN_MENU)] * 2)
    assert data["expected"] == {
        "actions": 90, "acknowledgments": 90, "polls": 854,
        "reason": "result", "pending_action": None,
    }
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == ("result", 90, 90, 854)
    assert result.pending_action is None and result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert timeline(events, "acted") == [(row["poll"], Action.from_dict(row["action"])) for row in data["actions"]]
    assert timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert frames[:30] == [Board(Phase.UNKNOWN)] * 30
    assert frames[30:32] == [Board(Phase.MAIN_MENU)] * 2
    assert timeline(events, "acted")[:3] == [
        (32, Action("open_play")), (35, Action("open_arena")), (38, Action("start_arena")),
    ]
    assert {board.turn for board in frames if board.phase == Phase.SHOP and board.turn is not None} == set(range(1, 10))
    first = next(board for board in frames if board.phase == Phase.SHOP)
    last = next(board for board in reversed(frames) if board.phase == Phase.SHOP
                and board.turn is not None and board.wins is not None and board.lives is not None)
    assert (first.turn, first.gold, first.wins, first.lives) == (1, 10, 0, 5)
    assert all(pet.occupied is False for pet in first.team)
    assert (last.turn, last.wins, last.lives) == (9, 3, 1)
    assert result.last_board == frames[-1] == Board(Phase.RESULT)
    assert len([row for row in events if row["event"] == "observed"]) == len(frames) == 854
    assert not any(row["event"] == "deferred" for row in events)

    # Export actual policy-generated events, with deliberately private extras.
    for row in events:
        row.update(frame_path="PRIVATE_FRAME_SENTINEL", account="PRIVATE_ACCOUNT_SENTINEL",
                   chosen_name="PRIVATE_NAME_SENTINEL", source_path="PRIVATE_SOURCE_SENTINEL")
    events[-1]["result"]["account"] = "PRIVATE_ACCOUNT_SENTINEL"
    exported = export_session("\n".join(json.dumps(row) for row in events))
    assert {key: value for key, value in exported.items() if key != "description"} == {
        key: value for key, value in data.items() if key != "description"
    }
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    assert all(set(board) == {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
               and Board.from_dict(board).to_dict() == board for board in data["boards"])
    assert all(set(pet) == {"occupied", "species", "attack", "health", "level"}
               for board in data["boards"] for name in ("shop", "team") for pet in board[name])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"}
               and set(row["action"]) == {"kind", "slot", "target"} for row in data["actions"])
    assert all(board == Board(board.phase) for board in frames if board.phase in {Phase.NAMING, Phase.NAMING_READY})
    assert all(token not in json.dumps(exported) for token in ("SENTINEL", ".local", ".png", "frame_path", "elapsed"))


def test_missing_late_roll_ocr_receipt_cannot_fake_acknowledgment_or_completion():
    data, frames = recording()
    index, roll = next((index, row) for index, row in reversed(list(enumerate(data["actions"])))
                       if row["action"]["kind"] == "roll")
    assert (roll["poll"], roll["acknowledged_poll"]) == (772, 774)
    uncertain = replace(frames[roll["acknowledged_poll"] - 1], gold=None)
    # Keep all earlier native evidence. Only the final roll's gold receipt is
    # unreadable; a later terminal frame must remain unread after the timeout.
    result, clicks, events = replay(frames[:roll["poll"]] + [uncertain] * 50 + [Board(Phase.RESULT)])
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == ("action_timeout", 88, 87, 822)
    assert result.pending_action == Action("roll") and result.error is None
    assert result.last_board == uncertain and result.last_board.phase != Phase.RESULT
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"][:index + 1]]
    assert timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"][:index + 1]
    ]
    assert timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"][:index]
    ]
    assert [row["poll"] for row in events if row["event"] == "proposed"][-1] == 772
