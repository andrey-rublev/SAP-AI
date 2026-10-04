"""Replay an uninterrupted native fresh-start arena through terminal, without IO."""
import json
from pathlib import Path

from desktop_session import DesktopSession, PHASE_ACTION_KINDS
from desktop_state import Action, Board, DesktopPolicy, Phase


def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_fresh_arena_terminal.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(frames, **bounds):
    observations, clicks, events = iter(frames), [], []
    result = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy(),
        phase_actions={phase: Action(kind) for phase, kind in PHASE_ACTION_KINDS.items()},
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
        max_actions=300, max_polls=len(frames), **bounds,
    ).run()
    return result, clicks, events


def action_timeline(events, kind):
    return [(row["poll"], Action.from_dict(row["action"]))
            for row in events if row["event"] == kind]


def test_native_fresh_arena_preserves_all_policy_actions_and_receipts_without_restart():
    data, frames = recording()
    menus = [Board(phase) for phase in (Phase.MAIN_MENU, Phase.MAIN_MENU,
                                      Phase.PLAY_MENU, Phase.PLAY_MENU,
                                      Phase.ARENA_SETUP, Phase.ARENA_SETUP)]
    result, clicks, events = replay(frames + menus)
    assert data["expected"] == {
        "actions": 94, "acknowledgments": 94, "polls": 874,
        "reason": "result", "pending_action": None,
    }
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "result", 94, 94, 874,
    )
    assert result.pending_action is None and result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert action_timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert action_timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert action_timeline(events, "acted")[:3] == [
        (2, Action("open_play")), (5, Action("open_arena")), (8, Action("start_arena")),
    ]
    assert action_timeline(events, "acknowledged")[:3] == [
        (4, Action("open_play")), (7, Action("open_arena")), (13, Action("start_arena")),
    ]
    assert [row for row in action_timeline(events, "acted") if row[1].kind in {"choose_name", "confirm_name"}] == [
        (30, Action("choose_name")), (33, Action("confirm_name")),
    ]
    assert [row for row in action_timeline(events, "acknowledged") if row[1].kind in {"choose_name", "confirm_name"}] == [
        (32, Action("choose_name")), (56, Action("confirm_name")),
    ]
    assert frames[0] == Board(Phase.MAIN_MENU)
    assert {board.turn for board in frames if board.phase == Phase.SHOP} == set(range(1, 10))
    first_shop = next(board for board in frames if board.phase == Phase.SHOP)
    last_shop = next(board for board in reversed(frames) if board.phase == Phase.SHOP)
    assert (first_shop.turn, first_shop.gold, first_shop.wins, first_shop.lives) == (1, 10, 0, 5)
    assert all(pet.occupied is False for pet in first_shop.team)
    assert (last_shop.turn, last_shop.wins, last_shop.lives) == (9, 2, 1)
    assert result.last_board == frames[-1] == Board(Phase.RESULT)
    assert len([row for row in events if row["event"] == "observed"]) == len(frames) == 874
    assert not any(row["event"] == "deferred" for row in events)
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    assert data["description"].startswith("Uninterrupted native MAIN_MENU-to-RESULT desktop arena recording;")
    assert all(Board.from_dict(row).to_dict() == row for row in data["boards"])
    assert all(Action.from_dict(row["action"]).to_dict() == row["action"] for row in data["actions"])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"} for row in data["actions"])
    # Choosing a name exposes only phase evidence in this fixture.
    assert all(board == Board(board.phase) for board in frames
               if board.phase in {Phase.NAMING, Phase.NAMING_READY})


def test_corrupted_native_name_selection_receipt_blocks_confirmation_and_retries():
    data, frames = recording()
    selected = next(row for row in data["actions"] if row["action"]["kind"] == "choose_name")
    assert (selected["poll"], selected["acknowledged_poll"]) == (30, 32)
    # Leave the naming dialog stably unselected instead of acknowledging the
    # observed selected-name phase. Earlier startup and purchase evidence stays.
    corrupted = [Board(Phase.NAMING) if board.phase == Phase.NAMING_READY else board
                 for board in frames]
    result, clicks, events = replay(corrupted, action_max_polls=6)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "action_timeout", 9, 8, 36,
    )
    assert result.pending_action == Action("choose_name") and result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"][:9]]
    assert action_timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"][:9]
    ]
    assert action_timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"][:8]
    ]
    assert clicks.count(Action("choose_name")) == 1 and Action("confirm_name") not in clicks
    assert [row["poll"] for row in events if row["event"] == "proposed"][-1] == 30
