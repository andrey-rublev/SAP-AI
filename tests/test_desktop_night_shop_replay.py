"""Replay native turn-two shop actions and the later tier recognition failure."""
import json
from dataclasses import replace
from pathlib import Path

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, Phase


def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_night_shop.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(frames, **bounds):
    observations, clicks, events = iter(frames), [], []
    result = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy(),
        phase_actions={Phase.END_TURN_CONFIRM: Action("confirm_end_turn"),
                       Phase.ROUND_RESULT: Action("continue_round")},
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
        max_polls=len(frames), **bounds,
    ).run()
    return result, clicks, events


def action_timeline(events, kind):
    return [(row["poll"], Action.from_dict(row["action"]))
            for row in events if row["event"] == kind]


def test_native_night_shop_preserves_five_acknowledged_actions_and_battle_handoff():
    data, frames = recording()
    result, clicks, events = replay(frames, max_actions=5)
    expected_actions = [Action.from_dict(row["action"]) for row in data["actions"][:5]]
    assert clicks == expected_actions == [Action("buy", 0, 3), Action("buy", 0, 4),
                                         Action("roll"), Action("end_turn"),
                                         Action("confirm_end_turn")]
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_actions", 5, 5, 43,
    )
    assert result.last_board.phase == Phase.BATTLE
    assert result.pending_action is None and result.error is None
    assert action_timeline(events, "acted") == [
        (row["poll"], action) for row, action in zip(data["actions"][:5], expected_actions)
    ] == list(zip([2, 8, 11, 15, 18], expected_actions))
    assert action_timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], action)
        for row, action in zip(data["actions"][:5], expected_actions)
    ] == list(zip([7, 10, 14, 17, 42], expected_actions))
    assert frames[0].turn == 2 and frames[0].gold == 10
    assert frames[14].turn == 2 and frames[14].gold == 3
    assert frames[6].team[3].species == "otter"


def test_native_night_shop_replay_retains_unknown_tier_failure_and_pending_continue():
    data, frames = recording()
    assert data["expected"] == {
        "actions": 6, "acknowledgments": 5, "polls": 96,
        "reason": "unknown_timeout", "pending_action": Action("continue_round").to_dict(),
    }
    result, clicks, events = replay(frames)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "unknown_timeout", 6, 5, 96,
    )
    assert result.pending_action == Action("continue_round") and result.error is None
    assert result.last_board == frames[-1] == Board(Phase.UNKNOWN)
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert action_timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert action_timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"]))
        for row in data["actions"] if row["acknowledged_poll"] is not None
    ]
    assert action_timeline(events, "acted")[-1] == (71, Action("continue_round"))
    assert all(board.phase == Phase.UNKNOWN for board in frames[-25:])


def test_native_purchase_with_contradictory_target_health_blocks_all_followup_input():
    _, frames = recording()
    assert frames[0].shop[0].health == 4 and frames[0].team[3].occupied is False
    corrupted = [
        replace(board, team=(*board.team[:3], replace(board.team[3], health=3), *board.team[4:]))
        if board.phase == Phase.SHOP and board.gold == 7 else board
        for board in frames
    ]
    # Gold, source compaction and destination occupancy still match the purchase.
    assert corrupted[6].gold == 7 and corrupted[6].shop == frames[6].shop
    assert corrupted[6].team[3].occupied is True and corrupted[6].team[3].health == 3
    result, clicks, events = replay(corrupted, action_max_polls=5)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "action_timeout", 1, 0, 7,
    )
    assert result.pending_action == Action("buy", 0, 3) and result.error is None
    assert clicks == [Action("buy", 0, 3)]
    assert [row["poll"] for row in events if row["event"] == "proposed"] == [2]
    assert not action_timeline(events, "acknowledged")


def test_native_night_shop_fixture_contains_only_typed_relative_evidence():
    data, frames = recording()
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    assert len(frames) == data["expected"]["polls"] == 96
    assert all(Board.from_dict(row).to_dict() == row for row in data["boards"])
    assert all(set(row) == {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
               for row in data["boards"])
    assert all(set(pet) == {"occupied", "species", "attack", "health", "level"}
               for row in data["boards"] for name in ("shop", "team") for pet in row[name])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"} for row in data["actions"])
    assert all(Action.from_dict(row["action"]).to_dict() == row["action"] for row in data["actions"])
    assert set(data["expected"]) == {
        "actions", "acknowledgments", "polls", "reason", "pending_action",
    }
