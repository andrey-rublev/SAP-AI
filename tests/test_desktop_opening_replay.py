"""Replay native opening purchases with the real policy and no desktop IO."""
import json
from pathlib import Path

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy


def load_recording():
    path = Path(__file__).parent / "fixtures" / "desktop_opening_purchases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(board) for board in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def test_native_opening_purchases_preserve_policy_actions_and_acknowledgment_polls():
    data, frames = load_recording()
    observations, clicks, events = iter(frames), [], []
    session = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy(),
        max_actions=3, max_polls=len(frames), event_callback=events.append,
        clock=lambda: 0., sleep=lambda _: None,
    )
    result = session.run()
    assert {key: getattr(result, key) for key in data["expected"]} == data["expected"]
    assert result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    for event, field in (("acted", "poll"), ("acknowledged", "acknowledged_poll")):
        assert [(row["poll"], Action.from_dict(row["action"]))
                for row in events if row["event"] == event] == [
            (row[field], Action.from_dict(row["action"])) for row in data["actions"]
        ]
    assert [row["poll"] for row in events if row["event"] == "acted"] == [2, 5, 8]
    assert [row["poll"] for row in events if row["event"] == "acknowledged"] == [4, 7, 10]
    assert result.last_board == frames[-1]
    assert frames[0].gold == 10 and frames[-1].gold == 1
    # A known shop identity becomes unknown on the team; no identity is invented.
    assert frames[0].shop[0].species == "ant"
    assert all(pet.species is None for pet in frames[-1].team)
    assert [pet.occupied for pet in frames[-1].team] == [True, True, True, False, False]
    assert all(pet.occupied is False for pet in frames[-1].shop)


def test_opening_replay_contains_only_typed_board_and_action_evidence():
    data, _ = load_recording()
    assert set(data) == {"description", "boards", "observation_runs", "actions", "expected"}
    assert all(set(board) == {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
               for board in data["boards"])
    assert all(set(pet) == {"occupied", "species", "attack", "health", "level"}
               for board in data["boards"] for name in ("shop", "team")
               for pet in board[name])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"}
               and set(row["action"]) == {"kind", "slot", "target"}
               for row in data["actions"])
    assert set(data["expected"]) == {
        "actions", "acknowledgments", "polls", "reason", "pending_action",
    }
