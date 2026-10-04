"""Typed evidence from a private recorded-image study; CI never opens images."""
import json
from pathlib import Path

from desktop_session import DesktopSession
from desktop_state import Action, Board, BoardChangedBeforeInput, DesktopPolicy


def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_preinput_image_splice.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(data, frames, *, max_polls=None):
    observations, clicks, events = iter(frames), [], []
    deferrals = iter(data.get("deferrals", []))
    next_deferral = next(deferrals, None)
    current_poll = 0

    def emit(row):
        nonlocal current_poll
        current_poll = row["poll"]
        events.append(row)

    def act(action):
        nonlocal next_deferral
        if next_deferral is not None and current_poll == next_deferral["poll"]:
            assert action == Action.from_dict(next_deferral["action"])
            next_deferral = next(deferrals, None)
            raise BoardChangedBeforeInput("recorded offline preflight rejected without input")
        clicks.append(action)

    result = DesktopSession(
        lambda: next(observations), act, DesktopPolicy(), event_callback=emit,
        max_actions=1, max_polls=len(frames) if max_polls is None else max_polls,
        clock=lambda: 0., sleep=lambda _: None,
    ).run()
    assert next_deferral is None
    return result, clicks, events


def test_recorded_image_splice_replans_only_after_fresh_stability_and_preserves_receipt():
    data, frames = recording()
    assert data["expected"] == {
        "actions": 1, "acknowledgments": 1, "polls": 7,
        "reason": "max_actions", "pending_action": None,
    }
    result, clicks, events = replay(data, frames)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_actions", 1, 1, 7,
    )
    action = Action("buy", 0, 4)
    assert result.pending_action is None and result.error is None
    assert clicks == [action]
    assert data["deferrals"] == [{"poll": 2, "action": action.to_dict()}]
    assert [(row["poll"], row["action"]) for row in events if row["event"] == "deferred"] == [
        (2, action.to_dict())
    ]
    assert [row["poll"] for row in events if row["event"] == "proposed"] == [2, 4]
    assert [(row["poll"], row["action"]) for row in events if row["event"] == "acted"] == [
        (row["poll"], row["action"]) for row in data["actions"]
    ] == [(4, action.to_dict())]
    assert [(row["poll"], row["action"]) for row in events if row["event"] == "acknowledged"] == [
        (6, action.to_dict())
    ]
    assert [row["stable_frames"] for row in events if row["event"] == "observed"] == [1, 2, 1, 2, 1, 2, 3]
    assert (frames[0].team[3].attack, frames[0].team[3].health) == (3, 2)
    assert (frames[2].team[3].attack, frames[2].team[3].health) == (5, 2)
    assert frames[0].gold == frames[2].gold == 7 and frames[-1].gold == 4
    assert result.last_board == frames[-1] and frames[-1].team[4].strength == 6


def test_recorded_image_splice_poll_bound_stops_before_fresh_stable_proposal():
    data, frames = recording()
    result, clicks, events = replay(data, frames, max_polls=3)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_polls", 0, 0, 3,
    )
    assert result.pending_action is None and result.proposed_action is None and result.error is None
    assert result.last_board == frames[2] and result.last_board.team[3].attack == 5
    assert not clicks
    assert [row["poll"] for row in events if row["event"] == "proposed"] == [2]
    assert [row["poll"] for row in events if row["event"] == "deferred"] == [2]
    assert not any(row["event"] in {"acted", "acknowledged"} for row in events)


def test_recorded_image_splice_fixture_contains_only_typed_evidence_and_deferrals():
    data, frames = recording()
    assert data["description"] == "Synthetic splice of stored native images re-evaluated offline; mouse IO mocked."
    assert set(data) == {"description", "boards", "observation_runs", "actions", "deferrals", "expected"}
    assert len(frames) == data["expected"]["polls"] == 7
    assert all(Board.from_dict(row).to_dict() == row for row in data["boards"])
    assert all(set(row) == {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
               for row in data["boards"])
    assert all(set(pet) == {"occupied", "species", "attack", "health", "level"}
               and pet["species"] in {None, "pig"}
               for row in data["boards"] for name in ("shop", "team") for pet in row[name])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"} for row in data["actions"])
    assert all(set(row) == {"poll", "action"} for row in data["deferrals"])
    assert all(set(row["action"]) == {"kind", "slot", "target"}
               and Action.from_dict(row["action"]).to_dict() == row["action"]
               for row in (*data["actions"], *data["deferrals"]))
    assert set(data["expected"]) == {"actions", "acknowledgments", "polls", "reason", "pending_action"}
