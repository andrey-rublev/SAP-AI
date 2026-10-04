"""Replay the second native fresh arena and its no-input deferral without IO."""
import json
from pathlib import Path

from desktop_session import DesktopSession, PHASE_ACTION_KINDS
from desktop_state import Action, Board, BoardChangedBeforeInput, DesktopPolicy, Phase


def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_fresh_deferral_terminal.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(data, frames, *, max_polls=None):
    observations, clicks, events = iter(frames), [], []
    deferrals = iter(data["deferrals"])
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
            # The native run predates optional preflight Board diagnostics.
            raise BoardChangedBeforeInput("recorded native preflight rejected without input")
        clicks.append(action)

    result = DesktopSession(
        lambda: next(observations), act, DesktopPolicy(), event_callback=emit,
        phase_actions={phase: Action(kind) for phase, kind in PHASE_ACTION_KINDS.items()},
        max_actions=300, max_polls=len(frames) if max_polls is None else max_polls,
        clock=lambda: 0., sleep=lambda _: None,
    ).run()
    assert next_deferral is None
    return result, clicks, events


def timeline(events, kind):
    return [(row["poll"], Action.from_dict(row["action"]))
            for row in events if row["event"] == kind]


def test_native_second_fresh_arena_preserves_all_actions_receipts_and_deferral_without_restart():
    data, frames = recording()
    menus = [Board(phase) for phase in (Phase.MAIN_MENU, Phase.MAIN_MENU,
                                      Phase.PLAY_MENU, Phase.PLAY_MENU,
                                      Phase.ARENA_SETUP, Phase.ARENA_SETUP)]
    result, clicks, events = replay(data, frames + menus)
    assert data["expected"] == {
        "actions": 90, "acknowledgments": 90, "polls": 818,
        "reason": "result", "pending_action": None,
    }
    assert {key: result.to_dict()[key] for key in data["expected"]} == data["expected"]
    assert result.error is None and result.last_board == frames[-1] == Board(Phase.RESULT)
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert timeline(events, "acted")[:3] == [
        (2, Action("open_play")), (5, Action("open_arena")), (8, Action("start_arena")),
    ]
    assert timeline(events, "acknowledged")[:3] == [
        (4, Action("open_play")), (7, Action("open_arena")), (15, Action("start_arena")),
    ]
    assert [(poll, item) for poll, item in timeline(events, "acted")
            if item.kind in {"choose_name", "confirm_name"}] == [
        (31, Action("choose_name")), (34, Action("confirm_name")),
    ]
    assert [(poll, item) for poll, item in timeline(events, "acknowledged")
            if item.kind in {"choose_name", "confirm_name"}] == [
        (33, Action("choose_name")), (57, Action("confirm_name")),
    ]
    assert frames[0] == Board(Phase.MAIN_MENU)
    first_shop = next(board for board in frames if board.phase == Phase.SHOP)
    last_shop = next(board for board in reversed(frames) if board.phase == Phase.SHOP)
    assert (first_shop.turn, first_shop.gold, first_shop.wins, first_shop.lives) == (1, 10, 0, 5)
    assert all(pet.occupied is False for pet in first_shop.team)
    assert (last_shop.turn, last_shop.wins, last_shop.lives) == (9, 2, 1)
    assert {board.turn for board in frames if board.phase == Phase.SHOP} == set(range(1, 10))
    assert len([row for row in events if row["event"] == "observed"]) == len(frames) == 818

    action = Action("buy", 0, 4)
    assert data["deferrals"] == [{"poll": 87, "action": action.to_dict()}]
    assert timeline(events, "deferred") == [(87, action)]
    assert not any("preflight_board" in row for row in events if row["event"] == "deferred")
    observed = {row["poll"]: row for row in events if row["event"] == "observed"}
    assert (frames[86].team[3].attack, frames[86].team[3].health) == (3, 2)
    assert frames[87].team[3].occupied is None
    assert (frames[88].team[3].attack, frames[88].team[3].health,
            frames[88].team[3].level) == (4, 2, None)
    assert frames[89] == frames[90] and frames[90].team[3].level == 1
    assert [observed[poll]["stable_frames"] for poll in (88, 89, 90, 91)] == [1, 1, 1, 2]
    assert [poll for poll, _ in timeline(events, "proposed") if 87 <= poll <= 91] == [87, 91]
    assert [(poll, item) for poll, item in timeline(events, "acted") if 87 <= poll <= 91] == [(91, action)]
    assert (93, action) in timeline(events, "acknowledged")


def test_native_deferral_poll_bound_stops_without_followup_input_or_pending_action():
    data, frames = recording()
    result, clicks, events = replay(data, frames, max_polls=88)
    recorded = [row for row in data["actions"] if row["poll"] < 87]
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_polls", 12, 12, 88,
    )
    assert result.pending_action is None and result.proposed_action is None and result.error is None
    assert result.last_board == frames[87] and result.last_board.team[3].occupied is None
    assert clicks == [Action.from_dict(row["action"]) for row in recorded]
    assert timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in recorded
    ]
    assert timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in recorded
    ]
    assert timeline(events, "deferred") == [(87, Action("buy", 0, 4))]
    assert timeline(events, "proposed")[-1] == (87, Action("buy", 0, 4))
    assert not any(row["event"] in {"acted", "acknowledged"} and row["poll"] >= 87
                   for row in events)


def test_native_second_fresh_fixture_contains_only_typed_evidence_and_legacy_deferral():
    data, frames = recording()
    assert data["description"].startswith("Second uninterrupted native MAIN_MENU-to-RESULT desktop arena recording, including one proven no-input deferral;")
    assert set(data) == {"description", "boards", "observation_runs", "actions", "deferrals", "expected"}
    assert len(frames) == data["expected"]["polls"] == 818
    assert all(Board.from_dict(row).to_dict() == row for row in data["boards"])
    assert all(set(row) == {"poll", "action", "acknowledged_poll"} for row in data["actions"])
    assert all(set(row) == {"poll", "action"} for row in data["deferrals"])
    assert all(Action.from_dict(row["action"]).to_dict() == row["action"]
               for row in (*data["actions"], *data["deferrals"]))
    assert set(data["expected"]) == {"actions", "acknowledgments", "polls", "reason", "pending_action"}
    assert all(board == Board(board.phase) for board in frames
               if board.phase in {Phase.NAMING, Phase.NAMING_READY})
