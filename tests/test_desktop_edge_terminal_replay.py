"""Replay native turn-five tier dismissal through terminal, with the real policy."""
import json
from pathlib import Path

from desktop_session import DesktopSession, PHASE_ACTION_KINDS
from desktop_state import Action, Board, DesktopPolicy, Phase


def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_edge_to_terminal.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def trailing_menus():
    return [Board(phase) for phase in (Phase.MAIN_MENU, Phase.MAIN_MENU,
                                     Phase.PLAY_MENU, Phase.PLAY_MENU,
                                     Phase.ARENA_SETUP, Phase.ARENA_SETUP)]


def replay(frames):
    observations, clicks, events = iter(frames), [], []
    result = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy(),
        phase_actions={phase: Action(kind) for phase, kind in PHASE_ACTION_KINDS.items()},
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
        max_polls=len(frames),
    ).run()
    return result, clicks, events


def action_timeline(events, kind):
    return [(row["poll"], Action.from_dict(row["action"]))
            for row in events if row["event"] == kind]


def test_native_edge_run_preserves_policy_timelines_and_stops_before_menu_restart():
    data, frames = recording()
    assert data["expected"] == {
        "actions": 42, "acknowledgments": 42, "polls": 377,
        "reason": "result", "pending_action": None,
    }
    result, clicks, events = replay(frames + trailing_menus())
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "result", 42, 42, 377,
    )
    assert result.pending_action is None and result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert action_timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert action_timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    observed = [row for row in events if row["event"] == "observed"]
    assert len(observed) == 377 and observed[-1]["stable_frames"] == 1
    assert result.last_board == frames[-1] == Board(Phase.RESULT)
    # This native recording resumes at tier three/turn five. It contains no
    # fresh-start menu or first-shop evidence, and terminal counters stay unknown.
    assert frames[0] == Board(Phase.TIER_UNLOCK)
    assert {board.turn for board in frames if board.phase == Phase.SHOP} == {5, 6, 7, 8}
    assert (frames[4].turn, frames[4].gold, frames[4].wins, frames[4].lives) == (5, 10, 2, 4)
    last_shop = next(board for board in reversed(frames) if board.phase == Phase.SHOP)
    assert (last_shop.turn, last_shop.wins, last_shop.lives) == (8, 2, 1)
    assert [(row["poll"], row["acknowledged_poll"]) for row in data["actions"]
            if row["action"]["kind"] == "dismiss_tier"] == [(2, 5), (194, 197)]
    assert action_timeline(events, "acted")[-1] == (323, Action("confirm_end_turn"))
    assert action_timeline(events, "acknowledged")[-1] == (345, Action("confirm_end_turn"))


def test_synthetic_early_terminal_keeps_pending_confirmation_and_never_restarts():
    data, frames = recording()
    final = data["actions"][-1]
    assert (final["poll"], final["acknowledged_poll"]) == (323, 345)
    action = Action.from_dict(final["action"])
    # Move the recorded terminal observation ahead of the final receipt. This
    # counterfactual checks immediate stopping; it is not a native-run claim.
    result, clicks, events = replay(frames[:final["poll"]] + [frames[-1]] + trailing_menus())
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "result", 42, 41, 324,
    )
    assert result.pending_action == action == Action("confirm_end_turn")
    assert result.last_board == Board(Phase.RESULT) and result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert action_timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    assert action_timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in data["actions"][:-1]
    ]
    assert [row["poll"] for row in events if row["event"] == "proposed"][-1] == 323
