"""Replay recorded tier dismissal, turn-three/four shops and the next tier failure."""
import json
from dataclasses import replace
from pathlib import Path

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, Phase


def recording(name="desktop_tier_shop_progress.json"):
    path = Path(__file__).parent / "fixtures" / name
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(item) for item in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(frames, **bounds):
    observations, clicks, events = iter(frames), [], []
    result = DesktopSession(
        lambda: next(observations), clicks.append, DesktopPolicy(),
        phase_actions={Phase.TIER_UNLOCK: Action("dismiss_tier"),
                       Phase.END_TURN_CONFIRM: Action("confirm_end_turn"),
                       Phase.ROUND_RESULT: Action("continue_round")},
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
        max_polls=len(frames), **bounds,
    ).run()
    return result, clicks, events


def action_timeline(events, kind):
    return [(row["poll"], Action.from_dict(row["action"]))
            for row in events if row["event"] == kind]


def test_native_tier_dismissal_and_two_shop_turns_preserve_twenty_acknowledged_actions():
    data, frames = recording()
    result, clicks, events = replay(frames, max_actions=20)
    recorded = data["actions"][:20]
    assert clicks == [Action.from_dict(row["action"]) for row in recorded]
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_actions", 20, 20, 150,
    )
    assert result.last_board.phase == Phase.BATTLE
    assert result.pending_action is None and result.error is None
    assert action_timeline(events, "acted") == [
        (row["poll"], Action.from_dict(row["action"])) for row in recorded
    ]
    assert [poll for poll, _ in action_timeline(events, "acted")] == [
        2, 6, 9, 13, 16, 19, 24, 27, 30, 33, 90, 100, 103, 106, 111, 114, 117, 120, 124, 127,
    ]
    assert action_timeline(events, "acknowledged") == [
        (row["acknowledged_poll"], Action.from_dict(row["action"])) for row in recorded
    ]
    assert [poll for poll, _ in action_timeline(events, "acknowledged")] == [
        5, 8, 12, 15, 18, 23, 26, 29, 32, 56, 99, 102, 105, 110, 113, 116, 119, 123, 126, 149,
    ]
    assert clicks[0] == Action("dismiss_tier")
    assert (frames[4].turn, frames[4].gold) == (3, 10)
    assert (frames[98].turn, frames[98].gold) == (4, 10)
    # Both replacements retain native unknown identity while proving receipts
    # and a visible gain from four to nine combined attack/health.
    for turn, slot, sell_poll, sell_ack, buy_ack in ((3, 0, 19, 23, 26), (4, 1, 106, 110, 113)):
        before, sold, bought = frames[sell_poll - 1], frames[sell_ack - 1], frames[buy_ack - 1]
        assert before.turn == sold.turn == bought.turn == turn
        assert before.team[slot].strength == 4 and before.team[slot].level == 1
        assert sold.team[slot].occupied is False and sold.gold == before.gold + 1
        assert bought.team[slot].strength == 9 and bought.team[slot].species is None
        assert bought.gold == sold.gold - 3


def test_native_tier_shop_progress_retains_unknown_next_tier_and_pending_continue():
    data, frames = recording()
    assert data["expected"] == {
        "actions": 21, "acknowledgments": 20, "polls": 202,
        "reason": "unknown_timeout", "pending_action": Action("continue_round").to_dict(),
    }
    result, clicks, events = replay(frames)
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "unknown_timeout", 21, 20, 202,
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
    assert action_timeline(events, "acted")[-1] == (177, Action("continue_round"))
    assert len(frames) == 202 and all(board.phase == Phase.UNKNOWN for board in frames[-25:])


def test_synthetic_recording_splice_rejects_stale_shop_after_tier_dismissal():
    night, prior_frames = recording("desktop_night_shop.json")
    _, tier_frames = recording()
    # This is a synthetic contextual splice, not an uninterrupted native run.
    # Retain turn-two observations through Continue, add one repeated stable
    # tier-overlay poll so its action can dispatch, then freeze a corrupted
    # recorded shop on the previous turn. The unmodified tier fixture starts
    # without a readable turn or prior-shop context and cannot prove staleness.
    stale = replace(tier_frames[3], turn=2)
    frames = [*prior_frames[:71], *tier_frames[:2], tier_frames[1], *([stale] * 25)]
    result, clicks, events = replay(frames, action_max_polls=25)
    assert clicks == [Action.from_dict(row["action"]) for row in night["actions"]] + [
        Action("dismiss_tier")
    ]
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "action_timeout", 7, 6, 99,
    )
    assert result.pending_action == Action("dismiss_tier") and result.error is None
    assert result.last_board == stale
    assert action_timeline(events, "acted")[-1] == (74, Action("dismiss_tier"))
    assert action_timeline(events, "acknowledged")[-1] == (73, Action("continue_round"))
    assert [row["poll"] for row in events if row["event"] == "proposed"][-1] == 74
