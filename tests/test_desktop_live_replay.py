"""Controller regressions from recorded Boards, without OCR, screenshots, or IO.

The scripted policy preserves recorded choices; these tests verify when the
controller sends and acknowledges them, not whether those choices play well.
Wall-clock capture latency is deliberately excluded from this poll replay.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from desktop_session import DesktopSession, PHASE_ACTION_KINDS, action_acknowledged
from desktop_state import Action, Board, Phase


@pytest.fixture
def recording():
    path = Path(__file__).parent / "fixtures" / "desktop_turns_6_to_8.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(board) for board in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"]
              for _ in range(count)]
    return data, frames


def replay(data, frames, **options):
    shop_actions = iter(Action.from_dict(row["action"]) for row in data["actions"]
                        if row["action"]["kind"] not in PHASE_ACTION_KINDS.values())

    class RecordedPolicy:
        def choose_action(self, board):
            return next(shop_actions, None)

    observations, clicks, events = iter(frames), [], []
    runner = DesktopSession(
        lambda: next(observations), clicks.append, RecordedPolicy(),
        event_callback=events.append, clock=lambda: 0., sleep=lambda _: None,
        max_actions=40, max_polls=len(frames),
        phase_actions={phase: Action(kind) for phase, kind in PHASE_ACTION_KINDS.items()},
        **options,
    )
    return runner.run(), clicks, events


def test_recorded_three_turn_run_preserves_every_action_and_ack_poll(recording):
    data, frames = recording
    result, clicks, events = replay(data, frames)
    expected = data["expected"]
    assert (result.actions, result.acknowledgments, result.polls, result.reason) == (
        32, 31, 327, "unknown_timeout",
    )
    assert {field: getattr(result, field) for field in
            ("actions", "acknowledgments", "polls", "reason")} == {
        field: expected[field] for field in ("actions", "acknowledgments", "polls", "reason")
    }
    assert result.pending_action == Action.from_dict(expected["pending_action"])
    assert result.error is None
    assert clicks == [Action.from_dict(row["action"]) for row in data["actions"]]
    assert [(event["poll"], Action.from_dict(event["action"])) for event in events
            if event["event"] == "acted"] == [
        (row["poll"], Action.from_dict(row["action"])) for row in data["actions"]
    ]
    acknowledged = [event for event in events if event["event"] == "acknowledged"]
    assert [(event["poll"], Action.from_dict(event["action"])) for event in acknowledged] == [
        (row["acknowledged_poll"], Action.from_dict(row["action"]))
        for row in data["actions"] if row["acknowledged_poll"] is not None
    ]
    assert all(Board.from_dict(event["board"]) == frames[event["poll"] - 1]
               for event in acknowledged)
    # The final Continue gets one click, then 25 unknown frames and no retry.
    assert all(frame.phase == Phase.UNKNOWN for frame in frames[302:])
    assert result.last_board == frames[-1]
    # No fixture field can silently smuggle screenshot paths into the replay.
    assert all(set(board) <= {"phase", "gold", "turn", "wins", "lives", "shop", "team"}
               for board in data["boards"])


def test_recorded_purchase_removes_middle_offer_and_compacts_duplicate_survivors(recording):
    _, frames = recording
    before, after = frames[15], frames[17]
    action = Action("buy", 1, 3)
    assert (before.gold, after.gold) == (11, 8)
    assert before.team[3].occupied is False
    assert (after.team[3].attack, after.team[3].health) == (3, 6)
    assert before.shop[2] == before.shop[3]  # Two observed 3/2 offers.
    assert after.shop[:3] == (before.shop[0], before.shop[2], before.shop[3])
    assert after.shop[3].occupied is False
    assert all(pet.species is None for pet in before.shop)  # No invented identity.
    assert action_acknowledged(before, after, action)


@pytest.mark.parametrize("missing_evidence", ["price", "target", "removal", "survivor_stats"])
def test_recorded_purchase_corruption_times_out_without_another_click(recording, missing_evidence):
    data, frames = recording
    before, after = frames[15], frames[17]
    if missing_evidence == "price":
        after = replace(after, gold=before.gold)
    elif missing_evidence == "target":
        after = replace(after, team=before.team)
    elif missing_evidence == "removal":
        # Drop the first offer instead of the purchased second one.
        after = replace(after, shop=(*before.shop[1:], after.shop[3]))
    else:
        after = replace(after, shop=(replace(after.shop[0], attack=None), *after.shop[1:]))
    assert not action_acknowledged(before, after, Action("buy", 1, 3))
    # The unchanged prefix retains the observed sale and its delayed OCR recovery.
    result, clicks, _ = replay(data, frames[:16] + [after] * 10, action_max_polls=10)
    assert clicks == [Action("continue_round"), Action("sell", 3), Action("buy", 1, 3)]
    assert (result.reason, result.actions, result.acknowledgments) == ("action_timeout", 3, 2)
    assert result.pending_action == Action("buy", 1, 3)


def test_recorded_end_turn_confirmation_waits_through_old_shop_and_unknowns(recording):
    data, frames = recording
    before, stale_shop, unknown, battle = frames[37], frames[38], frames[39], frames[60]
    action = Action("confirm_end_turn")
    assert before.phase == Phase.END_TURN_CONFIRM
    assert stale_shop.turn == 6 and stale_shop.phase == Phase.SHOP
    assert unknown.phase == Phase.UNKNOWN
    assert battle.phase == Phase.BATTLE
    assert not action_acknowledged(before, stale_shop, action)
    assert not action_acknowledged(before, unknown, action)
    assert action_acknowledged(before, battle, action)
    result, clicks, events = replay(data, frames[:60])
    assert result.pending_action == action
    assert (result.actions, result.acknowledgments) == (10, 9)
    assert clicks[-1] == action and clicks.count(action) == 1
    assert not any(event["event"] == "acted" and event["poll"] > 38 for event in events)
