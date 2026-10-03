"""Offline replay export preserves typed evidence and rejects broken timelines."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from desktop_session import DesktopSession
from desktop_state import Action, Board, PetSlot, Phase
from tools.export_desktop_replay import export_session, main


def recording(*, pending=False):
    source = PetSlot(True, "fish", 2, 3, 1)
    empty = PetSlot(False)
    before = Board(Phase.SHOP, gold=10, turn=1, shop=(source,), team=(empty,) * 5)
    after = replace(before, gold=7, shop=(empty,), team=(source, empty, empty, empty, empty))
    action = Action("buy", 0, 0)
    events = []

    class Policy:
        def choose_action(self, board):
            return action

    frames = iter([before, before, after, after, after] if not pending else [before] * 5)
    runner = DesktopSession(lambda: next(frames), lambda _: None, Policy(),
                            event_callback=events.append, clock=lambda: 0., sleep=lambda _: None,
                            max_actions=1, max_polls=5, action_max_polls=3)
    runner.run()
    return events


def encode(events):
    return "\n".join(json.dumps(event) for event in events)


def test_export_matches_actual_recorded_opening_fixture():
    # Reconstruct event structure around existing native evidence, without any IO.
    path = Path(__file__).parent / "fixtures" / "desktop_opening_purchases.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(board) for board in expected["boards"]]
    frames = iter(boards[index] for index, count in expected["observation_runs"] for _ in range(count))
    events = []
    actions = iter(Action.from_dict(row["action"]) for row in expected["actions"])

    class Policy:
        def choose_action(self, board):
            return next(actions)

    runner = DesktopSession(lambda: next(frames), lambda _: None, Policy(),
                            event_callback=events.append, clock=lambda: 0., sleep=lambda _: None,
                            max_actions=3, max_polls=11)
    runner.run()
    actual = export_session(encode(events))
    assert {key: value for key, value in actual.items() if key != "description"} == {
        key: value for key, value in expected.items() if key != "description"
    }


@pytest.mark.parametrize("pending", [False, True])
def test_export_preserves_completed_and_unacknowledged_action_evidence(pending):
    result = export_session(encode(recording(pending=pending)))
    assert result["observation_runs"] == ([[0, 5]] if pending else [[0, 2], [1, 3]])
    assert result["expected"]["acknowledgments"] == (0 if pending else 1)
    assert result["actions"] == [{"poll": 2, "action": Action("buy", 0, 0).to_dict(),
                                  "acknowledged_poll": None if pending else 4}]
    assert result["expected"]["pending_action"] == (Action("buy", 0, 0).to_dict() if pending else None)


def test_export_drops_paths_images_timestamps_and_extra_account_fields():
    events = recording()
    for row in events:
        row.update(frame_path="private-capture.png", elapsed=123.4, account="private-user")
    events[-1]["result"]["account"] = "private-user"
    exported = json.dumps(export_session(encode(events)))
    assert all(text not in exported for text in ("private-capture", "private-user", "elapsed", "frame_path"))


def test_multiple_runs_select_newest_or_explicit_one_based_session():
    healthy, pending = recording(), recording(pending=True)
    logs = encode(healthy + pending)
    assert export_session(logs)["expected"]["reason"] == "action_timeout"
    assert export_session(logs, session=1)["expected"]["reason"] == "max_actions"
    assert export_session(logs, session=-2) == export_session(logs, session=1)
    with pytest.raises(ValueError, match="incomplete"):
        export_session(encode(healthy + pending[:-1]))
    assert export_session(encode(healthy + pending[:-1]), session=1)["expected"]["reason"] == "max_actions"


@pytest.mark.parametrize("session", [0, True, 1.5, 3, -3])
def test_invalid_or_missing_run_selection_is_rejected(session):
    with pytest.raises(ValueError):
        export_session(encode(recording()), session=session)


@pytest.mark.parametrize("corruption", [
    "missing_start", "missing_finish", "skipped_poll", "boolean_poll", "orphan_ack",
    "duplicate_ack", "wrong_action", "different_ack_board", "stale_proposal",
    "same_poll_ack", "wrong_count", "wrong_pending", "wrong_last_board", "io_error",
    "unsupported_reason", "board_metadata", "action_metadata", "result_not_object",
])
def test_corrupted_or_unreproducible_evidence_is_rejected(corruption):
    events = deepcopy(recording())
    observations = [row for row in events if row["event"] == "observed"]
    acted = next(row for row in events if row["event"] == "acted")
    acknowledged = next(row for row in events if row["event"] == "acknowledged")
    proposed = next(row for row in events if row["event"] == "proposed")
    result = events[-1]["result"]
    if corruption == "missing_start":
        events.pop(0)
    elif corruption == "missing_finish":
        events.pop()
    elif corruption == "skipped_poll":
        observations[0]["poll"] = 2
    elif corruption == "boolean_poll":
        observations[0]["poll"] = True
    elif corruption == "orphan_ack":
        events.remove(acted)
    elif corruption == "duplicate_ack":
        events.insert(events.index(acknowledged) + 1, deepcopy(acknowledged))
    elif corruption == "wrong_action":
        acted["action"] = Action("roll").to_dict()
    elif corruption == "different_ack_board":
        acknowledged["board"]["gold"] = 99
    elif corruption == "stale_proposal":
        events.remove(acted)
        acted["poll"] = 3
        events.insert(events.index(observations[2]) + 1, acted)
    elif corruption == "same_poll_ack":
        events.remove(acknowledged)
        acknowledged["poll"], acknowledged["board"] = 2, proposed["board"]
        events.insert(events.index(acted) + 1, acknowledged)
    elif corruption == "wrong_count":
        result["actions"] = 2
    elif corruption == "wrong_pending":
        result["pending_action"] = Action("buy", 0, 0).to_dict()
    elif corruption == "wrong_last_board":
        result["last_board"]["gold"] = 99
    elif corruption == "io_error":
        result["error"] = "injected input failure"
    elif corruption == "unsupported_reason":
        result["reason"] = "private-account-text"
    elif corruption == "board_metadata":
        observations[0]["board"]["account"] = "private-user"
    elif corruption == "result_not_object":
        events[-1]["result"] = []
    else:
        acted["action"]["account"] = "private-user"
    with pytest.raises((ValueError, TypeError, KeyError)):
        export_session(encode(events))


def test_new_dispatch_cannot_share_the_prior_acknowledgment_poll():
    events = recording()
    ack = next(row for row in events if row["event"] == "acknowledged")
    roll = Action("roll").to_dict()
    events[events.index(ack) + 1:events.index(ack) + 1] = [
        {"event": "proposed", "poll": ack["poll"], "board": deepcopy(ack["board"]), "action": roll},
        {"event": "acted", "poll": ack["poll"], "action": roll},
    ]
    final_observation = next(row for row in events if row["event"] == "observed" and row["poll"] == 5)
    final_observation["board"]["gold"] = 6
    events.insert(-1, {"event": "acknowledged", "poll": 5,
                       "board": deepcopy(final_observation["board"]), "action": roll})
    events[-1]["result"].update(actions=2, acknowledgments=2,
                                last_board=deepcopy(final_observation["board"]))
    with pytest.raises(ValueError, match="proposal contradicts"):
        export_session(encode(events))


def test_cli_exports_a_new_fixture_without_overwriting_logs_or_outputs(tmp_path, capsys):
    source, output = tmp_path / "recording.jsonl", tmp_path / "replay.json"
    source.write_text(encode(recording()), encoding="utf-8")
    original = source.read_bytes()
    main(["--log", str(source), "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == export_session(source.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == {"actions": 1, "polls": 5}
    saved = output.read_bytes()
    for target in (source, output):
        with pytest.raises(SystemExit):
            main(["--log", str(source), "--output", str(target)])
    assert source.read_bytes() == original and output.read_bytes() == saved


def test_invalid_log_creates_no_output_directory(tmp_path):
    source, output = tmp_path / "recording.jsonl", tmp_path / "new" / "replay.json"
    source.write_text(encode(recording()[:-1]), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--log", str(source), "--output", str(output)])
    assert not output.parent.exists()
