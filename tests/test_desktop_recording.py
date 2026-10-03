"""Private frame recording contracts, using only synthetic frames and fake IO."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import desktop
from desktop_state import Board, PetSlot, Phase


@pytest.fixture
def frame():
    return np.zeros((8, 12, 3), dtype=np.uint8)


def event(gold=10, phase="shop", **fields):
    return {"event": "observed", "poll": 1, "board": {
        "phase": phase, "gold": gold, "team": [{"occupied": True, "attack": 2}], **fields,
    }}


@pytest.fixture
def writer(monkeypatch):
    write = Mock()
    monkeypatch.setattr(desktop, "save_image", write)
    return write


def test_recording_is_opt_in_and_limit_has_a_positive_default():
    args = desktop.parse_args(["run", "--profile", "private.json"])
    assert args.record_dir is None
    assert args.record_limit == 200


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_cli_rejects_invalid_record_limit(limit):
    with pytest.raises(SystemExit):
        desktop.parse_args(["run", "--profile", "private.json", "--record-limit", limit])


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_recorder_rejects_invalid_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        desktop.FrameRecorder(tmp_path, limit)


def test_only_changed_boards_record_not_duplicate_polls_or_pixels(tmp_path, frame, writer):
    recorder = desktop.FrameRecorder(tmp_path)
    first = event()
    recorded = recorder.record(first, frame)
    assert "frame_path" in recorded and "frame_path" not in first
    duplicate = {**event(), "poll": 2, "stable_frames": 2}
    assert "frame_path" not in recorder.record(duplicate, frame + 1)
    changed = recorder.record(event(gold=9), frame)
    assert "frame_path" in changed
    assert writer.call_count == 2
    assert writer.call_args_list[0].args[0] is frame
    assert Path(recorded["frame_path"]).name == "000001.png"
    assert Path(changed["frame_path"]).name == "000002.png"


def test_phase_and_nested_stat_changes_record_and_history_is_copied(tmp_path, frame, writer):
    recorder = desktop.FrameRecorder(tmp_path)
    observed = event()
    recorder.record(observed, frame)
    observed["board"]["team"][0]["attack"] = 3
    recorder.record(observed, frame)
    recorder.record(event(phase="battle"), frame)
    recorder.record(event(), frame)
    assert writer.call_count == 4


def test_dictionary_order_does_not_create_duplicate_records(tmp_path, frame, writer):
    recorder = desktop.FrameRecorder(tmp_path)
    a = event()
    b = {**a, "board": dict(reversed(list(a["board"].items())))}
    recorder.record(a, frame)
    recorder.record(b, frame)
    writer.assert_called_once()


def test_runs_have_unique_directories_even_with_same_frame(tmp_path, frame, writer):
    paths = []
    for _ in range(2):
        recorder = desktop.FrameRecorder(tmp_path)
        paths.append(Path(recorder.record(event(), frame)["frame_path"]))
    assert paths[0] != paths[1]
    assert paths[0].parent != paths[1].parent
    assert all(path.is_absolute() and path.parent.is_dir() for path in paths)
    assert all("Z-" in path.parent.name for path in paths)


def test_limit_stops_files_but_keeps_events_and_reports_once(tmp_path, frame, writer):
    recorder = desktop.FrameRecorder(tmp_path, limit=2)
    events = [recorder.record(event(gold=gold), frame) for gold in (10, 9, 8, 7, 6)]
    assert writer.call_count == recorder.saved == 2
    assert len(events) == 5
    assert [e["board"]["gold"] for e in events] == [10, 9, 8, 7, 6]
    assert sum("record_limit_reached" in e for e in events) == 1
    assert events[1]["record_limit_reached"] is True and events[1]["record_limit"] == 2
    assert all("frame_path" not in e for e in events[2:])


def test_non_observation_events_never_need_frame_or_create_directory(tmp_path, writer):
    directory = tmp_path / "unused"
    recorder = desktop.FrameRecorder(directory)
    for name in ("started", "action", "acknowledged", "stopped"):
        data = {"event": name}
        assert recorder.record(data, None) is data
    writer.assert_not_called()
    assert not directory.exists()


def test_missing_frame_fails_without_advancing_recorder(tmp_path, frame, writer):
    recorder = desktop.FrameRecorder(tmp_path)
    with pytest.raises(RuntimeError, match="captured frame"):
        recorder.record(event(), None)
    assert recorder.saved == 0
    assert "frame_path" in recorder.record(event(), frame)
    writer.assert_called_once()


def test_failed_write_propagates_and_does_not_report_success(tmp_path, frame, writer):
    writer.side_effect = OSError("disk full")
    recorder = desktop.FrameRecorder(tmp_path, limit=1)
    with pytest.raises(OSError, match="disk full"):
        recorder.record(event(), frame)
    assert recorder.saved == 0


def fake_runtime(monkeypatch, frame):
    board = Board(Phase.SHOP, gold=10, turn=1,
                  shop=(PetSlot(True, attack=2, health=3),), team=(PetSlot(False),) * 5)
    runtime = SimpleNamespace(observe=Mock(return_value=board), act=Mock(),
                              last_frame=frame, phase_actions=lambda *, start_arena=False: {})
    monkeypatch.setattr(desktop.VisionProfile, "load", lambda path: SimpleNamespace(validate=Mock()))
    monkeypatch.setattr(desktop, "Perceptor", Mock())
    monkeypatch.setattr(desktop, "WindowsGameWindow", Mock())
    monkeypatch.setattr(desktop, "DesktopRuntime", Mock(return_value=runtime))
    return runtime


def run_args(tmp_path):
    return ["run", "--profile", "fake.json", "--max-polls", "3", "--max-actions", "1",
            "--log", str(tmp_path / "session.jsonl"), "--last-frame", str(tmp_path / "last.png"),
            "--stop-file", str(tmp_path / "STOP"), "--record-dir", str(tmp_path / "frames")]


def test_cli_event_log_links_recorded_frames_and_skips_stable_poll(tmp_path, frame, writer, monkeypatch):
    runtime = fake_runtime(monkeypatch, frame)
    desktop.main(run_args(tmp_path))
    records = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()]
    observed = [entry for entry in records if entry["event"] == "observed"]
    assert len(observed) == 2
    assert "frame_path" in observed[0] and "frame_path" not in observed[1]
    # One observation recording and the existing final diagnostic screenshot.
    assert writer.call_count == 2
    runtime.act.assert_not_called()


def test_recording_io_failure_stops_session_before_desktop_action(tmp_path, frame, writer, monkeypatch):
    runtime = fake_runtime(monkeypatch, frame)
    def fail_recording(image, path):
        if Path(path).name != "last.png":
            raise OSError("recording disk full")
    writer.side_effect = fail_recording
    with pytest.raises(RuntimeError, match="event_error"):
        desktop.main(run_args(tmp_path) + ["--execute"])
    runtime.act.assert_not_called()


def test_cli_without_recording_does_not_construct_recorder(tmp_path, frame, writer, monkeypatch):
    fake_runtime(monkeypatch, frame)
    recorder = Mock(side_effect=AssertionError("recording must be opt-in"))
    monkeypatch.setattr(desktop, "FrameRecorder", recorder)
    desktop.main(run_args(tmp_path)[:-2])
    recorder.assert_not_called()
    writer.assert_called_once()
