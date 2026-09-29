"""CLI integration with injected observation and input; no desktop access."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import desktop
from desktop_session import SessionResult
from desktop_state import Action, Phase


@pytest.fixture
def cli_runtime(monkeypatch):
    profile = Mock()
    monkeypatch.setattr(desktop.VisionProfile, "load", Mock(return_value=profile))
    monkeypatch.setattr(desktop, "Perceptor", Mock())
    window = Mock()
    monkeypatch.setattr(desktop, "WindowsGameWindow", window)
    runtime = SimpleNamespace(observe=Mock(), act=Mock(), last_frame=None,
                              phase_actions=Mock(return_value={Phase.ROUND_RESULT: Action("continue_round")}))
    monkeypatch.setattr(desktop, "DesktopRuntime", Mock(return_value=runtime))
    return runtime, window


def arguments(tmp_path):
    return ["run", "--profile", "unused.json", "--log", str(tmp_path / "session.jsonl"),
            "--stop-file", str(tmp_path / "STOP"), "--last-frame", str(tmp_path / "last.png")]


def test_existing_stop_file_prevents_observation_and_input(tmp_path, cli_runtime):
    runtime, _ = cli_runtime
    (tmp_path / "STOP").touch()
    desktop.main(arguments(tmp_path) + ["--execute"])
    runtime.observe.assert_not_called()
    runtime.act.assert_not_called()
    assert '"reason": "stopped"' in (tmp_path / "session.jsonl").read_text()


def test_cli_passes_calibrated_phase_actions_and_checks_deadline(tmp_path, cli_runtime, monkeypatch):
    runtime, _ = cli_runtime
    clock = [10.0]
    monkeypatch.setattr(desktop.time, "monotonic", lambda: clock[0])

    def session(*args, **kwargs):
        assert kwargs["phase_actions"] == runtime.phase_actions.return_value
        assert kwargs["preview"] is True
        assert not kwargs["should_stop"]()
        clock[0] = 11.5
        assert kwargs["should_stop"]()
        return SimpleNamespace(run=lambda: SessionResult(0, 0, 0, "stopped"))

    monkeypatch.setattr(desktop, "DesktopSession", session)
    desktop.main(arguments(tmp_path) + ["--max-seconds", "1.5"])


def test_failed_observation_saves_last_frame_for_diagnosis(tmp_path, cli_runtime):
    from PIL import Image
    runtime, _ = cli_runtime
    runtime.last_frame = np.full((10, 20, 3), 127, dtype=np.uint8)
    runtime.observe.side_effect = ValueError("bad template")
    with pytest.raises(RuntimeError, match="observation_error"):
        desktop.main(arguments(tmp_path))
    with Image.open(tmp_path / "last.png") as saved:
        np.testing.assert_array_equal(np.asarray(saved), runtime.last_frame)
    runtime.act.assert_not_called()


def test_unacknowledged_input_is_reported_as_failure(tmp_path, cli_runtime, monkeypatch):
    result = SessionResult(1, 0, 2, "max_polls", pending_action=Action("roll"))
    monkeypatch.setattr(desktop, "DesktopSession", Mock(return_value=SimpleNamespace(run=lambda: result)))
    with pytest.raises(RuntimeError, match="max_polls"):
        desktop.main(arguments(tmp_path))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_wall_clock_bound_must_be_finite_positive(value):
    with pytest.raises(SystemExit):
        desktop.parse_args(["run", "--profile", "unused.json", "--max-seconds", value])
