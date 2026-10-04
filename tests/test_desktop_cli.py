"""CLI integration with injected observation and input; no desktop access."""
from types import SimpleNamespace
from unittest.mock import Mock
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import desktop
from desktop_session import SessionResult
from desktop_state import Action, Board, Phase
from desktop_vision import NumberTemplate, Perceptor, PhaseTemplate, Rect, SlotConfig, VisionProfile


@pytest.fixture
def cli_runtime(monkeypatch):
    profile = Mock()
    monkeypatch.setattr(desktop.VisionProfile, "load", Mock(return_value=profile))
    monkeypatch.setattr(desktop, "Perceptor", Mock())
    monkeypatch.setattr(desktop, "validate_profile_assets", Mock())
    monkeypatch.setattr(desktop, "validate_live_dependencies", Mock())
    window = Mock()
    monkeypatch.setattr(desktop, "WindowsGameWindow", window)
    runtime = SimpleNamespace(observe=Mock(), act=Mock(), last_frame=None,
                              phase_actions=Mock(return_value={Phase.ROUND_RESULT: Action("continue_round")}))
    monkeypatch.setattr(desktop, "DesktopRuntime", Mock(return_value=runtime))
    return runtime, window


def arguments(tmp_path):
    return ["run", "--profile", "unused.json", "--log", str(tmp_path / "session.jsonl"),
            "--stop-file", str(tmp_path / "STOP"), "--last-frame", str(tmp_path / "last.png")]


def play_arguments(tmp_path):
    return ["play", "--log", str(tmp_path / "session.jsonl"),
            "--stop-file", str(tmp_path / "STOP"), "--last-frame", str(tmp_path / "last.png")]


def test_existing_stop_file_prevents_observation_and_input(tmp_path, cli_runtime):
    runtime, _ = cli_runtime
    (tmp_path / "STOP").touch()
    desktop.main(arguments(tmp_path) + ["--execute"])
    runtime.observe.assert_not_called()
    runtime.act.assert_not_called()
    assert '"reason": "stopped"' in (tmp_path / "session.jsonl").read_text()


@pytest.mark.parametrize("start_arena", [False, True])
def test_cli_passes_calibrated_phase_actions_and_checks_deadline(tmp_path, cli_runtime, monkeypatch, start_arena):
    runtime, _ = cli_runtime
    clock = [10.0]
    monkeypatch.setattr(desktop.time, "monotonic", lambda: clock[0])

    def session(*args, **kwargs):
        assert kwargs["phase_actions"] == runtime.phase_actions.return_value
        assert kwargs["preview"] is True
        assert desktop.DesktopRuntime.call_args.kwargs["should_stop"] is kwargs["should_stop"]
        assert not kwargs["should_stop"]()
        clock[0] = 11.5
        assert kwargs["should_stop"]()
        return SimpleNamespace(run=lambda: SessionResult(0, 0, 0, "stopped"))

    monkeypatch.setattr(desktop, "DesktopSession", session)
    flags = ["--start-arena"] if start_arena else []
    desktop.main(arguments(tmp_path) + ["--max-seconds", "1.5"] + flags)
    runtime.phase_actions.assert_called_once_with(start_arena=start_arena)


def test_start_arena_flag_stays_in_preview_until_execute_is_enabled(tmp_path, cli_runtime):
    runtime, _ = cli_runtime
    runtime.observe.return_value = Board(Phase.MAIN_MENU)
    runtime.phase_actions.return_value = {Phase.MAIN_MENU: Action("open_play")}
    desktop.main(arguments(tmp_path) + ["--start-arena"])
    assert desktop.DesktopRuntime.call_args.kwargs["execute"] is False
    runtime.phase_actions.assert_called_once_with(start_arena=True)
    assert runtime.observe.call_count == 2
    runtime.act.assert_not_called()
    log = (tmp_path / "session.jsonl").read_text()
    assert '"reason": "preview"' in log
    assert '"kind": "open_play"' in log


def test_start_arena_is_disabled_by_default():
    assert desktop.parse_args(["run", "--profile", "unused.json"]).start_arena is False
    args = desktop.parse_args(["run", "--profile", "unused.json", "--start-arena"])
    assert args.start_arena is True
    assert args.execute is False


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


def test_play_defaults_enable_execution_and_arena_with_finite_bounds():
    args = desktop.parse_args(["play"])
    assert args.profile == ".local/desktop/calibration-native.json"
    assert args.window == "Super Auto Pets"
    assert args.execute is True
    assert args.start_arena is True
    assert (args.max_actions, args.max_polls, args.max_seconds, args.action_timeout) == (600, 12000, 3600, 30)
    assert args.stop_file == ".local/desktop/STOP"
    assert args.log == ".local/desktop/session.jsonl"
    run = desktop.parse_args(["run", "--profile", "unused.json"])
    assert (run.execute, run.start_arena, run.max_actions, run.max_polls, run.max_seconds) == (False, False, 40, 600, 1800)


def test_play_explicit_profile_and_bounds_override_defaults():
    args = desktop.parse_args(["play", "--profile", "other.json", "--window", "other game",
                               "--max-actions", "1", "--max-polls", "20", "--max-seconds", "15.5"])
    assert (args.profile, args.window, args.max_actions, args.max_polls, args.max_seconds) == (
        "other.json", "other game", 1, 20, 15.5)


def test_stopped_play_never_constructs_window_or_loads_profile(tmp_path, cli_runtime):
    runtime, window = cli_runtime
    (tmp_path / "STOP").touch()
    desktop.main(play_arguments(tmp_path))
    desktop.VisionProfile.load.assert_not_called()
    desktop.validate_profile_assets.assert_not_called()
    desktop.validate_live_dependencies.assert_not_called()
    window.assert_not_called()
    window.return_value.activate.assert_not_called()
    runtime.observe.assert_not_called()
    runtime.act.assert_not_called()
    events = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "finished"]
    assert events[-1]["result"] == SessionResult(0, 0, 0, "stopped").to_dict()


def test_play_preflights_then_activates_once_before_session_observation(tmp_path, cli_runtime, monkeypatch):
    runtime, window = cli_runtime
    order = []
    profile = desktop.VisionProfile.load.return_value
    profile.image_size = (2560, 1440)
    profile.validate.side_effect = lambda **kwargs: order.append("profile")
    desktop.validate_profile_assets.side_effect = lambda perceptor: order.append("assets")
    desktop.validate_live_dependencies.side_effect = lambda: order.append("dependencies")
    window.side_effect = lambda title: order.append("window") or window.return_value

    def activate(size):
        assert size == profile.image_size
        assert callable(window.return_value.should_stop)
        assert not window.return_value.should_stop()
        order.append("activate")

    window.return_value.activate.side_effect = activate
    runtime.observe.side_effect = lambda: order.append("observe") or Board(Phase.RESULT)
    desktop.main(play_arguments(tmp_path))
    assert order == ["profile", "assets", "dependencies", "window", "activate", "observe"]
    window.return_value.activate.assert_called_once_with(profile.image_size)
    assert desktop.DesktopRuntime.call_args.kwargs["execute"] is True
    runtime.phase_actions.assert_called_once_with(start_arena=True)
    runtime.act.assert_not_called()
    events = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()]
    assert events[0]["preview"] is False
    assert events[-1]["result"]["reason"] == "result"


def test_play_passes_bounded_run_configuration(tmp_path, cli_runtime, monkeypatch):
    runtime, _ = cli_runtime
    session = Mock(return_value=SimpleNamespace(run=lambda: SessionResult(0, 0, 0, "max_actions")))
    monkeypatch.setattr(desktop, "DesktopSession", session)
    desktop.main(play_arguments(tmp_path) + ["--max-actions", "2", "--max-polls", "20", "--action-timeout", "7"])
    kwargs = session.call_args.kwargs
    assert kwargs["preview"] is False
    assert (kwargs["max_actions"], kwargs["max_polls"], kwargs["action_timeout"]) == (2, 20, 7)
    assert kwargs["should_stop"] is desktop.DesktopRuntime.call_args.kwargs["should_stop"]
    assert kwargs["phase_actions"] == runtime.phase_actions.return_value


def test_play_rejects_uncalibrated_profile_before_perception_or_window(tmp_path, cli_runtime):
    _, window = cli_runtime
    desktop.VisionProfile.load.return_value.validate.side_effect = ValueError("profile must be explicitly calibrated")
    with pytest.raises(ValueError, match="explicitly calibrated"):
        desktop.main(play_arguments(tmp_path))
    desktop.Perceptor.assert_not_called()
    desktop.validate_live_dependencies.assert_not_called()
    window.assert_not_called()


def test_play_missing_profile_gives_actionable_message(tmp_path, cli_runtime):
    _, window = cli_runtime
    desktop.VisionProfile.load.side_effect = FileNotFoundError("missing")
    with pytest.raises(RuntimeError, match="calibrated desktop profile at .local/desktop/calibration-native.json; see docs/desktop.md"):
        desktop.main(play_arguments(tmp_path))
    window.assert_not_called()


@pytest.mark.parametrize("failure", ["assets", "dependencies"])
def test_play_preflight_failure_prevents_window_construction(tmp_path, cli_runtime, failure):
    runtime, window = cli_runtime
    check = desktop.validate_profile_assets if failure == "assets" else desktop.validate_live_dependencies
    check.side_effect = RuntimeError("preflight failed")
    with pytest.raises(RuntimeError, match="preflight failed"):
        desktop.main(play_arguments(tmp_path))
    window.assert_not_called()
    runtime.observe.assert_not_called()
    runtime.act.assert_not_called()
    if failure == "assets":
        desktop.validate_live_dependencies.assert_not_called()


def test_stop_requested_during_preflight_prevents_window_construction(tmp_path, cli_runtime):
    runtime, window = cli_runtime
    desktop.validate_live_dependencies.side_effect = lambda: (tmp_path / "STOP").touch()
    desktop.main(play_arguments(tmp_path))
    window.assert_not_called()
    runtime.observe.assert_not_called()
    runtime.act.assert_not_called()
    assert '"reason": "stopped"' in (tmp_path / "session.jsonl").read_text()


def test_stop_requested_after_window_construction_prevents_activation(tmp_path, cli_runtime):
    runtime, window = cli_runtime
    desktop.DesktopRuntime.side_effect = lambda *args, **kwargs: (tmp_path / "STOP").touch() or runtime
    desktop.main(play_arguments(tmp_path))
    window.assert_called_once()
    window.return_value.activate.assert_not_called()
    runtime.observe.assert_not_called()
    runtime.act.assert_not_called()


def test_play_activation_failure_does_not_observe_or_act(tmp_path, cli_runtime):
    runtime, window = cli_runtime
    window.return_value.activate.side_effect = RuntimeError("Windows refused focus")
    with pytest.raises(RuntimeError, match="Windows refused focus"):
        desktop.main(play_arguments(tmp_path))
    runtime.observe.assert_not_called()
    runtime.act.assert_not_called()


@pytest.mark.parametrize("execute", [False, True])
def test_run_never_activates_or_invokes_play_dependency_preflight(tmp_path, cli_runtime, execute):
    runtime, window = cli_runtime
    runtime.observe.return_value = Board(Phase.RESULT)
    desktop.main(arguments(tmp_path) + (["--execute"] if execute else []))
    window.return_value.activate.assert_not_called()
    desktop.validate_profile_assets.assert_not_called()
    desktop.validate_live_dependencies.assert_not_called()


def test_play_reports_unconfigured_transition_as_failure(tmp_path, cli_runtime):
    runtime, _ = cli_runtime
    runtime.observe.return_value = Board(Phase.MAIN_MENU)
    runtime.phase_actions.return_value = {}
    with pytest.raises(RuntimeError, match="transition_disabled"):
        desktop.main(play_arguments(tmp_path))
    runtime.act.assert_not_called()


@pytest.mark.parametrize("reference", ["phase.png", "empty.png", "ant-alt.png", "number.png"])
@pytest.mark.parametrize("fault", ["missing", "wrong_size"])
def test_profile_asset_preflight_checks_all_reference_kinds(tmp_path, reference, fault):
    region = Rect(1, 1, 2, 2)
    profile = VisionProfile(
        image_size=(6, 6), base_dir=tmp_path, calibrated=True,
        hud={"gold": region},
        phase_templates=(PhaseTemplate(Phase.SHOP, region, "phase.png"),),
        shop=(SlotConfig(region, empty_template="empty.png",
                         species_templates={"ant": ["ant.png", "ant-alt.png"]}),),
        numeric_templates=(NumberTemplate("gold", 0, "number.png"),),
    ).validate(require_calibrated=True)
    for name in ("phase.png", "empty.png", "ant.png", "ant-alt.png", "number.png"):
        desktop.save_image(np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / name)
    if fault == "missing":
        (tmp_path / reference).unlink()
        error = FileNotFoundError
    else:
        desktop.save_image(np.zeros((3, 2, 3), dtype=np.uint8), tmp_path / reference)
        error = ValueError
    with pytest.raises(error):
        desktop.validate_profile_assets(Perceptor(profile, ocr=None))


@pytest.mark.skipif(sys.platform != "win32", reason="cmd wrapper runs on Windows")
def test_cmd_launcher_missing_venv_is_clear_without_running_python(tmp_path):
    directory = tmp_path / "project with spaces"
    directory.mkdir()
    launcher = directory / "play.cmd"
    shutil.copyfile(Path(desktop.__file__).with_name("play.cmd"), launcher)
    result = subprocess.run(["cmd.exe", "/d", "/c", str(launcher)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "project Python environment is missing" in result.stderr
    assert "python -m venv .venv" in result.stderr
    assert '-m pip install -e ".[live]"' in result.stderr
    assert "Tesseract OCR must also be installed" in result.stderr
