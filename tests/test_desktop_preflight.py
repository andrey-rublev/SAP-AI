"""Launcher preflight with synthetic assets and mocks; no desktop/native IO."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

import sap_ai.desktop.cli as desktop
from sap_ai.desktop.runtime import DesktopUnavailable


@pytest.fixture
def preflight(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    template = tmp_path / "shop.png"
    Image.new("RGB", (4, 4), (200, 200, 200)).save(template)
    config = {"version": 1, "image_size": [80, 60], "calibrated": True,
              "phase_templates": [{"phase": "shop", "region": [0, 0, 4, 4],
                                   "template": "shop.png"}]}
    profile.write_text(json.dumps(config), encoding="utf-8")
    forbidden = {}
    for name in ("desktop_control_lease", "WindowsGameWindow", "connect_game_window",
                 "run_controller", "DesktopRuntime", "DesktopSession", "run_desktop_session",
                 "run_with_completed_restart", "run_arena_batch", "FrameRecorder",
                 "read_image", "save_image", "stopped_before_start"):
        forbidden[name] = Mock(side_effect=AssertionError(f"preflight must not call {name}"))
        monkeypatch.setattr(desktop, name, forbidden[name])
    assets = Mock(wraps=desktop.validate_profile_assets)
    dependencies = Mock(return_value=r"C:\OCR\tesseract.exe")
    monkeypatch.setattr(desktop, "validate_profile_assets", assets)
    monkeypatch.setattr(desktop, "validate_live_dependencies", dependencies)
    outputs = tmp_path / "outputs"
    stop = tmp_path / "STOP"
    args = ["play", "--check", "--profile", str(profile), "--stop-file", str(stop),
            "--log", str(outputs / "session.jsonl"), "--last-frame", str(outputs / "last.png"),
            "--record-dir", str(outputs / "frames")]
    environment = SimpleNamespace(profile=profile, template=template, config=config,
                                  forbidden=forbidden, assets=assets, dependencies=dependencies,
                                  outputs=outputs, stop=stop, args=args)
    yield environment
    for function in forbidden.values():
        function.assert_not_called()


@pytest.mark.parametrize("stopped", [False, True])
@pytest.mark.parametrize("existing_outputs", [False, True])
def test_check_reports_resources_and_stop_without_creating_or_changing_outputs(
        preflight, capsys, stopped, existing_outputs):
    if stopped:
        preflight.stop.write_bytes(b"keep this STOP\n")
    if existing_outputs:
        preflight.outputs.mkdir()
        (preflight.outputs / "session.jsonl").write_bytes(b"existing receipt\n")
        (preflight.outputs / "last.png").write_bytes(b"existing diagnostic\n")
    original = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (*preflight.outputs.glob("*"), preflight.stop) if path.is_file()}

    # play still has execute=True, and play.cmd injects restart-completed.
    desktop.main([*preflight.args, "--restart-completed", "--arenas", "3"])

    report = json.loads(capsys.readouterr().out)
    assert report == {"status": "checks_passed", "live_validation": False,
                      "profile": str(preflight.profile.resolve()), "expected_client_size": [80, 60],
                      "tesseract_path": r"C:\OCR\tesseract.exe", "species_hover": False,
                      "stop_file": str(preflight.stop.resolve()), "stop_present": stopped}
    preflight.assets.assert_called_once()
    preflight.dependencies.assert_called_once_with()
    assert preflight.outputs.exists() is existing_outputs
    assert not (preflight.outputs / "frames").exists()
    assert preflight.stop.exists() is stopped
    for path, (content, modified) in original.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == (content, modified)


@pytest.mark.parametrize("fault", ["missing", "malformed", "uncalibrated", "invalid_geometry"])
def test_invalid_profile_fails_before_assets_dependencies_or_perception(
        preflight, monkeypatch, fault):
    if fault == "missing":
        preflight.profile.unlink()
    elif fault == "malformed":
        preflight.profile.write_text("{broken", encoding="utf-8")
    else:
        config = dict(preflight.config)
        config["calibrated" if fault == "uncalibrated" else "image_size"] = (
            False if fault == "uncalibrated" else [0, 60])
        preflight.profile.write_text(json.dumps(config), encoding="utf-8")
    perception = Mock(side_effect=AssertionError("invalid profile must not construct perception"))
    monkeypatch.setattr(desktop, "Perceptor", perception)
    with pytest.raises((RuntimeError, ValueError)):
        desktop.main(preflight.args)
    perception.assert_not_called()
    preflight.assets.assert_not_called()
    preflight.dependencies.assert_not_called()
    assert not preflight.outputs.exists()


@pytest.mark.parametrize("fault", ["missing", "wrong_size"])
def test_asset_failure_prevents_dependency_check_and_all_controller_io(preflight, fault):
    if fault == "missing":
        preflight.template.unlink()
    else:
        Image.new("RGB", (3, 4), (200, 200, 200)).save(preflight.template)
    with pytest.raises((FileNotFoundError, ValueError)):
        desktop.main(preflight.args)
    preflight.assets.assert_called_once()
    preflight.dependencies.assert_not_called()
    assert not preflight.outputs.exists()


@pytest.mark.parametrize("message", ["missing OCR dependencies", "Tesseract version query timed out"])
def test_dependency_failure_is_reported_as_cli_failure_without_output_files(
        preflight, monkeypatch, capsys, message):
    preflight.dependencies.side_effect = DesktopUnavailable(message)
    monkeypatch.setattr("sys.argv", ["sap_ai/desktop/cli.py", *preflight.args])
    with pytest.raises(SystemExit, match=message):
        desktop.cli()
    preflight.assets.assert_called_once()
    preflight.dependencies.assert_called_once_with()
    assert capsys.readouterr().out == ""
    assert not preflight.outputs.exists()


def test_check_remains_play_only_and_preserves_normal_play_defaults():
    normal = desktop.parse_args(["play"])
    assert normal.check is False and normal.execute is True and normal.start_arena is True
    checked = desktop.parse_args(["play", "--check"])
    assert checked.check is True and checked.execute is True
    assert checked.profile == normal.profile == ".local/desktop/calibration-native.json"
    assert checked.restart_completed is normal.restart_completed is False
    with pytest.raises(SystemExit):
        desktop.parse_args(["run", "--profile", "unused.json", "--check"])


def test_cmd_forwarding_accepts_check_with_its_existing_restart_default():
    launcher = (Path(desktop.__file__).parents[2] / "play.cmd").read_text(encoding="utf-8")
    assert '".venv\\Scripts\\python.exe" -m sap_ai.desktop.cli play --restart-completed %*' in launcher
    args = desktop.parse_args(["play", "--restart-completed", "--check"])
    assert args.check is args.restart_completed is args.execute is True
