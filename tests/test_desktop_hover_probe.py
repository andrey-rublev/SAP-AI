"""Explicit development hover diagnostics with injected clocks, camera and mouse."""
from dataclasses import asdict, replace
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from sap_ai.desktop.runtime import DesktopUnavailable, GameWindowNotFound
from sap_ai.desktop.control import DesktopControlConflict
from sap_ai.desktop.state import Board, PetSlot, Phase
from sap_ai.desktop.tooltip_layout import TooltipLayout
from sap_ai.desktop.vision import Rect, SlotConfig, VisionProfile
from tools import probe_desktop_hover as probe


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "desktop_control_lease", Mock(side_effect=nullcontext))
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "PRIVATE_ROOT", tmp_path / ".local")
    profile = VisionProfile(image_size=(100, 100), calibrated=True, pointer_rest=(50, 95),
        shop=(SlotConfig(Rect(10, 60, 10, 10)), SlotConfig(Rect(30, 60, 10, 10))),
        team=tuple(SlotConfig(Rect(10 + index * 15, 20, 10, 10)) for index in range(5)))
    board = Board(Phase.SHOP, gold=10, turn=10,
                  shop=(PetSlot(True, None, 2, 3, 1),) * 2,
                  team=(PetSlot(True, None, 2, 3, 1),) * 5)
    order, state = [], {"tooltip": True, "stopped": False}
    output = tmp_path / ".local" / "probe"
    layout = TooltipLayout((0, 0, 100, 100), (28, 32), (38, 42), (3, 3, 20, 10),
                           dark_perimeter_max_channel=60)
    layout_path = tmp_path / ".local" / "tooltip-layout.json"
    layout_path.parent.mkdir()
    layout_path.write_text(json.dumps({"version": 1, "calibrated": True, "image_size": [100, 100],
                                       "layout": asdict(layout)}))
    window = Mock()
    window.stop_requested.side_effect = lambda: state["stopped"]
    def hover(point, size):
        order.append(("hover", point))
        state["tooltip"] = point != profile.pointer_rest
    def capture():
        # Classification exists before every new development capture.
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["data_usage"] == "development_diagnostic" and manifest["held_out_evaluation"] is False
        order.append(("capture", state["tooltip"]))
        frame = np.full((100, 100, 3), 90, dtype=np.uint8)
        if state["tooltip"]:
            frame[34:76, 54:86] = 0
            frame[35:75, 55:85] = (247, 247, 245)
            frame[40:45, 60:74] = (190, 100, 15)
        return frame
    def observe(frame):
        assert not state["tooltip"], "tooltip-covered frames must not enter Board perception"
        return board
    window.hover.side_effect, window.capture.side_effect = hover, capture
    perceptor = SimpleNamespace(observe=Mock(side_effect=observe))
    load = Mock(side_effect=lambda path: order.append(("profile", path)) or profile)
    monkeypatch.setattr(probe.VisionProfile, "load", load)
    monkeypatch.setattr(probe, "Perceptor", Mock(return_value=perceptor))
    assets = Mock(side_effect=lambda value: order.append(("assets", value)))
    dependencies = Mock(side_effect=lambda: order.append(("dependencies", None)))
    factory = Mock(side_effect=lambda title: order.append(("window", title)) or window)
    monkeypatch.setattr(probe, "validate_profile_assets", assets)
    monkeypatch.setattr(probe, "validate_live_dependencies", dependencies)
    monkeypatch.setattr(probe, "WindowsGameWindow", factory)
    return SimpleNamespace(root=tmp_path, output=output, profile=profile, board=board, window=window,
                           perceptor=perceptor, load=load, assets=assets, dependencies=dependencies,
                           factory=factory, order=order, state=state, clock=Clock(), layout=layout,
                           layout_path=layout_path)


def arguments(*flags):
    return probe.parse_args(["--target", "shop:0", "--output-dir", ".local/probe",
                             "--tooltip-layout", ".local/tooltip-layout.json", *flags])


def run(environment, *flags):
    return probe.run_probe(arguments(*flags), clock=environment.clock, sleep=environment.clock.sleep)


def test_competing_controller_rejects_probe_before_window_or_pointer(environment, monkeypatch):
    monkeypatch.setattr(probe, "desktop_control_lease", Mock(side_effect=DesktopControlConflict("another controller")))
    report = run(environment)
    assert report["reason"] == "error" and "another controller" in report["error"]
    assert report["captures"] == report["pointer_moves"] == 0
    environment.factory.assert_not_called()
    environment.window.hover.assert_not_called()
    environment.window.capture.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_probe_lease_covers_window_and_all_moves_then_releases(environment, monkeypatch, fail):
    held, exited = [False], []
    @contextmanager
    def claim():
        held[0] = True
        try:
            yield
        finally:
            held[0] = False
            exited.append(True)
    monkeypatch.setattr(probe, "desktop_control_lease", claim)
    construct, hover = environment.factory.side_effect, environment.window.hover.side_effect
    def guarded_construct(*args):
        assert held[0]
        return construct(*args)
    def guarded_hover(*args):
        assert held[0]
        if fail:
            raise DesktopUnavailable("movement failure")
        return hover(*args)
    environment.factory.side_effect = guarded_construct
    environment.window.hover.side_effect = guarded_hover
    report = run(environment)
    assert report["reason"] == ("error" if fail else "completed")
    assert exited == [True] and not held[0]


def test_probe_uses_only_calibrated_centers_and_parks_between_targets(environment):
    report = run(environment, "--target", "team:1")
    assert (report["reason"], report["captures"], report["pointer_moves"]) == ("completed", 12, 6)
    assert report["pointer_restored"] is True and report["data_usage"] == "development_diagnostic"
    assert environment.window.hover.call_args_list == [
        call((50, 95), (100, 100)), call((15, 65), (100, 100)),
        call((50, 95), (100, 100)), call((50, 95), (100, 100)),
        call((30, 25), (100, 100)), call((50, 95), (100, 100)),
    ]
    assert [kind for kind, value in environment.order[:4]] == ["profile", "assets", "dependencies", "window"]
    assert [value for kind, value in environment.order if kind == "capture"] == [False, False, True, True, False, False] * 2
    assert json.loads((environment.output / "report.json").read_text()) == report
    assert len(list(environment.output.glob("*.png"))) == 16
    for target in report["targets"]:
        assert target["before"]["phase"] == target["after"]["phase"] == "shop"
        assert target["before"]["team"][0]["attack"] == 2
        assert target["after"]["team"][0]["attack"] == 2
        assert target["appearance_sequence_verified"] is True
        assert [sample["tooltip_status"] for sample in target["samples"]] == ["absent", "absent", "unique", "unique", "absent", "absent"]
        assert all("board" not in sample for sample in target["samples"] if sample["stage"] == "hover")
        # Diagnostic captures do not manufacture tooltip identities.
        assert target["before"]["shop"][0]["species"] is None
        assert target["after"]["shop"][0]["species"] is None
        assert Path(target["before_frame"]).parent == Path(target["after_frame"]).parent == environment.output
    assert environment.perceptor.observe.call_count == 8
    manifest = json.loads((environment.output / "manifest.json").read_text())
    assert manifest["max_captures"] == 12 and manifest["max_pointer_moves"] == 6
    assert manifest["tooltip_layout"]["sha256"] == hashlib.sha256(environment.layout_path.read_bytes()).hexdigest()
    assert "not proof of target identity" in report["association_limit"]
    assert report["identity_interpretation"] == "none"
    for name in ("activate", "click", "drag", "close_for_restart"):
        getattr(environment.window, name).assert_not_called()
    assert environment.window.mouse.mock_calls == []


@pytest.mark.parametrize("phase", [Phase.UNKNOWN, Phase.BATTLE, Phase.RESULT])
def test_non_shop_frames_never_receive_a_target_hover(environment, phase):
    environment.perceptor.observe.side_effect = None
    environment.perceptor.observe.return_value = Board(phase)
    report = run(environment)
    assert report["reason"] == "error" and "freshly observed SHOP" in report["error"]
    environment.window.hover.assert_called_once_with((50, 95), (100, 100))
    assert report["targets"][0]["hover_attempted"] is False


def test_changing_shop_is_not_retried_or_hovered(environment):
    environment.perceptor.observe.side_effect = [environment.board, replace(environment.board, gold=9)]
    report = run(environment)
    assert report["reason"] == "error" and "shop changed" in report["error"]
    assert report["captures"] == 2
    environment.window.hover.assert_called_once()


@pytest.mark.parametrize("occupied", [False, None])
def test_empty_or_unknown_occupancy_cannot_receive_a_target_hover(environment, occupied):
    environment.perceptor.observe.side_effect = None
    environment.perceptor.observe.return_value = replace(environment.board, shop=(PetSlot(occupied),))
    report = run(environment)
    assert report["reason"] == "error" and "known occupied" in report["error"]
    environment.window.hover.assert_called_once_with((50, 95), (100, 100))


def test_deadline_during_stability_wait_prevents_target_hover_and_further_capture(environment):
    report = run(environment, "--max-seconds", ".2")
    assert report["reason"] == "stopped" and report["elapsed"] == pytest.approx(.2)
    assert report["captures"] == 1
    environment.window.hover.assert_called_once_with((50, 95), (100, 100))


def test_held_escape_prevents_all_movement_and_capture(environment):
    environment.state["stopped"] = True
    report = run(environment)
    assert report["reason"] == "stopped" and report["captures"] == report["pointer_moves"] == 0
    environment.window.hover.assert_not_called()
    environment.window.capture.assert_not_called()


def test_stop_during_target_travel_never_captures_or_moves_again(environment):
    original = environment.window.hover.side_effect
    def hover(point, size):
        original(point, size)
        if point != environment.profile.pointer_rest:
            environment.state["stopped"] = True
    environment.window.hover.side_effect = hover
    report = run(environment)
    assert report["reason"] == "stopped" and report["captures"] == 2
    assert report["targets"][0]["hover_attempted"] is True
    assert report["targets"][0]["after"] is None and report["pointer_restored"] is False
    assert environment.window.hover.call_count == 2


def test_focus_failure_after_target_hover_is_retained_without_cleanup_or_retry(environment):
    capture = environment.window.capture.side_effect
    def focused_capture():
        if environment.state["tooltip"]:
            raise DesktopUnavailable("game lost focus")
        return capture()
    environment.window.capture.side_effect = focused_capture
    report = run(environment)
    assert report["reason"] == "error" and "lost focus" in report["error"]
    assert environment.window.hover.call_count == 2 and report["captures"] == 2


def test_missing_window_never_launches_or_activates_anything(environment):
    environment.factory.side_effect = GameWindowNotFound("open existing game first")
    report = run(environment)
    assert report["reason"] == "error" and "open existing game" in report["error"]
    assert report["captures"] == report["pointer_moves"] == 0
    environment.window.activate.assert_not_called()


def test_existing_stop_file_skips_preflight_window_and_output_creation(environment):
    stop = environment.root / ".local" / "desktop" / "STOP"
    stop.parent.mkdir(parents=True)
    stop.touch()
    report = run(environment)
    assert report["reason"] == "stopped" and report["directory"] is None
    environment.load.assert_not_called()
    environment.factory.assert_not_called()
    assert not environment.output.exists()


@pytest.mark.parametrize("stage", ["assets", "dependencies"])
def test_preflight_failures_prevent_all_desktop_access(environment, stage):
    getattr(environment, stage).side_effect = ValueError("preflight failed")
    with pytest.raises(ValueError, match="preflight failed"):
        run(environment)
    if stage == "assets":
        environment.dependencies.assert_not_called()
    environment.factory.assert_not_called()
    assert not environment.output.exists()


def test_stop_during_dependency_preflight_skips_window_and_output_creation(environment):
    def dependencies():
        stop = environment.root / ".local" / "desktop" / "STOP"
        stop.parent.mkdir(parents=True)
        stop.touch()
    environment.dependencies.side_effect = dependencies
    report = run(environment)
    assert report["reason"] == "stopped" and report["directory"] is None
    environment.factory.assert_not_called()
    assert not environment.output.exists()


def test_probe_requires_explicit_pointer_rest_and_calibrated_slot(environment):
    environment.load.side_effect = None
    environment.load.return_value = replace(environment.profile, pointer_rest=None)
    with pytest.raises(ValueError, match="pointer_rest"):
        run(environment)
    environment.factory.assert_not_called()


def test_uncalibrated_profile_prevents_all_desktop_access(environment):
    environment.load.side_effect = None
    environment.load.return_value = replace(environment.profile, calibrated=False)
    with pytest.raises(ValueError, match="explicitly calibrated"):
        run(environment)
    environment.factory.assert_not_called()


def test_missing_profile_target_prevents_all_desktop_access(environment):
    with pytest.raises(ValueError, match="no calibrated shop:4"):
        run(environment, "--target", "shop:4")
    environment.factory.assert_not_called()


def test_ocr_wrapper_observes_stop_before_any_target_movement(environment, monkeypatch):
    def perceptor(profile, *, ocr):
        def observe(frame):
            ocr(frame[:2, :2])
            return environment.board
        return SimpleNamespace(observe=observe)
    monkeypatch.setattr(probe, "Perceptor", perceptor)
    def ocr(image):
        environment.state["stopped"] = True
        return "2"
    monkeypatch.setattr(probe, "tesseract_ocr", ocr)
    report = run(environment)
    assert report["reason"] == "stopped" and report["captures"] == 1
    environment.window.hover.assert_called_once_with((50, 95), (100, 100))


def test_probe_reserves_a_full_ocr_timeout_in_the_remaining_global_budget(environment, monkeypatch):
    def perceptor(profile, *, ocr):
        def observe(frame):
            ocr(frame[:2, :2])
            return environment.board
        return SimpleNamespace(observe=observe)
    monkeypatch.setattr(probe, "Perceptor", perceptor)
    engine = Mock(return_value="2")
    monkeypatch.setattr(probe, "tesseract_ocr", engine)
    report = run(environment, "--max-seconds", "2")
    assert report["reason"] == "stopped" and report["captures"] == 1
    engine.assert_not_called()
    environment.window.hover.assert_called_once_with((50, 95), (100, 100))


@pytest.mark.parametrize("path", [".local/../outside", "outside", ".local"])
def test_output_escapes_are_rejected_before_preflight_or_capture(environment, path):
    with pytest.raises(ValueError, match="private|traversal"):
        run(environment, "--output-dir", path)
    environment.load.assert_not_called()
    environment.factory.assert_not_called()


def test_existing_output_directory_cannot_be_reused(environment):
    environment.output.mkdir(parents=True)
    sentinel = environment.output / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="must be new"):
        run(environment)
    assert sentinel.read_text() == "keep"
    environment.load.assert_not_called()


@pytest.mark.parametrize("linked_mode", ["symlink", "reparse"])
def test_linked_or_junction_output_paths_are_rejected(environment, monkeypatch, linked_mode):
    linked = environment.root / ".local" / "linked"
    linked.mkdir(parents=True)
    original = Path.lstat
    def lstat(path):
        if path == linked:
            return SimpleNamespace(st_mode=stat.S_IFLNK if linked_mode == "symlink" else stat.S_IFDIR,
                                   st_file_attributes=0x400 if linked_mode == "reparse" else 0)
        return original(path)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="linked or redirected"):
        run(environment, "--output-dir", ".local/linked/probe")
    environment.factory.assert_not_called()


def test_new_development_captures_do_not_reclassify_historic_data(environment):
    baseline = environment.root / ".local" / "historic-held-out.json"
    baseline.parent.mkdir(exist_ok=True)
    baseline.write_text('{"usage":"held_out","frames":["baseline.png"]}')
    original = baseline.read_bytes()
    assert run(environment)["reason"] == "completed"
    assert baseline.read_bytes() == original
    manifest = json.loads((environment.output / "manifest.json").read_text())
    assert manifest["identity_interpretation"] == "none" and "baseline" not in json.dumps(manifest)


@pytest.mark.parametrize("change", ["uncalibrated", "size", "search", "extra", "missing"])
def test_private_layout_requires_explicit_calibration_and_matching_geometry_before_native_io(environment, change):
    data = json.loads(environment.layout_path.read_text())
    if change == "uncalibrated":
        data["calibrated"] = False
    elif change == "size":
        data["image_size"] = [101, 100]
    elif change == "search":
        data["layout"]["search_rect"] = [0, 0, 101, 100]
    elif change == "extra":
        data["layout"]["guessed_coordinates"] = True
    else:
        del data["layout"]
    environment.layout_path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        run(environment)
    environment.factory.assert_not_called()
    environment.assets.assert_not_called()
    assert not environment.output.exists()


def test_layout_source_must_also_stay_private(environment):
    with pytest.raises(ValueError, match="private .local"):
        run(environment, "--tooltip-layout", "public-layout.json")
    environment.factory.assert_not_called()


def test_explicit_caller_layout_needs_no_mutable_path_fallback(environment):
    environment.layout_path.unlink()
    result = probe.run_probe(arguments(), clock=environment.clock, sleep=environment.clock.sleep,
                             tooltip_layout=environment.layout)
    assert result["reason"] == "completed"
    manifest = json.loads((environment.output / "manifest.json").read_text())
    assert manifest["tooltip_layout"]["source"] == "explicit_caller"
    assert "path" not in manifest["tooltip_layout"]


def test_stale_panel_after_parking_is_saved_as_rejected_evidence_without_reading_stats(environment):
    environment.window.hover.side_effect = lambda point, size: None
    result = run(environment)
    assert result["reason"] == "error" and "absence not established" in result["error"]
    environment.perceptor.observe.assert_not_called()
    assert result["captures"] == result["pointer_moves"] == 1
    target = result["targets"][0]
    assert target["appearance_sequence_verified"] is False and target["hover_attempted"] is False
    assert target["samples"][0]["tooltip_status"] == "unique"
    assert Path(target["samples"][0]["frame"]).is_file()


def test_changed_clear_board_rejects_target_and_prevents_the_next_target(environment):
    environment.perceptor.observe.side_effect = [environment.board, environment.board,
                                                replace(environment.board, gold=9), replace(environment.board, gold=9)]
    result = run(environment, "--target", "team:1")
    assert result["reason"] == "error" and "underlying shop changed" in result["error"]
    assert len(result["targets"]) == 1 and result["targets"][0]["appearance_sequence_verified"] is False
    assert result["captures"] == 6 and result["pointer_moves"] == 3


def test_cli_refuses_to_guess_a_tooltip_layout():
    with pytest.raises(SystemExit):
        probe.parse_args(["--target", "shop:0"])


@pytest.mark.parametrize("flags", [
    ["--max-seconds", "31"], ["--max-seconds", "nan"], ["--settle-seconds", "3"],
    ["--target", "food:0"], ["--target", "shop:5"], ["--target", "team:-1"],
    ["--target", "shop:0"],
    ["--target", "shop:1", "--target", "team:0", "--target", "team:1", "--target", "team:2", "--target", "team:3"],
])
def test_cli_enforces_target_and_time_bounds(flags):
    with pytest.raises(SystemExit):
        arguments(*flags)
