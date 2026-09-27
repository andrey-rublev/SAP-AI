"""Tests for the pure logic in autogui (no real mouse/OCR involved)."""
import numpy as np
import pytest
from unittest.mock import MagicMock

import autogui


def obs(team):
    return np.array([10, 1, 6, 5, 4, *team], dtype=np.float32)


def test_weakest_slot_none_without_obs():
    assert autogui._weakest_team_slot(None) is None


def test_weakest_slot_picks_lowest_occupied():
    assert autogui._weakest_team_slot(obs([0, 3, 0, 7, 2])) == 4  # value 2 is lowest


def test_weakest_slot_none_when_empty_team():
    assert autogui._weakest_team_slot(obs([0, 0, 0, 0, 0])) is None


def test_perform_sell_requires_observation():
    with pytest.raises(ValueError):
        autogui.perform_action(4, obs=None)


def test_perform_action_rejects_unknown():
    with pytest.raises(ValueError):
        autogui.perform_action(99)


def test_parse_int_extracts_digits():
    assert autogui.parse_int("A 3 / 5") == 3
    assert autogui.parse_int("no digits", default=7) == 7
    assert autogui.parse_int("") == 0


def test_build_observation_shape_and_values():
    obs = autogui.build_observation(gold=10, shop=[1, 2, 3], team=[4, 5, 0, 0, 0], turn=2)
    assert obs.shape == (10,)
    assert list(obs) == [10, 2, 1, 2, 3, 4, 5, 0, 0, 0]


def test_build_observation_pads_short_lists():
    obs = autogui.build_observation(gold=5, shop=[6], team=[], turn=0)
    assert list(obs) == [5, 0, 6, 0, 0, 0, 0, 0, 0, 0]


def test_build_observation_feeds_mask_from_obs():
    from game import SuperAutoPetsEnv

    obs = autogui.build_observation(gold=10, shop=[0, 4, 0], team=[0, 0, 0, 0, 0])
    mask = SuperAutoPetsEnv.mask_from_obs(obs)
    assert list(mask) == [False, True, False, True, False, True]


@pytest.fixture
def desktop(monkeypatch):
    """Every desktop action in these tests is simulated, never real IO."""
    fake = MagicMock()
    fake.size.return_value = (1920, 1080)
    monkeypatch.setattr(autogui, "pyautogui", fake)
    monkeypatch.setattr(autogui, "LAYOUT", autogui.Layout(calibrated=True))
    monkeypatch.setattr(autogui, "_CONTROL_ENABLED", True)
    return fake


def test_layout_roundtrip_and_placeholder_rejection(tmp_path):
    path = tmp_path / "layout.json"
    autogui.Layout().save(path)
    with pytest.raises(ValueError, match="uncalibrated"):
        autogui.Layout.load(path)
    autogui.Layout(calibrated=True).save(path)
    loaded = autogui.Layout.load(path)
    assert tuple(loaded.shop_slots[1]) == (430, 620)
    assert loaded.calibrated is True


@pytest.mark.parametrize("field,value", [
    ("team_slots", [(0, 0)] * 5),
    ("shop_slots", [(0, 0)]),
    ("gold_region", (0, 0, -1, 20)),
    ("end_turn_button", (1920, 100)),
    ("roll_button", (1.5, 20)),
    ("screen_size", (0, 1080)),
    ("calibrated", "true"),
])
def test_layout_rejects_invalid_geometry(field, value):
    layout = autogui.Layout(calibrated=True)
    setattr(layout, field, value)
    with pytest.raises(ValueError):
        layout.validate()


def test_layout_rejects_incomplete_json(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text('{"version": 1, "calibrated": true}')
    with pytest.raises(ValueError, match="missing"):
        autogui.Layout.load(path)


def test_wrong_resolution_prevents_click_and_screenshot(desktop):
    desktop.size.return_value = (1280, 720)
    with pytest.raises(ValueError, match="resolution"):
        autogui.capture_screen()
    with pytest.raises(ValueError, match="resolution"):
        autogui.perform_action(5, obs([0] * 5))
    desktop.screenshot.assert_not_called()
    desktop.moveTo.assert_not_called()


def test_default_layout_never_touches_desktop(monkeypatch):
    monkeypatch.setattr(autogui, "LAYOUT", autogui.Layout())
    dependency = MagicMock(side_effect=AssertionError("desktop must not load"))
    monkeypatch.setattr(autogui, "_dependency", dependency)
    with pytest.raises(ValueError, match="uncalibrated"):
        autogui.perform_action(5, obs([0] * 5))
    dependency.assert_not_called()


def test_calibration_does_not_implicitly_enable_clicks(desktop, monkeypatch):
    monkeypatch.setattr(autogui, "_CONTROL_ENABLED", False)
    with pytest.raises(RuntimeError, match="disabled"):
        autogui.perform_action(5, obs([0] * 5))
    desktop.moveTo.assert_not_called()


def test_buy_destination_uses_empty_team_slot_not_shop_index(desktop):
    autogui.perform_action(0, obs([9, 8, 0, 0, 0]))
    desktop.moveTo.assert_called_once_with(300, 620, duration=0.15)
    desktop.dragTo.assert_called_once_with(940, 380, duration=0.3, button="left")


def test_scalar_match_only_used_when_explicitly_requested_for_toy_mapping():
    state = obs([0, 8, 6, 0, 0])
    assert autogui._buy_target_slot(0, state) == 0
    assert autogui._buy_target_slot(0, state, allow_scalar_merge=True) == 2


def test_full_team_scalar_match_is_not_a_live_merge(desktop):
    state = obs([6, 7, 8, 9, 10])
    assert not autogui.live_action_mask(state)[:3].any()
    with pytest.raises(ValueError, match="not supported"):
        autogui.perform_action(0, state)
    desktop.moveTo.assert_not_called()


def test_sell_uses_only_team_slots_when_metadata_present(desktop):
    state = np.append(obs([9, 8, 7, 6, 5]), [0, 1, 0])
    autogui.perform_action(4, state)
    desktop.moveTo.assert_called_once_with(1180, 380, duration=0.15)
    desktop.dragTo.assert_called_once_with(960, 900, duration=0.3, button="left")


@pytest.mark.parametrize("text", ["", "?", "3 / 5", "-1", "1O", "2 3"])
def test_unreadable_number_is_not_silently_zero(monkeypatch, text):
    monkeypatch.setattr(autogui, "capture_screen", lambda region: np.zeros((10, 10, 3)))
    monkeypatch.setattr(autogui, "read_text", lambda image: text)
    with pytest.raises(autogui.PerceptionError):
        autogui.read_number((0, 0, 10, 10))


def test_read_board_uses_one_snapshot_and_keeps_legacy_schema(monkeypatch):
    capture = MagicMock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8))
    monkeypatch.setattr(autogui, "capture_screen", capture)
    numbers = iter(["10", "1", "2", "3", "4", "5", "0", "0", "0"])
    monkeypatch.setattr(autogui, "read_text", lambda image: next(numbers))
    assert list(autogui.read_board(turn=2)) == [10, 2, 1, 2, 3, 4, 5, 0, 0, 0]
    capture.assert_called_once_with()


def test_optional_dependency_failure_has_actionable_error(monkeypatch):
    monkeypatch.setattr(autogui, "pyautogui", None)
    def unavailable(name):
        raise KeyError("DISPLAY")
    monkeypatch.setattr(autogui.importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="desktop session"):
        autogui._dependency("pyautogui")


def test_template_command_does_not_read_the_screen(tmp_path, monkeypatch):
    capture = MagicMock(side_effect=AssertionError("must not read screen"))
    monkeypatch.setattr(autogui, "capture_screen", capture)
    path = tmp_path / "layout.json"
    autogui.main(["--write-template", str(path)])
    assert path.is_file()
    with pytest.raises(SystemExit):
        autogui.main(["--write-template", str(path)])
    capture.assert_not_called()
