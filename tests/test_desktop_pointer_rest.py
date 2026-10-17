"""Calibrated pointer parking and tooltip occlusion without desktop access."""
import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from desktop_runtime import DesktopRuntime, DesktopUnavailable, WindowsGameWindow
from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase
from desktop_vision import Rect, SlotConfig, VisionProfile


def profile(rest=(50, 95)):
    return VisionProfile(image_size=(100, 100), calibrated=True, pointer_rest=rest,
                         shop=(SlotConfig(Rect(10, 60, 10, 10)),),
                         team=tuple(SlotConfig(Rect(10 + index * 15, 20, 10, 10)) for index in range(5)))


def test_pointer_rest_round_trips_separately_from_gameplay_buttons(tmp_path):
    original = profile()
    original.validate(require_calibrated=True)
    assert original.to_dict()["pointer_rest"] == [50, 95]
    assert original.to_dict()["buttons"] == {}
    path = tmp_path / "profile.json"
    original.save(path)
    assert VisionProfile.load(path).pointer_rest == (50, 95)
    assert VisionProfile.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_legacy_profile_schema_omits_unconfigured_pointer_rest():
    data = profile(None).to_dict()
    assert "pointer_rest" not in data
    restored = VisionProfile.from_dict(data)
    assert restored.pointer_rest is None and restored.to_dict() == data


@pytest.mark.parametrize("point", [True, "50,95", [1], [1, 2, 3], [True, 2], [1.5, 2], [-1, 2], [100, 2], [2, 100]])
def test_pointer_rest_rejects_wrong_types_and_out_of_bounds_coordinates(point):
    with pytest.raises(ValueError, match="pointer_rest"):
        replace(profile(), pointer_rest=point).validate()
    data = profile(None).to_dict()
    data["pointer_rest"] = point
    with pytest.raises(ValueError, match="pointer_rest"):
        VisionProfile.from_dict(data)


def native_window():
    target = WindowsGameWindow.__new__(WindowsGameWindow)
    target.handle, target.mouse = 42, Mock()
    target.should_stop = lambda: False
    target.user32 = Mock()
    target.user32.IsWindow.return_value = True
    target.user32.IsIconic.return_value = False
    target.user32.GetForegroundWindow.return_value = 42
    def bounds(handle, pointer):
        rect = ctypes.cast(pointer, ctypes.POINTER(ctypes.wintypes.RECT)).contents
        rect.right, rect.bottom = 100, 100
        return True
    def origin(handle, pointer):
        point = ctypes.cast(pointer, ctypes.POINTER(ctypes.wintypes.POINT)).contents
        point.x, point.y = 500, 200
        return True
    target.user32.GetClientRect.side_effect = bounds
    target.user32.ClientToScreen.side_effect = origin
    return target


def test_native_hover_only_moves_inside_the_calibrated_foreground_client():
    target = native_window()
    target.hover((20, 30), (100, 100))
    assert target.mouse.mock_calls == [call.moveTo(520, 230, duration=.15)]
    target.user32.SetForegroundWindow.assert_not_called()
    target.user32.PostMessageW.assert_not_called()


@pytest.mark.parametrize("failure", ["focus", "size", "closed", "minimized", "point", "stop"])
def test_invalid_hover_cannot_move_or_press_any_mouse_button(failure):
    target = native_window()
    if failure == "focus":
        target.user32.GetForegroundWindow.return_value = 99
    elif failure == "closed":
        target.user32.IsWindow.return_value = False
    elif failure == "minimized":
        target.user32.IsIconic.return_value = True
    elif failure == "stop":
        target.should_stop = lambda: True
    with pytest.raises((DesktopUnavailable, ValueError)):
        target.hover((100, 30) if failure == "point" else (20, 30),
                     (101, 100) if failure == "size" else (100, 100))
    assert target.mouse.mock_calls == []


@pytest.mark.parametrize("failure", ["focus", "moved", "stop"])
def test_hover_rechecks_after_travel_without_any_followup_input(failure):
    target = native_window()
    if failure == "moved":
        target.geometry = Mock(side_effect=[(500, 200, 100, 100), (501, 200, 100, 100)])
    else:
        def after_move(*args, **kwargs):
            if failure == "focus":
                target.user32.GetForegroundWindow.return_value = 99
            else:
                target.should_stop = lambda: True
        target.mouse.moveTo.side_effect = after_move
    with pytest.raises(DesktopUnavailable):
        target.hover((20, 30), (100, 100))
    assert target.mouse.mock_calls == [call.moveTo(520, 230, duration=.15)]


def tooltip_runtime(*, execute=True, rest=(50, 95)):
    empty, fish = PetSlot(False), PetSlot(True, "fish", 2, 3, 1)
    before = Board(Phase.SHOP, gold=3, turn=10, shop=(fish,), team=(empty,) * 5)
    after = replace(before, gold=0, shop=(empty,), team=(fish, empty, empty, empty, empty))
    state = {"tooltip": True, "purchased": False, "clicks": 0}
    history = []
    window = SimpleNamespace(hover=Mock(), click=Mock(), capture=Mock())
    def hover(point, size):
        history.append(("park", state["tooltip"]))
        state["tooltip"] = False
    def capture():
        history.append(("capture", state["tooltip"]))
        return np.full((100, 100, 3), int(state["tooltip"]), dtype=np.uint8)
    def observe(frame):
        board = after if state["purchased"] else before
        if frame[0, 0, 0]:
            # A tooltip hides stats; perception retains that uncertainty.
            occupied = PetSlot(True, "fish", None, None, 1)
            return replace(board, team=(occupied, empty, empty, empty, empty))
        return board
    def click(point, size):
        state["clicks"] += 1
        state["tooltip"] = True
        if state["clicks"] == 2:
            state["purchased"] = True
    window.hover.side_effect, window.capture.side_effect, window.click.side_effect = hover, capture, click
    driver = DesktopRuntime(profile(rest), SimpleNamespace(observe=observe), window, execute=execute,
                            should_stop=lambda: False)
    return driver, window, history, state, before, after


def test_startup_and_post_input_parking_clear_tooltip_before_observation_and_acknowledgment():
    driver, window, history, state, before, after = tooltip_runtime()
    result = DesktopSession(driver.observe, driver.act, DesktopPolicy(), max_actions=1, max_polls=5,
                            clock=lambda: 0., sleep=lambda _: None).run()
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == ("max_actions", 1, 1, 5)
    assert result.last_board == after and result.pending_action is None
    assert window.click.call_args_list == [call((15, 65), (100, 100)), call((15, 25), (100, 100))]
    assert window.hover.call_args_list == [call((50, 95), (100, 100))] * 2
    assert [entry for entry in history if entry[0] == "park"] == [("park", True), ("park", True)]
    assert all(not occluded for kind, occluded in history if kind == "capture")


@pytest.mark.parametrize("execute,rest", [(False, (50, 95)), (True, None)])
def test_preview_and_unconfigured_profiles_keep_pointer_and_uncertainty_unchanged(execute, rest):
    driver, window, _, _, before, _ = tooltip_runtime(execute=execute, rest=rest)
    observed = driver.observe()
    assert observed.team[0].occupied is True and observed.team[0].attack is None
    window.hover.assert_not_called()
    window.click.assert_not_called()


def test_idle_observations_do_not_repeat_pointer_movement():
    driver, window, _, _, _, _ = tooltip_runtime()
    driver.observe()
    driver.observe()
    driver.observe()
    window.hover.assert_called_once()


def test_stop_before_startup_parking_prevents_movement_capture_and_click():
    driver, window, _, _, _, _ = tooltip_runtime()
    driver.should_stop = lambda: True
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        driver.observe()
    window.hover.assert_not_called()
    window.capture.assert_not_called()
    window.click.assert_not_called()


def test_startup_parking_focus_failure_cannot_capture_or_click():
    driver, window, _, _, _, _ = tooltip_runtime()
    window.hover.side_effect = DesktopUnavailable("game lost focus")
    with pytest.raises(DesktopUnavailable, match="focus"):
        driver.observe()
    window.capture.assert_not_called()
    window.click.assert_not_called()


def test_post_input_parking_failure_retains_pending_purchase_and_never_retries():
    driver, window, _, _, _, _ = tooltip_runtime()
    successful_park = window.hover.side_effect
    parks = 0
    def park(point, size):
        nonlocal parks
        parks += 1
        if parks == 2:
            raise DesktopUnavailable("lost focus during parking")
        return successful_park(point, size)
    window.hover.side_effect = park
    result = DesktopSession(driver.observe, driver.act, DesktopPolicy(), max_actions=2, max_polls=10,
                            clock=lambda: 0., sleep=lambda _: None).run()
    assert result.reason == "observation_error" and result.actions == 1 and result.acknowledgments == 0
    assert result.pending_action == Action("buy", 0, 0) and "focus" in result.error
    assert window.click.call_count == 2 and window.hover.call_count == 2


def test_missing_stats_without_a_tooltip_remain_unknown_after_parking():
    driver, window, _, _, before, _ = tooltip_runtime()
    unreadable = replace(before, shop=(PetSlot(True, "fish", None, None, 1),))
    driver.perceptor = SimpleNamespace(observe=lambda frame: unreadable)
    assert driver.observe() == unreadable
    window.hover.assert_called_once()
    window.click.assert_not_called()
