"""Normal WM_CLOSE lifecycle with mocked Win32 calls and deterministic time."""
import ctypes
from unittest.mock import Mock

import pytest

from desktop_runtime import DesktopUnavailable, WindowsGameWindow


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


def window():
    result = WindowsGameWindow.__new__(WindowsGameWindow)
    result.handle = 42
    result.mouse = Mock()
    result.should_stop = lambda: False
    native = Mock()
    native.IsWindow.return_value = True
    native.IsIconic.return_value = False
    native.GetForegroundWindow.return_value = 42
    native.PostMessageW.return_value = True
    def client_rect(handle, pointer):
        rect = ctypes.cast(pointer, ctypes.POINTER(ctypes.wintypes.RECT)).contents
        rect.right, rect.bottom = 100, 100
        return True
    native.GetClientRect.side_effect = client_rect
    native.ClientToScreen.return_value = True
    result.user32 = native
    return result


def test_normal_close_posts_once_and_returns_only_after_window_disappears():
    target, clock = window(), Clock()
    target.user32.IsWindow.side_effect = [True, True, True, False]
    assert target.close_for_restart((100, 100), timeout=1, clock=clock, sleep=clock.sleep) is True
    assert clock.now == pytest.approx(.2)
    target.user32.PostMessageW.assert_called_once_with(42, 0x0010, 0, 0)
    target.user32.SetForegroundWindow.assert_not_called()
    assert target.mouse.mock_calls == []


@pytest.mark.parametrize("state", ["closed", "minimized", "focus", "size"])
def test_invalid_target_never_receives_a_close(state):
    target = window()
    if state == "closed":
        target.user32.IsWindow.return_value = False
    elif state == "minimized":
        target.user32.IsIconic.return_value = True
    elif state == "focus":
        target.user32.GetForegroundWindow.return_value = 50
    with pytest.raises(DesktopUnavailable):
        target.close_for_restart((200, 100) if state == "size" else (100, 100))
    target.user32.PostMessageW.assert_not_called()
    assert target.mouse.mock_calls == []


def test_existing_stop_skips_even_geometry_or_close_queries():
    target = window()
    target.should_stop = lambda: True
    assert target.close_for_restart((100, 100)) is False
    assert target.user32.mock_calls == []


def test_stop_after_geometry_prevents_the_native_close():
    target = window()
    target.should_stop = Mock(side_effect=[False, True])
    assert target.close_for_restart((100, 100)) is False
    target.user32.PostMessageW.assert_not_called()


def test_stop_while_shutdown_is_pending_cannot_report_ready_for_relaunch():
    target, clock = window(), Clock()
    target.should_stop = lambda: clock.now >= .1
    assert target.close_for_restart((100, 100), clock=clock, sleep=clock.sleep) is False
    target.user32.PostMessageW.assert_called_once()
    assert clock.now == pytest.approx(.1)


def test_stopping_when_target_disappears_still_blocks_relaunch():
    target = window()
    target.user32.IsWindow.side_effect = [True, False]
    target.should_stop = Mock(side_effect=[False, False, True])
    assert target.close_for_restart((100, 100)) is False
    target.user32.PostMessageW.assert_called_once()


def test_native_close_failure_is_not_retried():
    target = window()
    target.user32.PostMessageW.return_value = False
    with pytest.raises(DesktopUnavailable, match="rejected normal shutdown"):
        target.close_for_restart((100, 100))
    target.user32.PostMessageW.assert_called_once()


def test_unclosed_target_times_out_without_any_second_close_request():
    target, clock = window(), Clock()
    with pytest.raises(DesktopUnavailable, match="timed out waiting"):
        target.close_for_restart((100, 100), timeout=.25, clock=clock, sleep=clock.sleep)
    assert clock.now == .25
    target.user32.PostMessageW.assert_called_once()


def test_focus_loss_during_pending_shutdown_stops_without_another_request():
    target = window()
    target.user32.GetForegroundWindow.side_effect = [42, 42, 50]
    with pytest.raises(DesktopUnavailable, match="lost focus during shutdown"):
        target.close_for_restart((100, 100))
    target.user32.PostMessageW.assert_called_once()
    target.user32.SetForegroundWindow.assert_not_called()


def test_close_race_does_not_mistake_a_destroyed_window_for_focus_loss():
    target = window()
    target.user32.IsWindow.side_effect = [True, True, False, False]
    target.user32.GetForegroundWindow.side_effect = [42, 42, 50]
    assert target.close_for_restart((100, 100)) is True
    target.user32.PostMessageW.assert_called_once()


def test_focus_loss_after_geometry_cannot_receive_a_close_request():
    target = window()
    target.user32.GetForegroundWindow.side_effect = [42, 50]
    with pytest.raises(DesktopUnavailable, match="lost focus before shutdown"):
        target.close_for_restart((100, 100))
    target.user32.PostMessageW.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "90"])
def test_invalid_close_deadline_never_touches_the_window(timeout):
    target = window()
    with pytest.raises(ValueError, match="close timeout"):
        target.close_for_restart((100, 100), timeout=timeout)
    assert target.user32.mock_calls == []
