"""Startup lifecycle contracts with mocked Steam, clocks and window discovery."""
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import desktop_launch
from desktop_launch import GAME_TITLE, STEAM_GAME_URI, connect_game_window, launch_installed_game
from desktop_runtime import DesktopUnavailable, GameWindowNotFound


def missing(stop=None):
    return GameWindowNotFound("game is closed", stop_requested=stop)


class Clock:
    def __init__(self):
        self.now = 0.
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.waits.append(duration)
        self.now += duration


def test_existing_window_never_launches_or_changes_focus():
    window, launch = Mock(), Mock()
    factory = Mock(return_value=window)
    assert connect_game_window(GAME_TITLE, factory, launch_game=launch) is window
    factory.assert_called_once_with(GAME_TITLE)
    launch.assert_not_called()
    assert window.mock_calls == []


def test_cold_start_launches_once_and_waits_for_the_client_without_input():
    window, launch, clock = Mock(), Mock(), Clock()
    factory = Mock(side_effect=[missing(), missing(), missing(), window])
    assert connect_game_window(GAME_TITLE, factory, launch_game=launch,
                               timeout=2, clock=clock, sleep=clock.sleep) is window
    launch.assert_called_once_with()
    assert factory.call_args_list == [call(GAME_TITLE)] * 4
    assert clock.waits == [.5, .5]
    assert window.mock_calls == []


def test_cold_start_waits_for_saved_geometry_after_the_window_title_appears():
    window, launch, clock = Mock(), Mock(), Clock()
    ready = Mock(side_effect=[False, False, True])
    factory = Mock(side_effect=[missing(), window, window, window])
    assert connect_game_window(GAME_TITLE, factory, launch_game=launch, is_ready=ready,
                               timeout=2, clock=clock, sleep=clock.sleep) is window
    assert clock.waits == [.5, .5] and ready.call_count == 3
    launch.assert_called_once()
    assert window.mock_calls == []


def test_cold_start_geometry_wait_preserves_deadline_without_focus_or_input():
    window, launch, clock = Mock(), Mock(), Clock()
    factory = Mock(side_effect=[missing(), window, window])
    with pytest.raises(DesktopUnavailable, match="timed out"):
        connect_game_window(GAME_TITLE, factory, launch_game=launch, is_ready=lambda _: False,
                            timeout=.75, clock=clock, sleep=clock.sleep)
    assert clock.now == .75
    launch.assert_called_once()
    assert window.mock_calls == []


def test_existing_window_keeps_immediate_geometry_validation_outside_launch_wait():
    window, ready = Mock(), Mock(return_value=False)
    assert connect_game_window(GAME_TITLE, Mock(return_value=window), is_ready=ready) is window
    ready.assert_not_called()


@pytest.mark.parametrize("title,launch", [(GAME_TITLE, False), ("custom game", True)])
def test_disabled_or_nonstandard_launch_preserves_missing_window_error(title, launch):
    start = Mock()
    error = missing()
    with pytest.raises(GameWindowNotFound) as caught:
        connect_game_window(title, Mock(side_effect=error), launch=launch, launch_game=start)
    assert caught.value is error
    start.assert_not_called()


def test_other_discovery_failures_never_launch_or_retry():
    error, start = DesktopUnavailable("invalid geometry"), Mock()
    factory = Mock(side_effect=error)
    with pytest.raises(DesktopUnavailable, match="invalid geometry"):
        connect_game_window(GAME_TITLE, factory, launch_game=start)
    start.assert_not_called()
    factory.assert_called_once()


def test_existing_stop_skips_discovery_and_launch():
    factory, start = Mock(), Mock()
    assert connect_game_window(GAME_TITLE, factory, should_stop=lambda: True, launch_game=start) is None
    factory.assert_not_called()
    start.assert_not_called()


def test_escape_latch_from_missing_native_window_prevents_launch():
    start = Mock()
    key_stop = Mock(return_value=True)
    factory = Mock(side_effect=missing(key_stop))
    assert connect_game_window(GAME_TITLE, factory, launch_game=start) is None
    factory.assert_called_once()
    key_stop.assert_called_once()
    start.assert_not_called()


def test_escape_during_startup_stops_polling_without_a_client_or_input():
    start, clock = Mock(), Clock()
    key_stop = Mock(side_effect=[False, False, True])
    factory = Mock(side_effect=missing(key_stop))
    assert connect_game_window(GAME_TITLE, factory, launch_game=start,
                               clock=clock, sleep=clock.sleep) is None
    start.assert_called_once()
    assert factory.call_count == 2 and clock.waits == [.5]


def test_external_runtime_budget_stops_launch_wait_early():
    start, clock = Mock(), Clock()
    factory = Mock(side_effect=missing())
    assert connect_game_window(GAME_TITLE, factory, launch_game=start,
                               timeout=90, should_stop=lambda: clock.now >= 1,
                               clock=clock, sleep=clock.sleep) is None
    assert clock.now == 1 and factory.call_count == 3
    start.assert_called_once()


def test_launch_timeout_is_finite_and_reports_manual_sign_in_without_retrying_launch():
    start, clock = Mock(), Clock()
    factory = Mock(side_effect=missing())
    with pytest.raises(DesktopUnavailable, match="complete any sign-in yourself"):
        connect_game_window(GAME_TITLE, factory, launch_game=start,
                            timeout=.75, clock=clock, sleep=clock.sleep)
    assert clock.now == .75 and clock.waits == [.5, .25]
    start.assert_called_once()


def test_window_appearing_after_launch_deadline_cannot_be_returned_for_control():
    window, clock = Mock(), Clock()
    calls = [0]
    def discover(_):
        calls[0] += 1
        if calls[0] == 1:
            raise missing()
        clock.now = 2
        return window
    with pytest.raises(DesktopUnavailable, match="timed out"):
        connect_game_window(GAME_TITLE, discover, launch_game=Mock(),
                            timeout=1, clock=clock, sleep=clock.sleep)
    assert window.mock_calls == []


def test_stop_during_window_discovery_cannot_return_client_for_control():
    window = Mock()
    stopped = [False]
    def discover(_):
        stopped[0] = True
        return window
    assert connect_game_window(GAME_TITLE, discover, should_stop=lambda: stopped[0]) is None
    assert window.mock_calls == []


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "90"])
def test_invalid_timeout_cannot_discover_or_launch(timeout):
    factory, start = Mock(), Mock()
    with pytest.raises(ValueError, match="launch timeout"):
        connect_game_window(GAME_TITLE, factory, timeout=timeout, launch_game=start)
    factory.assert_not_called()
    start.assert_not_called()


def test_installed_game_launch_uses_only_its_fixed_steam_identifier(monkeypatch):
    start = Mock()
    monkeypatch.setattr(desktop_launch, "os", SimpleNamespace(name="nt", startfile=start))
    launch_installed_game()
    start.assert_called_once_with(STEAM_GAME_URI)


def test_steam_launch_failure_is_actionable_without_fallback_commands(monkeypatch):
    start = Mock(side_effect=OSError("missing protocol handler"))
    monkeypatch.setattr(desktop_launch, "os", SimpleNamespace(name="nt", startfile=start))
    with pytest.raises(DesktopUnavailable, match="open the game manually"):
        launch_installed_game()
    start.assert_called_once()


def test_non_windows_cannot_launch_steam(monkeypatch):
    start = Mock()
    monkeypatch.setattr(desktop_launch, "os", SimpleNamespace(name="posix", startfile=start))
    with pytest.raises(DesktopUnavailable, match="requires Windows"):
        launch_installed_game()
    start.assert_not_called()
