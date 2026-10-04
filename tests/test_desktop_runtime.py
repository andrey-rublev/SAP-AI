"""Window IO contracts with fake frames/mouse; never touches a real desktop."""
import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, call
import subprocess
import sys

import numpy as np
import pytest

from desktop import main
import desktop_runtime
from desktop_runtime import DesktopRuntime, DesktopUnavailable, WindowsGameWindow, prepare_numeric_crop, tesseract_ocr, validate_live_dependencies
from desktop_session import DesktopSession
from desktop_state import Action, Board, BoardChangedBeforeInput, DesktopPolicy, PetSlot, Phase
from desktop_vision import Perceptor, Rect, VisionProfile


def board(gold=10):
    return Board(Phase.SHOP, gold=gold, turn=1,
                 shop=(PetSlot(True, attack=2, health=3),), team=(PetSlot(False),) * 5)


def runtime(*, execute=True, current=None):
    profile = SimpleNamespace(
        validate=Mock(), image_size=(100, 100), buttons={"roll": (5, 90), "end_turn": (95, 90)},
        phase_templates=(),
        shop=[SimpleNamespace(portrait=Rect(10, 60, 10, 10))],
        team=[SimpleNamespace(portrait=Rect(10 + i * 15, 20, 10, 10)) for i in range(5)],
    )
    perception = SimpleNamespace(observe=Mock(return_value=current or board()))
    window = SimpleNamespace(capture=Mock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
                             drag=Mock(), click=Mock())
    driver = DesktopRuntime(profile, perception, window, execute=execute)
    driver.last_board = board()
    return driver, window


def test_purchase_targets_correct_client_slot():
    driver, window = runtime()
    driver.act(Action("buy", slot=0, target=4))
    assert window.click.call_args_list == [
        call((15, 65), (100, 100)), call((75, 25), (100, 100)),
    ]
    window.drag.assert_not_called()


def test_merge_selects_shop_pet_then_matching_team_target():
    observed = Board(Phase.SHOP, gold=10, turn=1,
                     shop=(PetSlot(True, "fish", 2, 3, 1),),
                     team=(PetSlot(False), PetSlot(True, "fish", 3, 4, 1),
                           PetSlot(False), PetSlot(False), PetSlot(False)))
    driver, window = runtime(current=observed)
    driver.last_board = observed
    driver.act(Action("merge", slot=0, target=1))
    assert window.click.call_args_list == [
        call((15, 65), (100, 100)), call((30, 25), (100, 100)),
    ]
    window.drag.assert_not_called()


@pytest.mark.parametrize("failed_click", [1, 2])
def test_purchase_click_failure_stops_without_retry(failed_click):
    driver, window = runtime()
    window.click.side_effect = [None] * (failed_click - 1) + [DesktopUnavailable("focus")]
    with pytest.raises(DesktopUnavailable, match="focus"):
        driver.act(Action("buy", 0, 4))
    expected = [call((15, 65), (100, 100)), call((75, 25), (100, 100))]
    assert window.click.call_args_list == expected[:failed_click]
    window.drag.assert_not_called()


def test_purchase_rechecks_focus_before_destination_click():
    driver, _ = runtime()
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.capture = Mock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    # Selecting the shop pet succeeds; focus is lost before placing it.
    window.geometry = Mock(side_effect=[(500, 200, 100, 100)] * 2 + [DesktopUnavailable("focus")])
    window.mouse = Mock()
    driver.window = window
    with pytest.raises(DesktopUnavailable, match="focus"):
        driver.act(Action("buy", 0, 4))
    window.mouse.click.assert_called_once_with(515, 265)
    window.mouse.moveTo.assert_called_once_with(515, 265, duration=0.15)


def test_purchase_rechecks_geometry_before_destination_press():
    driver, _ = runtime()
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.capture = Mock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    # The client moves while the pointer travels to the destination.
    window.geometry = Mock(side_effect=[(500, 200, 100, 100)] * 3 + [(501, 200, 100, 100)])
    window.mouse = Mock()
    driver.window = window
    with pytest.raises(DesktopUnavailable, match="moved"):
        driver.act(Action("buy", 0, 4))
    window.mouse.click.assert_called_once_with(515, 265)
    assert window.mouse.moveTo.call_args_list == [
        call(515, 265, duration=0.15), call(575, 225, duration=0.15),
    ]


def sell_runtime():
    observed = Board(Phase.SHOP, gold=10, turn=1,
                     shop=(PetSlot(True, "duck", 2, 2, 1),),
                     team=(PetSlot(False), PetSlot(False), PetSlot(True, "ant", 2, 3, 1),
                           PetSlot(False), PetSlot(False)))
    driver, window = runtime(current=observed)
    driver.last_board = observed
    driver.profile.buttons["sell"] = (55, 90)
    return driver, window


def test_sell_selects_teammate_then_clicks_calibrated_sell_button():
    driver, window = sell_runtime()
    driver.act(Action("sell", 2))
    assert window.click.call_args_list == [call((45, 25), (100, 100)), call((55, 90), (100, 100))]
    window.drag.assert_not_called()


def test_missing_sell_point_rejects_sequence_before_selecting_teammate():
    driver, window = sell_runtime()
    driver.profile.buttons.pop("sell")
    with pytest.raises(ValueError, match="calibrated sell point"):
        driver.act(Action("sell", 2))
    window.click.assert_not_called()
    window.drag.assert_not_called()


@pytest.mark.parametrize("failed_click", [1, 2])
def test_sell_click_failure_stops_without_retry(failed_click):
    driver, window = sell_runtime()
    window.click.side_effect = [None] * (failed_click - 1) + [DesktopUnavailable("focus")]
    with pytest.raises(DesktopUnavailable, match="focus"):
        driver.act(Action("sell", 2))
    expected = [call((45, 25), (100, 100)), call((55, 90), (100, 100))]
    assert window.click.call_args_list == expected[:failed_click]
    window.drag.assert_not_called()


@pytest.mark.parametrize("failure", ["focus", "moved"])
def test_sell_rechecks_focus_and_geometry_before_pressing_sell(failure):
    driver, _ = sell_runtime()
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.capture = Mock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    if failure == "focus":
        geometry = [(500, 200, 100, 100)] * 2 + [DesktopUnavailable("focus")]
    else:
        geometry = [(500, 200, 100, 100)] * 3 + [(501, 200, 100, 100)]
    window.geometry, window.mouse = Mock(side_effect=geometry), Mock()
    driver.window = window
    with pytest.raises(DesktopUnavailable, match=failure):
        driver.act(Action("sell", 2))
    window.mouse.click.assert_called_once_with(545, 225)
    expected = [call(545, 225, duration=0.15)]
    if failure == "moved":
        expected.append(call(555, 290, duration=0.15))
    assert window.mouse.moveTo.call_args_list == expected


def test_preview_cannot_click():
    driver, window = runtime(execute=False)
    with pytest.raises(DesktopUnavailable, match="disabled"):
        driver.act(Action("roll"))
    window.capture.assert_not_called()
    window.click.assert_not_called()


def test_changed_board_rejected_before_mouse_input():
    current = board(9)
    driver, window = runtime(current=current)
    with pytest.raises(BoardChangedBeforeInput, match="changed") as caught:
        driver.act(Action("buy", 0, 0))
    assert caught.value.preflight_board is current
    assert driver.last_board == board()
    window.drag.assert_not_called()
    window.click.assert_not_called()


def test_changed_board_runtime_session_reobserves_and_replans_before_any_click():
    driver, window = runtime()
    previous = board()
    fresh = Board(Phase.SHOP, gold=10, turn=1, shop=previous.shop,
                  team=(PetSlot(True, attack=4, health=1, level=1), *previous.team[1:]))
    bought = Board(Phase.SHOP, gold=7, turn=1, shop=(PetSlot(False),),
                   team=(fresh.team[0], PetSlot(True, attack=2, health=3), *fresh.team[2:]))
    # A changed pre-input capture rejects target zero. Two new observations
    # must establish the changed board before policy chooses target one.
    driver.perceptor.observe.side_effect = [previous, previous, fresh,
                                            fresh, fresh, fresh, bought, bought, bought]
    events = []
    result = DesktopSession(driver.observe, driver.act, DesktopPolicy(),
                            max_actions=1, max_polls=10, clock=lambda: 0.,
                            sleep=lambda _: None, event_callback=events.append).run()
    assert (result.reason, result.actions, result.acknowledgments, result.polls) == (
        "max_actions", 1, 1, 7,
    )
    assert result.pending_action is None and result.error is None
    assert [(row["poll"], row["action"]["target"]) for row in events
            if row["event"] == "proposed"] == [(2, 0), (4, 1)]
    assert [row["poll"] for row in events if row["event"] == "deferred"] == [2]
    assert [row["preflight_board"] for row in events if row["event"] == "deferred"] == [fresh.to_dict()]
    assert [row["poll"] for row in events if row["event"] == "acted"] == [4]
    assert window.click.call_args_list == [
        call((15, 65), (100, 100)), call((30, 25), (100, 100)),
    ]
    window.drag.assert_not_called()


def test_missing_button_fails_without_input():
    driver, window = runtime()
    del driver.profile.buttons["roll"]
    with pytest.raises(ValueError, match="calibrated"):
        driver.act(Action("roll"))
    window.click.assert_not_called()


def test_illegal_action_rejected_before_capture():
    driver, window = runtime()
    with pytest.raises(ValueError, match="legal"):
        driver.act(Action("sell", 0))
    window.capture.assert_not_called()


def test_stop_requested_before_action_skips_capture_and_input():
    driver, window = runtime()
    driver.should_stop = lambda: True
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        driver.act(Action("buy", 0, 0))
    window.capture.assert_not_called()
    window.click.assert_not_called()


def test_stop_file_created_during_fresh_ocr_prevents_selection(tmp_path):
    driver, window = runtime()
    stop = tmp_path / "STOP"
    driver.should_stop = stop.exists
    def observe(frame):
        stop.touch()
        return board()
    driver.perceptor.observe.side_effect = observe
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        driver.act(Action("buy", 0, 0))
    window.capture.assert_called_once()
    window.click.assert_not_called()


@pytest.mark.parametrize("kind", ["buy", "sell", "choose_name"])
def test_stop_between_selection_clicks_prevents_destination(kind):
    driver, window = (sell_runtime() if kind == "sell" else phase_runtime(Phase.NAMING)
                      if kind == "choose_name" else runtime())
    stopped = [False]
    driver.should_stop = lambda: stopped[0]
    window.click.side_effect = lambda *args: stopped.__setitem__(0, True)
    action = Action("sell", 2) if kind == "sell" else Action("choose_name") if kind == "choose_name" else Action("buy", 0, 0)
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        driver.act(action)
    assert window.click.call_count == 1
    window.drag.assert_not_called()


def test_deadline_expiring_during_phase_preflight_prevents_continuation():
    driver, window = phase_runtime(Phase.ROUND_RESULT)
    clock = [1.0]
    driver.should_stop = lambda: clock[0] >= 2.0
    def observe(frame):
        clock[0] = 2.0
        return Board(Phase.ROUND_RESULT)
    driver.perceptor.observe.side_effect = observe
    with pytest.raises(DesktopUnavailable, match="deadline"):
        driver.act(Action("continue_round"))
    window.click.assert_not_called()


def test_native_click_rechecks_deadline_after_pointer_travel():
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.geometry = Mock(return_value=(500, 200, 100, 100))
    clock = [0.0]
    window.should_stop = lambda: clock[0] >= 1.0
    window.mouse = Mock()
    window.mouse.moveTo.side_effect = lambda *args, **kwargs: clock.__setitem__(0, 1.0)
    with pytest.raises(DesktopUnavailable, match="deadline"):
        window.click((20, 30), (100, 100))
    window.mouse.moveTo.assert_called_once_with(520, 230, duration=0.15)
    window.mouse.click.assert_not_called()


def test_native_drag_stop_after_press_still_releases_button():
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.geometry = Mock(return_value=(0, 0, 100, 100))
    stopped = [False]
    window.should_stop = lambda: stopped[0]
    window.mouse = Mock(FAILSAFE=True)
    window.mouse.mouseDown.side_effect = lambda *args, **kwargs: stopped.__setitem__(0, True)
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        window.drag((10, 20), (30, 40), (100, 100))
    window.mouse.moveTo.assert_called_once_with(10, 20, duration=0.15)
    window.mouse.mouseUp.assert_called_once_with(button="left", _pause=False)
    assert window.mouse.FAILSAFE is True


def test_runtime_shares_stop_predicate_with_native_window():
    driver, _ = runtime()
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    should_stop = lambda: True
    controlled = DesktopRuntime(driver.profile, driver.perceptor, window, execute=True, should_stop=should_stop)
    assert controlled.should_stop is window.should_stop is should_stop


PHASE_TRANSITIONS = (
    (Phase.NAMING, "choose_name", ("name_adjective", "name_noun")),
    (Phase.NAMING_READY, "confirm_name", ("confirm_name",)),
    (Phase.ROUND_RESULT, "continue_round", ("continue_round",)),
    (Phase.TIER_UNLOCK, "dismiss_tier", ("dismiss_tier",)),
    (Phase.LIFE_REWARD, "dismiss_life_reward", ("dismiss_life_reward",)),
    (Phase.END_TURN_CONFIRM, "confirm_end_turn", ("confirm_end_turn",)),
)


MENU_TRANSITIONS = (
    (Phase.MAIN_MENU, "open_play", ("open_play",)),
    (Phase.PLAY_MENU, "open_arena", ("open_arena",)),
    (Phase.ARENA_SETUP, "start_arena", ("start_arena",)),
)


def phase_runtime(phase, *, execute=True):
    current = Board(phase)
    driver, window = runtime(execute=execute, current=current)
    driver.last_board = current
    driver.profile.phase_templates = (SimpleNamespace(phase=phase),)
    driver.profile.buttons.update({
        "name_adjective": (20, 30), "name_noun": (70, 30),
        "confirm_name": (50, 80), "continue_round": (80, 90), "dismiss_tier": (50, 50),
        "dismiss_life_reward": (40, 55),
        "confirm_end_turn": (60, 70),
        "open_play": (20, 80), "open_arena": (50, 80), "start_arena": (80, 80),
    })
    return driver, window


@pytest.mark.parametrize("phase,kind,points", PHASE_TRANSITIONS)
def test_phase_action_requires_both_template_and_all_calibrated_points(phase, kind, points):
    driver, window = phase_runtime(phase)
    assert driver.phase_actions() == {phase: Action(kind)}
    driver.profile.phase_templates = ()
    assert driver.phase_actions() == {}
    driver.profile.phase_templates = (SimpleNamespace(phase=phase),)
    for point in points:
        saved = driver.profile.buttons.pop(point)
        assert driver.phase_actions() == {}
        driver.profile.buttons[point] = saved
    window.capture.assert_not_called()
    window.click.assert_not_called()


def test_phase_action_map_contains_only_supported_explicit_transitions():
    driver, _ = phase_runtime(Phase.NAMING)
    driver.profile.phase_templates = tuple(SimpleNamespace(phase=phase) for phase in Phase)
    driver.profile.buttons["continue"] = (50, 60)
    assert driver.phase_actions() == {phase: Action(kind) for phase, kind, _ in PHASE_TRANSITIONS}
    assert driver.phase_actions(start_arena=True) == {
        phase: Action(kind) for phase, kind, _ in PHASE_TRANSITIONS + MENU_TRANSITIONS
    }
    driver.profile.buttons.pop("name_noun")
    assert driver.phase_actions() == {phase: Action(kind) for phase, kind, _ in PHASE_TRANSITIONS
                                     if phase != Phase.NAMING}


def test_choose_name_clicks_adjective_then_noun_with_calibrated_coordinates():
    driver, window = phase_runtime(Phase.NAMING)
    driver.act(Action("choose_name"))
    assert window.click.call_args_list == [call((20, 30), (100, 100)), call((70, 30), (100, 100))]
    window.drag.assert_not_called()


@pytest.mark.parametrize("missing", ["name_adjective", "name_noun"])
def test_missing_name_point_rejects_whole_sequence_without_partial_selection(missing):
    driver, window = phase_runtime(Phase.NAMING)
    driver.profile.buttons.pop(missing)
    with pytest.raises(ValueError, match="both calibrated name options"):
        driver.act(Action("choose_name"))
    window.click.assert_not_called()
    window.drag.assert_not_called()


@pytest.mark.parametrize("failed_click", [1, 2])
def test_name_selection_failure_is_not_retried(failed_click):
    driver, window = phase_runtime(Phase.NAMING)
    window.click.side_effect = [None] * (failed_click - 1) + [DesktopUnavailable("focus")]
    with pytest.raises(DesktopUnavailable, match="focus"):
        driver.act(Action("choose_name"))
    expected = [call((20, 30), (100, 100)), call((70, 30), (100, 100))]
    assert window.click.call_args_list == expected[:failed_click]
    window.drag.assert_not_called()


@pytest.mark.parametrize("phase,kind,points", PHASE_TRANSITIONS[1:] + MENU_TRANSITIONS)
def test_single_phase_button_dispatches_once(phase, kind, points):
    driver, window = phase_runtime(phase)
    driver.act(Action(kind))
    window.click.assert_called_once_with(driver.profile.buttons[points[0]], (100, 100))
    window.drag.assert_not_called()


@pytest.mark.parametrize("phase,kind,points", PHASE_TRANSITIONS + MENU_TRANSITIONS)
def test_transition_wrong_phase_is_rejected_before_capture_or_input(phase, kind, points):
    driver, window = phase_runtime(Phase.BATTLE)
    with pytest.raises(ValueError, match="not legal"):
        driver.act(Action(kind))
    window.capture.assert_not_called()
    window.click.assert_not_called()


@pytest.mark.parametrize("phase,kind,points", PHASE_TRANSITIONS)
def test_transition_preflight_phase_change_prevents_input(phase, kind, points):
    driver, window = phase_runtime(phase)
    driver.perceptor.observe.return_value = Board(Phase.UNKNOWN)
    with pytest.raises(BoardChangedBeforeInput, match="changed"):
        driver.act(Action(kind))
    window.click.assert_not_called()
    window.drag.assert_not_called()


@pytest.mark.parametrize("phase,kind", [(phase, kind) for phase, kind, _ in
                                      PHASE_TRANSITIONS + MENU_TRANSITIONS])
def test_preview_phase_action_never_captures_or_clicks(phase, kind):
    driver, window = phase_runtime(phase, execute=False)
    with pytest.raises(DesktopUnavailable, match="disabled"):
        driver.act(Action(kind))
    window.capture.assert_not_called()
    window.click.assert_not_called()


@pytest.mark.parametrize("phase,kind,points", MENU_TRANSITIONS)
def test_menu_proposals_require_opt_in_template_and_button(phase, kind, points):
    driver, window = phase_runtime(phase)
    assert driver.phase_actions() == {}
    assert driver.phase_actions(start_arena=True) == {phase: Action(kind)}
    driver.profile.phase_templates = ()
    assert driver.phase_actions(start_arena=True) == {}
    driver.profile.phase_templates = (SimpleNamespace(phase=phase),)
    driver.profile.buttons.pop(points[0])
    assert driver.phase_actions(start_arena=True) == {}
    window.capture.assert_not_called()
    window.click.assert_not_called()


@pytest.mark.parametrize("phase,kind,points", PHASE_TRANSITIONS[1:] + MENU_TRANSITIONS)
def test_missing_single_transition_button_cannot_send_input(phase, kind, points):
    driver, window = phase_runtime(phase)
    driver.profile.buttons.pop(points[0])
    with pytest.raises(ValueError, match="no calibrated"):
        driver.act(Action(kind))
    window.click.assert_not_called()
    window.drag.assert_not_called()


@pytest.mark.parametrize("phase,kind,points", MENU_TRANSITIONS)
def test_menu_preflight_phase_change_prevents_input(phase, kind, points):
    driver, window = phase_runtime(phase)
    driver.perceptor.observe.return_value = Board(Phase.UNKNOWN)
    with pytest.raises(BoardChangedBeforeInput, match="changed"):
        driver.act(Action(kind))
    window.click.assert_not_called()
    window.drag.assert_not_called()


def test_observe_retains_failed_perception_frame_for_diagnostics():
    driver, window = runtime()
    previous_board = driver.observe()
    frame = np.full((100, 100, 3), 17, dtype=np.uint8)
    window.capture.return_value = frame
    driver.perceptor.observe.side_effect = TimeoutError("OCR timeout")
    with pytest.raises(TimeoutError, match="OCR timeout"):
        driver.observe()
    assert driver.last_frame is frame
    assert driver.last_board is previous_board
    assert driver.perceptor.observe.call_args.args[0] is frame
    window.click.assert_not_called()


def test_action_preflight_retains_failed_frame_and_sends_no_input():
    driver, window = runtime()
    previous_board = driver.last_board
    frame = np.full((100, 100, 3), 23, dtype=np.uint8)
    window.capture.return_value = frame
    driver.perceptor.observe.side_effect = ValueError("bad frame")
    with pytest.raises(ValueError, match="bad frame"):
        driver.act(Action("roll"))
    assert driver.last_frame is frame
    assert driver.last_board is previous_board
    assert driver.perceptor.observe.call_args.args[0] is frame
    window.click.assert_not_called()
    window.drag.assert_not_called()


def test_click_applies_client_origin_and_checks_before_press():
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.geometry = Mock(return_value=(500, 200, 100, 100))
    window.mouse = Mock()
    window.click((20, 30), (100, 100))
    window.mouse.moveTo.assert_called_once_with(520, 230, duration=0.15)
    window.mouse.click.assert_called_once_with(520, 230)
    assert window.geometry.call_count == 2


def test_moved_window_cannot_click_after_moving_pointer():
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.geometry = Mock(side_effect=[(500, 200, 100, 100), (501, 200, 100, 100)])
    window.mouse = Mock()
    with pytest.raises(DesktopUnavailable, match="moved"):
        window.click((20, 30), (100, 100))
    window.mouse.click.assert_not_called()


def test_wrong_size_cannot_send_input():
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.geometry = Mock(return_value=(500, 200, 200, 100))
    window.mouse = Mock()
    with pytest.raises(DesktopUnavailable, match="size"):
        window.click((20, 30), (100, 100))
    window.mouse.moveTo.assert_not_called()


def test_wrong_foreground_window_fails_closed():
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.handle = 2
    window.user32 = SimpleNamespace(IsWindow=lambda handle: True, IsIconic=lambda handle: False,
                                    GetForegroundWindow=lambda: 3)
    with pytest.raises(DesktopUnavailable, match="focus"):
        window.geometry()


def activation_window(*, foreground=3, size=(100, 100)):
    """Native window facade with real ctypes structs and mocked Win32 calls."""
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.handle, window.mouse = 2, Mock()
    window._escape_stopped = False
    native = Mock()
    native.IsWindow.return_value = True
    native.IsIconic.return_value = False
    focused = [foreground]
    native.GetForegroundWindow.side_effect = lambda: focused[0]
    def client_rect(handle, pointer):
        rect = ctypes.cast(pointer, ctypes.POINTER(ctypes.wintypes.RECT)).contents
        rect.right, rect.bottom = size
        return True
    def client_origin(handle, pointer):
        point = ctypes.cast(pointer, ctypes.POINTER(ctypes.wintypes.POINT)).contents
        point.x, point.y = 500, 200
        return True
    def foreground_request(handle):
        focused[0] = handle
        return True
    native.GetClientRect.side_effect = client_rect
    native.ClientToScreen.side_effect = client_origin
    native.SetForegroundWindow.side_effect = foreground_request
    window.user32 = native
    return window


@pytest.mark.parametrize("state,stopped", [(0, False), (1, False), (-32768, True), (-32767, True)])
def test_escape_stop_uses_only_the_current_held_key_bit(state, stopped):
    window = activation_window()
    window.user32.GetAsyncKeyState.return_value = state
    assert window.stop_requested() is stopped
    assert window.user32.mock_calls == [call.GetAsyncKeyState(0x1B)]
    assert window.mouse.mock_calls == []


def test_escape_stop_remains_latched_after_release_without_more_key_queries():
    window = activation_window()
    window.user32.GetAsyncKeyState.side_effect = [0, -32768, 0]
    assert window.stop_requested() is False
    assert window.stop_requested() is True
    assert window.stop_requested() is True
    assert window.user32.GetAsyncKeyState.call_args_list == [call(0x1B), call(0x1B)]


def test_escape_latch_and_native_short_prototype_are_initialized_for_each_new_run(monkeypatch):
    native = Mock()
    native.FindWindowW.return_value = 2
    native.GetAsyncKeyState.side_effect = [-32768, 0]
    monkeypatch.setattr(desktop_runtime, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(desktop_runtime.ctypes, "WinDLL", Mock(return_value=native), raising=False)
    monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace())
    first, second = WindowsGameWindow(), WindowsGameWindow()
    assert native.GetAsyncKeyState.argtypes == [ctypes.wintypes.INT]
    assert native.GetAsyncKeyState.restype is ctypes.wintypes.SHORT
    assert first.stop_requested() is True
    assert second.stop_requested() is False
    assert first.stop_requested() is True
    assert native.GetAsyncKeyState.call_args_list == [call(0x1B), call(0x1B)]
    native.SetForegroundWindow.assert_not_called()


def test_escape_stop_accepts_the_signed_result_of_a_typed_native_function():
    window = activation_window()
    callback = ctypes.CFUNCTYPE(ctypes.wintypes.SHORT, ctypes.wintypes.INT)(lambda key: -32768)
    window.user32.GetAsyncKeyState = callback
    assert window.stop_requested() is True
    assert window._escape_stopped is True


def test_held_escape_prevents_startup_focus_handoff():
    window = activation_window()
    window.user32.GetAsyncKeyState.return_value = -32768
    window.should_stop = window.stop_requested
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        window.activate((100, 100))
    assert window.user32.mock_calls == [call.GetAsyncKeyState(0x1B)]
    assert window.mouse.mock_calls == []


def test_escape_held_during_pointer_travel_prevents_mouse_press_and_latches():
    window = activation_window()
    window.geometry = Mock(return_value=(500, 200, 100, 100))
    window.user32.GetAsyncKeyState.side_effect = [0, -32768, 0]
    window.should_stop = window.stop_requested
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        window.click((20, 30), (100, 100))
    window.mouse.moveTo.assert_called_once_with(520, 230, duration=0.15)
    window.mouse.click.assert_not_called()
    assert window.stop_requested() is True
    assert window.user32.GetAsyncKeyState.call_count == 2


def test_escape_after_drag_press_releases_mouse_and_latches_stop():
    window = activation_window()
    window.geometry = Mock(return_value=(500, 200, 100, 100))
    window.mouse.FAILSAFE = True
    window.user32.GetAsyncKeyState.side_effect = [0, 0, -32768, 0]
    window.should_stop = window.stop_requested
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        window.drag((10, 20), (30, 40), (100, 100))
    window.mouse.mouseDown.assert_called_once_with(510, 220, button="left")
    window.mouse.mouseUp.assert_called_once_with(button="left", _pause=False)
    assert window.mouse.FAILSAFE is True
    assert window.stop_requested() is True
    assert window.user32.GetAsyncKeyState.call_count == 3


def test_missing_window_fails_construction_before_mouse_import(monkeypatch):
    native = Mock()
    native.FindWindowW.return_value = 0
    monkeypatch.setattr(desktop_runtime, "os", SimpleNamespace(name="nt"))
    loader = Mock(return_value=native)
    monkeypatch.setattr(desktop_runtime.ctypes, "WinDLL", loader, raising=False)
    monkeypatch.setitem(sys.modules, "pyautogui", None)
    with pytest.raises(DesktopUnavailable, match="open the 'Super Auto Pets' game window first"):
        WindowsGameWindow()
    native.SetForegroundWindow.assert_not_called()


def test_startup_activates_target_once_after_checking_calibrated_size():
    window = activation_window()
    assert window.activate((100, 100)) == (500, 200, 100, 100)
    window.user32.SetForegroundWindow.assert_called_once_with(2)
    calls = window.user32.mock_calls
    assert next(i for i, item in enumerate(calls) if item[0] == "GetClientRect") < calls.index(call.SetForegroundWindow(2))
    assert window.user32.GetClientRect.call_count == 2
    assert window.user32.GetForegroundWindow.call_count == 2
    window.mouse.assert_not_called()
    assert window.mouse.mock_calls == []


def test_startup_does_not_change_focus_when_target_is_already_foreground():
    window = activation_window(foreground=2)
    assert window.activate((100, 100)) == (500, 200, 100, 100)
    window.user32.SetForegroundWindow.assert_not_called()


@pytest.mark.parametrize("state,diagnostic", [("closed", "closed"), ("minimized", "restore the minimized")])
def test_startup_rejects_unavailable_client_before_focus_request(state, diagnostic):
    window = activation_window()
    if state == "closed":
        window.user32.IsWindow.return_value = False
    else:
        window.user32.IsIconic.return_value = True
    with pytest.raises(DesktopUnavailable, match=diagnostic):
        window.activate((100, 100))
    window.user32.GetClientRect.assert_not_called()
    window.user32.SetForegroundWindow.assert_not_called()
    assert window.mouse.mock_calls == []


def test_startup_size_mismatch_cannot_change_foreground():
    window = activation_window(size=(200, 100))
    with pytest.raises(DesktopUnavailable, match="calibrated profile"):
        window.activate((100, 100))
    window.user32.SetForegroundWindow.assert_not_called()
    window.user32.GetForegroundWindow.assert_not_called()


@pytest.mark.parametrize("reported_success", [False, True])
def test_startup_requires_actual_foreground_ownership_without_retry(reported_success):
    window = activation_window()
    window.user32.SetForegroundWindow.side_effect = None
    window.user32.SetForegroundWindow.return_value = reported_success
    with pytest.raises(DesktopUnavailable, match="lost focus"):
        window.activate((100, 100))
    window.user32.SetForegroundWindow.assert_called_once_with(2)
    window.user32.ClientToScreen.assert_not_called()
    assert window.mouse.mock_calls == []


@pytest.mark.parametrize("size", [(100, 100), (200, 100)])
def test_startup_stop_requested_skips_even_native_queries(size):
    window = activation_window(size=size)
    window.should_stop = lambda: True
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        window.activate((100, 100))
    assert window.user32.mock_calls == []


def test_stop_created_during_startup_size_check_prevents_focus_request():
    window = activation_window()
    stopped = [False]
    window.should_stop = lambda: stopped[0]
    read_rect = window.user32.GetClientRect.side_effect
    def stop_after_rect(*args):
        stopped[0] = True
        return read_rect(*args)
    window.user32.GetClientRect.side_effect = stop_after_rect
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        window.activate((100, 100))
    window.user32.SetForegroundWindow.assert_not_called()


@pytest.mark.parametrize("changed", ["size", "closed", "minimized"])
def test_startup_rechecks_client_after_foreground_handoff(changed):
    window = activation_window()
    focus = window.user32.SetForegroundWindow.side_effect
    def change_during_handoff(handle):
        if changed == "size":
            window.user32.GetClientRect.side_effect = activation_window(size=(200, 100)).user32.GetClientRect.side_effect
        elif changed == "closed":
            window.user32.IsWindow.return_value = False
        else:
            window.user32.IsIconic.return_value = True
        return focus(handle)
    window.user32.SetForegroundWindow.side_effect = change_during_handoff
    with pytest.raises(DesktopUnavailable, match="changed during startup" if changed == "size" else "closed or minimized"):
        window.activate((100, 100))
    window.user32.SetForegroundWindow.assert_called_once_with(2)
    assert window.mouse.mock_calls == []


def test_later_capture_guard_does_not_reactivate_a_window_that_lost_focus():
    window = activation_window()
    window.activate((100, 100))
    window.user32.GetForegroundWindow.side_effect = None
    window.user32.GetForegroundWindow.return_value = 3
    with pytest.raises(DesktopUnavailable, match="lost focus"):
        window.geometry()
    window.user32.SetForegroundWindow.assert_called_once_with(2)


@pytest.mark.parametrize("failure", [RuntimeError("corner abort"), KeyboardInterrupt()])
def test_interrupted_drag_always_releases_and_restores_failsafe(failure):
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.geometry = Mock(return_value=(500, 200, 100, 100))
    window.mouse = Mock(FAILSAFE=True)
    window.mouse.moveTo.side_effect = [None, failure]
    release_flags = []
    window.mouse.mouseUp.side_effect = lambda **kwargs: release_flags.append(window.mouse.FAILSAFE)
    with pytest.raises(type(failure)):
        window.drag((10, 20), (30, 40), (100, 100))
    window.mouse.mouseDown.assert_called_once_with(510, 220, button="left")
    window.mouse.mouseUp.assert_called_once_with(button="left", _pause=False)
    assert release_flags == [False]
    assert window.mouse.FAILSAFE is True


def test_drag_releases_when_game_loses_focus_after_press():
    window = WindowsGameWindow.__new__(WindowsGameWindow)
    window.geometry = Mock(side_effect=[(0, 0, 100, 100)] * 3 + [DesktopUnavailable("focus")])
    window.mouse = Mock(FAILSAFE=True)
    with pytest.raises(DesktopUnavailable):
        window.drag((10, 20), (30, 40), (100, 100))
    window.mouse.mouseUp.assert_called_once_with(button="left", _pause=False)
    assert window.mouse.FAILSAFE is True


def test_template_command_is_offline_and_bounds_checked(tmp_path):
    from PIL import Image
    source, target = tmp_path / "frame.png", tmp_path / "template.png"
    Image.new("RGB", (20, 20), "red").save(source)
    main(["template", "--image", str(source), "--region", "1", "2", "3", "4", "--output", str(target)])
    with Image.open(target) as image:
        assert image.size == (3, 4)
    with pytest.raises(ValueError):
        main(["template", "--image", str(source), "--region", "19", "19", "3", "4", "--output", str(target)])


@pytest.fixture
def fake_tesseract(monkeypatch):
    """Replace the optional OCR package before import; never launch a process."""
    ocr = SimpleNamespace(image_to_string=Mock(return_value=" 12\n"), pytesseract=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pytesseract", ocr)
    monkeypatch.setattr("desktop_runtime.shutil.which", lambda name: "tesseract")
    return ocr


def numeric_crop(*, light_background=True):
    """Two synthetic, separate complete glyphs, with no installed font needed."""
    frame = np.full((30, 36, 3), 255 if light_background else 0, dtype=np.uint8)
    foreground = 0 if light_background else 255
    frame[5:25, 6:11] = foreground
    frame[5:25, 20:25] = foreground
    return frame


def test_ocr_passes_finite_subprocess_timeout_and_preserves_preprocessing(fake_tesseract):
    assert tesseract_ocr(numeric_crop()) == "12"
    args, kwargs = fake_tesseract.image_to_string.call_args
    assert kwargs["timeout"] == 3.0
    assert kwargs["config"].startswith("--psm 13 ")
    assert args[0].mode == "L" and args[0].size == (129, 132)


def test_numeric_preprocessing_normalizes_contrast_without_dropping_a_digit():
    light = np.asarray(prepare_numeric_crop(numeric_crop()))
    dark = np.asarray(prepare_numeric_crop(numeric_crop(light_background=False)))
    np.testing.assert_array_equal(light, dark)
    columns = np.any(light == 0, axis=0)
    assert np.count_nonzero(np.diff(np.r_[False, columns, False].astype(int)) == 1) == 2
    assert np.all(light[:36] == 255) and np.all(light[-36:] == 255)


def test_numeric_preprocessing_removes_frame_lines_and_small_noise():
    clean = numeric_crop()
    noisy = clean.copy()
    noisy[0] = 0
    noisy[-1] = 0
    noisy[2, 15] = 0
    np.testing.assert_array_equal(prepare_numeric_crop(clean), prepare_numeric_crop(noisy))


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_clipped_glyph_rejects_entire_value_instead_of_remaining_digit(edge, fake_tesseract):
    frame = numeric_crop()
    if edge == "left":
        frame[5:25, :7] = 0
    elif edge == "right":
        frame[5:25, 24:] = 0
    elif edge == "top":
        frame[:6, 6:11] = 0
    else:
        frame[24:, 6:11] = 0
    assert prepare_numeric_crop(frame) is None
    assert tesseract_ocr(frame) == ""
    fake_tesseract.image_to_string.assert_not_called()


def test_clipped_glyph_joined_to_full_width_border_is_not_dropped(fake_tesseract):
    frame = numeric_crop()
    frame[0] = 0
    frame[:6, 6:11] = 0
    assert prepare_numeric_crop(frame) is None
    assert tesseract_ocr(frame) == ""
    fake_tesseract.image_to_string.assert_not_called()


@pytest.mark.parametrize("value", [0, 128, 255])
def test_blank_crop_is_unknown_without_starting_ocr(value, fake_tesseract):
    assert tesseract_ocr(np.full((30, 40, 3), value, dtype=np.uint8)) == ""
    fake_tesseract.image_to_string.assert_not_called()


def test_ocr_timeout_becomes_clear_fatal_error(fake_tesseract):
    failure = RuntimeError("Tesseract process timeout")
    fake_tesseract.image_to_string.side_effect = failure
    with pytest.raises(TimeoutError, match="Tesseract OCR exceeded 3s timeout") as caught:
        tesseract_ocr(numeric_crop())
    assert caught.value.__cause__ is failure


def test_ocr_timeout_aborts_observation_without_retrying_other_crops_or_clicking(fake_tesseract, monkeypatch):
    fake_tesseract.image_to_string.side_effect = RuntimeError("Tesseract process timeout")
    profile = VisionProfile(image_size=(72, 30),
                            hud={"gold": Rect(0, 0, 36, 30), "turn": Rect(36, 0, 36, 30)})
    perceptor = Perceptor(profile, ocr=tesseract_ocr)
    monkeypatch.setattr(perceptor, "_phase", lambda frame: Phase.SHOP)
    frame = np.concatenate([numeric_crop(), numeric_crop()], axis=1)
    click = Mock()
    runner = DesktopSession(lambda: perceptor.observe(frame), click, DesktopPolicy(),
                            sleep=lambda duration: None)
    result = runner.run()
    assert result.reason == "observation_error"
    assert result.error == "TimeoutError: Tesseract OCR exceeded 3s timeout"
    assert result.actions == 0
    fake_tesseract.image_to_string.assert_called_once()
    click.assert_not_called()


def test_other_ocr_runtime_errors_propagate(fake_tesseract):
    failure = RuntimeError("OCR unavailable")
    fake_tesseract.image_to_string.side_effect = failure
    perceptor = Perceptor(VisionProfile(image_size=(36, 30)), ocr=tesseract_ocr)
    with pytest.raises(RuntimeError, match="OCR unavailable") as caught:
        perceptor._number(numeric_crop(), Rect(0, 0, 36, 30), 0, 99)
    assert caught.value is failure


def test_missing_pytesseract_reports_the_current_python_environment_requirement(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    with pytest.raises(DesktopUnavailable, match="pytesseract in the current Python environment") as caught:
        tesseract_ocr(numeric_crop())
    assert isinstance(caught.value.__cause__, ImportError)


@pytest.fixture
def live_dependencies(monkeypatch):
    """All preflight dependencies are fakes; no child process or package import."""
    specifications = Mock(return_value=object())
    version = Mock(return_value=SimpleNamespace(returncode=0, stdout="tesseract 5.5.0\n", stderr=""))
    monkeypatch.setattr(desktop_runtime.importlib.util, "find_spec", specifications)
    monkeypatch.setattr(desktop_runtime.shutil, "which", lambda name: "mock-tesseract.exe")
    monkeypatch.setattr(desktop_runtime.subprocess, "run", version)
    return specifications, version


def test_live_dependency_preflight_only_queries_specs_and_bounded_hidden_version(live_dependencies, monkeypatch):
    specifications, version = live_dependencies
    import builtins
    original_import = builtins.__import__
    def forbid_live_import(name, *args, **kwargs):
        if name in {"PIL", "pyautogui", "pytesseract"}:
            raise AssertionError("preflight imported a desktop dependency")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", forbid_live_import)
    assert validate_live_dependencies() == "mock-tesseract.exe"
    assert specifications.call_args_list == [call("PIL"), call("pyautogui"), call("pytesseract")]
    version.assert_called_once_with(
        ["mock-tesseract.exe", "--version"], capture_output=True, text=True,
        timeout=5.0, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.mark.parametrize("missing,label", [("PIL", "Pillow"), ("pyautogui", "pyautogui"), ("pytesseract", "pytesseract")])
def test_missing_live_package_is_actionable_and_skips_ocr_process(live_dependencies, missing, label):
    specifications, version = live_dependencies
    specifications.side_effect = lambda name: None if name == missing else object()
    with pytest.raises(DesktopUnavailable, match=f"missing live Python dependencies: {label}") as caught:
        validate_live_dependencies()
    assert '.\\.venv\\Scripts\\python.exe -m pip install -e ".[live]"' in str(caught.value)
    version.assert_not_called()


@pytest.mark.parametrize("error", [ImportError("broken package"), ValueError("missing specification")])
def test_unavailable_live_package_specification_is_reported_as_missing(live_dependencies, error):
    specifications, version = live_dependencies
    specifications.side_effect = error
    with pytest.raises(DesktopUnavailable, match="Pillow, pyautogui, pytesseract"):
        validate_live_dependencies()
    version.assert_not_called()


def test_live_dependency_lookup_prefers_path_without_testing_default_install(live_dependencies, monkeypatch):
    probe = Mock(side_effect=AssertionError("PATH executable should be preferred"))
    monkeypatch.setattr(desktop_runtime.Path, "is_file", probe)
    assert validate_live_dependencies() == "mock-tesseract.exe"
    probe.assert_not_called()


def test_live_dependency_preflight_and_ocr_share_default_executable(live_dependencies, fake_tesseract, monkeypatch):
    monkeypatch.setattr(desktop_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(desktop_runtime.Path, "is_file", lambda path: True)
    executable = validate_live_dependencies()
    assert executable == r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    assert tesseract_ocr(numeric_crop()) == "12"
    assert fake_tesseract.pytesseract.tesseract_cmd == executable
    live_dependencies[1].assert_called_once()


def test_missing_tesseract_installation_fails_without_starting_any_process(live_dependencies, monkeypatch):
    monkeypatch.setattr(desktop_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(desktop_runtime.Path, "is_file", lambda path: False)
    with pytest.raises(DesktopUnavailable, match="install Tesseract OCR and add it to PATH"):
        validate_live_dependencies()
    live_dependencies[1].assert_not_called()


def test_missing_tesseract_during_numeric_ocr_is_actionable(fake_tesseract, monkeypatch):
    monkeypatch.setattr(desktop_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(desktop_runtime.Path, "is_file", lambda path: False)
    with pytest.raises(DesktopUnavailable, match="install Tesseract OCR"):
        tesseract_ocr(numeric_crop())
    fake_tesseract.image_to_string.assert_not_called()


@pytest.mark.parametrize("error,diagnostic", [
    (OSError("permission denied"), "cannot run Tesseract OCR"),
    (subprocess.TimeoutExpired("mock-tesseract.exe", 5), "dependency check exceeded 5s"),
])
def test_live_dependency_version_error_is_bounded_and_actionable(live_dependencies, error, diagnostic):
    version = live_dependencies[1]
    version.side_effect = error
    with pytest.raises(DesktopUnavailable, match=diagnostic) as caught:
        validate_live_dependencies()
    assert caught.value.__cause__ is error
    version.assert_called_once()


@pytest.mark.parametrize("returncode,stdout", [(1, "tesseract 5.5.0"), (0, ""), (0, "some other command 1.0")])
def test_invalid_tesseract_version_result_rejects_preflight(live_dependencies, returncode, stdout):
    version = live_dependencies[1]
    version.return_value = SimpleNamespace(returncode=returncode, stdout=stdout, stderr="private process details")
    with pytest.raises(DesktopUnavailable, match="version check failed") as caught:
        validate_live_dependencies()
    assert "private process details" not in str(caught.value)
    version.assert_called_once()


@pytest.mark.parametrize("failure", [RuntimeError("OCR unavailable"), OSError("OCR process failed")])
@pytest.mark.parametrize("pending", [False, True])
def test_ocr_engine_failure_stops_session_without_more_crops_actions_or_retries(
        fake_tesseract, monkeypatch, failure, pending):
    fake_tesseract.image_to_string.side_effect = failure
    profile = VisionProfile(image_size=(72, 30),
                            hud={"gold": Rect(0, 0, 36, 30), "turn": Rect(36, 0, 36, 30)})
    perceptor = Perceptor(profile, ocr=tesseract_ocr)
    monkeypatch.setattr(perceptor, "_phase", lambda frame: Phase.SHOP)
    frame = np.concatenate([numeric_crop(), numeric_crop()], axis=1)
    initial = iter([board(), board()] if pending else [])
    def observe():
        known = next(initial, None)
        return known if known is not None else perceptor.observe(frame)
    click = Mock()
    result = DesktopSession(observe, click, DesktopPolicy(), sleep=lambda _: None).run()
    assert result.reason == "observation_error"
    assert result.error == f"{type(failure).__name__}: {failure}"
    assert (result.actions, result.acknowledgments, result.polls) == (int(pending), 0, 2 if pending else 0)
    assert result.pending_action == (Action("buy", 0, 0) if pending else None)
    assert click.call_args_list == ([call(Action("buy", 0, 0))] if pending else [])
    fake_tesseract.image_to_string.assert_called_once()
