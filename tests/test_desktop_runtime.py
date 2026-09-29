"""Window IO contracts with fake frames/mouse; never touches a real desktop."""
from types import SimpleNamespace
from unittest.mock import Mock, call
import sys

import numpy as np
import pytest

from desktop import main
from desktop_runtime import DesktopRuntime, DesktopUnavailable, WindowsGameWindow, prepare_numeric_crop, tesseract_ocr
from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase
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
    driver, window = runtime(current=board(9))
    with pytest.raises(DesktopUnavailable, match="changed"):
        driver.act(Action("buy", 0, 0))
    window.drag.assert_not_called()
    window.click.assert_not_called()


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


PHASE_TRANSITIONS = (
    (Phase.NAMING, "choose_name", ("name_adjective", "name_noun")),
    (Phase.NAMING_READY, "confirm_name", ("confirm_name",)),
    (Phase.ROUND_RESULT, "continue_round", ("continue_round",)),
    (Phase.TIER_UNLOCK, "dismiss_tier", ("dismiss_tier",)),
)


def phase_runtime(phase, *, execute=True):
    current = Board(phase)
    driver, window = runtime(execute=execute, current=current)
    driver.last_board = current
    driver.profile.phase_templates = (SimpleNamespace(phase=phase),)
    driver.profile.buttons.update({
        "name_adjective": (20, 30), "name_noun": (70, 30),
        "confirm_name": (50, 80), "continue_round": (80, 90), "dismiss_tier": (50, 50),
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


@pytest.mark.parametrize("phase,kind,points", PHASE_TRANSITIONS[1:])
def test_single_phase_button_dispatches_once(phase, kind, points):
    driver, window = phase_runtime(phase)
    driver.act(Action(kind))
    window.click.assert_called_once_with(driver.profile.buttons[points[0]], (100, 100))
    window.drag.assert_not_called()


@pytest.mark.parametrize("phase,kind,points", PHASE_TRANSITIONS)
def test_transition_wrong_phase_is_rejected_before_capture_or_input(phase, kind, points):
    driver, window = phase_runtime(Phase.BATTLE)
    with pytest.raises(ValueError, match="not legal"):
        driver.act(Action(kind))
    window.capture.assert_not_called()
    window.click.assert_not_called()


def test_phase_change_during_preflight_prevents_name_selection():
    driver, window = phase_runtime(Phase.NAMING)
    driver.perceptor.observe.return_value = Board(Phase.NAMING_READY)
    with pytest.raises(DesktopUnavailable, match="changed"):
        driver.act(Action("choose_name"))
    window.click.assert_not_called()


def test_preview_phase_action_never_captures_or_clicks():
    driver, window = phase_runtime(Phase.NAMING, execute=False)
    with pytest.raises(DesktopUnavailable, match="disabled"):
        driver.act(Action("choose_name"))
    window.capture.assert_not_called()
    window.click.assert_not_called()


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
    ocr = SimpleNamespace(image_to_string=Mock(return_value=" 12\n"))
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


def test_other_ocr_runtime_errors_still_yield_unknown(fake_tesseract):
    fake_tesseract.image_to_string.side_effect = RuntimeError("OCR unavailable")
    perceptor = Perceptor(VisionProfile(image_size=(36, 30)), ocr=tesseract_ocr)
    assert perceptor._number(numeric_crop(), Rect(0, 0, 36, 30), 0, 99) is None
