"""Window IO contracts with fake frames/mouse; never touches a real desktop."""
from types import SimpleNamespace
from unittest.mock import Mock, call
import sys

import numpy as np
import pytest

from desktop import main
from desktop_runtime import DesktopRuntime, DesktopUnavailable, WindowsGameWindow, tesseract_ocr
from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase
from desktop_vision import Perceptor, Rect, VisionProfile


def board(gold=10):
    return Board(Phase.SHOP, gold=gold, turn=1,
                 shop=(PetSlot(True, attack=2, health=3),), team=(PetSlot(False),) * 5)


def runtime(*, execute=True, current=None):
    profile = SimpleNamespace(
        validate=Mock(), image_size=(100, 100), buttons={"roll": (5, 90), "end_turn": (95, 90)},
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


def test_ocr_passes_finite_subprocess_timeout_and_preserves_preprocessing(fake_tesseract):
    assert tesseract_ocr(np.zeros((2, 3, 3), dtype=np.uint8)) == "12"
    args, kwargs = fake_tesseract.image_to_string.call_args
    assert kwargs["timeout"] == 3.0
    assert args[0].mode == "L" and args[0].size == (9, 6)


def test_ocr_timeout_becomes_clear_fatal_error(fake_tesseract):
    failure = RuntimeError("Tesseract process timeout")
    fake_tesseract.image_to_string.side_effect = failure
    with pytest.raises(TimeoutError, match="Tesseract OCR exceeded 3s timeout") as caught:
        tesseract_ocr(np.zeros((1, 1, 3), dtype=np.uint8))
    assert caught.value.__cause__ is failure


def test_ocr_timeout_aborts_observation_without_retrying_other_crops_or_clicking(fake_tesseract, monkeypatch):
    fake_tesseract.image_to_string.side_effect = RuntimeError("Tesseract process timeout")
    profile = VisionProfile(image_size=(2, 1),
                            hud={"gold": Rect(0, 0, 1, 1), "turn": Rect(1, 0, 1, 1)})
    perceptor = Perceptor(profile, ocr=tesseract_ocr)
    monkeypatch.setattr(perceptor, "_phase", lambda frame: Phase.SHOP)
    frame = np.zeros((1, 2, 3), dtype=np.uint8)
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
    perceptor = Perceptor(VisionProfile(image_size=(1, 1)), ocr=tesseract_ocr)
    assert perceptor._number(np.zeros((1, 1, 3), dtype=np.uint8), Rect(0, 0, 1, 1), 0, 99) is None
