"""Synthetic transition pixels; no recorded images, native IO or OCR process."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
from PIL import Image
import pytest

from sap_ai.desktop.runtime import CachedOCR, tesseract_ocr
from sap_ai.desktop.session import DesktopSession
from sap_ai.desktop.state import Action, DesktopPolicy, PetSlot, Phase
from sap_ai.desktop.vision import Perceptor, PhaseTemplate, Rect, SlotConfig, VisionProfile


def transition_scene(tmp_path):
    Image.new("RGB", (4, 4), (200, 200, 200)).save(tmp_path / "shop.png")
    Image.new("RGB", (4, 4), (0, 0, 0)).save(tmp_path / "empty.png")
    profile = VisionProfile(
        image_size=(300, 120), base_dir=tmp_path,
        phase_templates=(PhaseTemplate(Phase.SHOP, Rect(0, 0, 4, 4), "shop.png"),),
        hud={name: Rect(5 + index, 0, 1, 1)
             for index, name in enumerate(("gold", "turn", "wins", "lives"))},
        shop=(SlotConfig(Rect(5, 8, 4, 4), Rect(5, 20, 36, 30),
                         Rect(45, 20, 36, 30), empty_template="empty.png"),),
        team=tuple(SlotConfig(Rect(5 + index * 55, 75, 4, 4),
                              Rect(5 + index * 55, 85, 20, 30),
                              Rect(28 + index * 55, 85, 20, 30),
                              empty_template="empty.png") for index in range(5)),
    )
    frame = np.zeros((120, 300, 3), dtype=np.uint8)
    frame[:4, :4] = 200
    frame[0, 5:9] = np.array((3, 9, 4, 1), dtype=np.uint8)[:, None]
    frame[8:12, 5:9] = 150
    # A connected synthetic "3"; the optional engine is mocked, but its real
    # complete-glyph preprocessing still decides whether OCR may run at all.
    glyph = np.array([[value == "1" for value in row] for row in
                      ("11110", "00001", "00001", "01110",
                       "00001", "00001", "11110")])
    glyph = np.repeat(np.repeat(glyph, 3, axis=0), 3, axis=1)
    for region in (profile.shop[0].attack, profile.shop[0].health):
        crop = region.crop(frame)
        crop[:] = 105
        crop[4:25, 10:25][glyph] = 255
        frame[region.y:region.y + region.height, region.x:region.x + region.width] = crop
    return profile, frame


@pytest.mark.parametrize("field", ["attack", "health"])
@pytest.mark.parametrize("transient", ["top_clip", "bottom_clip", "separate_badge_rim"])
def test_partial_stat_transition_abstains_then_requires_fresh_stable_pixels(
        tmp_path, monkeypatch, field, transient):
    profile, settled = transition_scene(tmp_path)
    animated = settled.copy()
    region = getattr(profile.shop[0], field)
    crop = region.crop(animated)
    if transient == "top_clip":
        crop[:5, 10:13] = 255
    elif transient == "bottom_clip":
        crop[24:, 10:13] = 255
    else:
        # The numeral is complete, but a separate tall edge fragment cannot
        # safely be distinguished from an additional clipped numeral.
        crop[18:, :3] = 255
    animated[region.y:region.y + region.height, region.x:region.x + region.width] = crop

    engine = Mock(return_value="3")
    monkeypatch.setitem(sys.modules, "pytesseract", SimpleNamespace(
        image_to_string=engine, pytesseract=SimpleNamespace()))
    monkeypatch.setattr("sap_ai.desktop.runtime._tesseract_executable", lambda: "mock-tesseract")

    def read_crop(image):
        if image.shape == (1, 1, 3):
            return str(int(image[0, 0, 0]))
        return tesseract_ocr(image)

    perceptor = Perceptor(profile, ocr=CachedOCR(read_crop))
    frames = iter((animated, settled, settled.copy()))
    boards, events = [], []

    def observe():
        board = perceptor.observe(next(frames))
        boards.append(board)
        return board

    act = Mock()
    policy = DesktopPolicy()
    result = DesktopSession(
        observe, act, policy, preview=True, max_polls=3,
        clock=lambda: 0., sleep=lambda _: None, event_callback=events.append,
    ).run()

    # Even the readable companion stat is withheld while one field is clipped;
    # unknown occupancy never becomes empty and cannot authorize a purchase.
    assert boards[0].shop == (PetSlot(None),)
    assert policy.choose_action(boards[0]) is None
    assert boards[1].shop == boards[2].shop == (PetSlot(True, attack=3, health=3),)
    assert boards[0].fingerprint() != boards[1].fingerprint()
    assert [event["stable_frames"] for event in events if event["event"] == "observed"] == [1, 1, 2]
    assert (result.reason, result.polls, result.actions, result.acknowledgments) == ("preview", 3, 0, 0)
    assert result.proposed_action == Action("buy", 0, 0)
    assert engine.called
    act.assert_not_called()
