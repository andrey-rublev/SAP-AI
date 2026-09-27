"""Synthetic image tests; never capture the screen or move the mouse."""
import copy

import numpy as np
import pytest

from desktop_state import Phase
from desktop_vision import Perceptor, Rect, VisionProfile, image_distance, read_number


def save_template(path, color, size=(4, 4)):
    from PIL import Image
    Image.new("RGB", size, (color, color, color)).save(path)


@pytest.fixture
def scene(tmp_path):
    save_template(tmp_path / "shop.png", 200)
    save_template(tmp_path / "battle.png", 100)
    save_template(tmp_path / "result.png", 50)
    save_template(tmp_path / "empty.png", 0)
    save_template(tmp_path / "ant.png", 150)
    save_template(tmp_path / "fish.png", 230)
    config = {
        "version": 1,
        "image_size": [40, 20],
        "calibrated": True,
        "phase_templates": [
            {"phase": phase.value, "region": [0, 0, 4, 4], "template": f"{phase.value}.png"}
            for phase in (Phase.SHOP, Phase.BATTLE, Phase.RESULT)
        ],
        "hud": {"gold": [5, 0, 1, 1], "turn": [6, 0, 1, 1], "wins": [7, 0, 1, 1], "lives": [8, 0, 1, 1]},
        "shop": [{"portrait": [0, 5, 4, 4], "attack": [5, 5, 1, 1], "health": [6, 5, 1, 1], "level": [7, 5, 1, 1], "empty_template": "empty.png", "species_templates": {"ant": "ant.png", "fish": "fish.png"}}],
        "team": [{"portrait": [10, 5, 4, 4], "attack": [15, 5, 1, 1], "health": [16, 5, 1, 1], "empty_template": "empty.png"}],
        "buttons": {"roll": [1, 15], "end_turn": [30, 15], "sell": [20, 15], "continue": [30, 10]},
    }
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    frame[:4, :4] = 200
    frame[5:9, :4] = 150
    for x, text_code in ((5, 10), (6, 1), (7, 0), (8, 5)):
        frame[0, x] = text_code
    frame[5, 5] = 2
    frame[5, 6] = 3
    frame[5, 7] = 1
    frame[5, 15:17] = 255  # unreadable stats on explicitly empty team slot
    ocr = lambda image: str(int(image[0, 0, 0])) if int(image[0, 0, 0]) != 255 else "?"
    return config, frame, ocr, tmp_path


def observe(scene, **kwargs):
    config, frame, ocr, directory = scene
    return Perceptor(VisionProfile.from_dict(config, base_dir=directory), ocr=kwargs.get("ocr", ocr)).observe(frame)


def test_reads_one_calibrated_shop_frame(scene):
    board = observe(scene)
    assert board.phase == Phase.SHOP
    assert (board.gold, board.turn, board.wins, board.lives) == (10, 1, 0, 5)
    assert (board.shop[0].occupied, board.shop[0].species, board.shop[0].attack, board.shop[0].health, board.shop[0].level) == (True, "ant", 2, 3, 1)
    assert board.team[0].occupied is False


@pytest.mark.parametrize("phase,color", [(Phase.BATTLE, 100), (Phase.RESULT, 50), (Phase.UNKNOWN, 255)])
def test_other_phases_never_read_shop_ocr(scene, phase, color):
    scene[1][:4, :4] = color
    def unexpected_ocr(image):
        pytest.fail("non-shop frames must not attempt shop OCR")
    board = observe(scene, ocr=unexpected_ocr)
    assert board.phase == phase
    assert board.gold is None and not board.shop and not board.team


def test_ambiguous_phase_returns_unknown(scene):
    config, frame, ocr, directory = scene
    config["phase_templates"][1]["template"] = "shop.png"
    assert observe(scene).phase == Phase.UNKNOWN


def test_multiple_references_of_same_phase_do_not_compete(scene):
    scene[0]["phase_templates"].append(copy.deepcopy(scene[0]["phase_templates"][0]))
    assert observe(scene).phase == Phase.SHOP


@pytest.mark.parametrize("attack,health", [(255, 3), (2, 255), (255, 255), (2, 0), (100, 3)])
def test_unreadable_or_invalid_stats_are_unknown_not_pets(scene, attack, health):
    scene[1][5, 5] = attack
    scene[1][5, 6] = health
    assert observe(scene).shop[0].occupied is None


def test_unknown_species_does_not_erase_verified_stats(scene):
    scene[1][5:9, :4] = 80
    slot = observe(scene).shop[0]
    assert slot.occupied is True and slot.attack == 2 and slot.species is None


def test_ambiguous_species_does_not_invent_merge_identity(scene):
    config = scene[0]
    config["shop"][0]["species_templates"]["fish"] = "ant.png"
    slot = observe(scene).shop[0]
    assert slot.occupied is True and slot.species is None


def test_empty_portrait_with_stats_is_contradictory(scene):
    scene[1][5:9, :4] = 0
    assert observe(scene).shop[0].occupied is None


@pytest.mark.parametrize("species_color,frame_color", [(2, 2), (2, 0), (0, 0), (10, 10)])
def test_empty_requires_margin_over_species_even_when_ocr_is_blank(scene, species_color, frame_color):
    # Both references match the threshold: species may tie, nearly tie, or win.
    save_template(scene[3] / "ant.png", species_color)
    scene[1][5:9, :4] = frame_color
    scene[1][5, 5:8] = 255
    slot = observe(scene).shop[0]
    assert slot.occupied is None
    assert slot.species is None


def test_species_pixels_and_missing_ocr_never_authorize_an_empty_target(scene):
    save_template(scene[3] / "ant.png", 2)
    scene[1][5:9, :4] = 2
    assert observe(scene, ocr=None).shop[0].occupied is None


def test_empty_can_win_by_a_clear_margin_with_blank_ocr(scene):
    save_template(scene[3] / "ant.png", 10)
    scene[1][5:9, :4] = 0
    scene[1][5, 5:8] = 255
    assert observe(scene).shop[0].occupied is False


def test_empty_requires_visual_evidence(scene):
    scene[0]["team"][0].pop("empty_template")
    assert observe(scene).team[0].occupied is None


def test_ocr_absence_or_exception_preserves_unknown(scene):
    board = observe(scene, ocr=None)
    assert board.gold is None and board.shop[0].occupied is None
    assert board.team[0].occupied is False
    def broken_ocr(image):
        raise RuntimeError("OCR subprocess unavailable")
    assert observe(scene, ocr=broken_ocr).gold is None


@pytest.mark.parametrize("text", ["", "?", "3/5", "1O", "2 3", "-1", "+1", "1.0", "100", "١", None, 3])
def test_strict_numbers(text):
    assert read_number(text, 0, 99) is None


def test_numbers_preserve_real_zero_and_whitespace():
    assert read_number(" 0\n", 0, 99) == 0
    assert read_number("0", 1, 99) is None
    assert read_number("50", 0, 99) == 50


@pytest.mark.parametrize("frame", [np.zeros((10, 20, 3), dtype=np.uint8), np.zeros((20, 40, 4), dtype=np.uint8), np.zeros((20, 40, 3), dtype=np.float32)])
def test_wrong_size_format_or_dtype_rejected_before_ocr(scene, frame):
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    with pytest.raises(ValueError, match="RGB uint8"):
        Perceptor(profile).observe(frame)


def test_wrong_template_size_rejected(scene):
    save_template(scene[3] / "shop.png", 200, size=(2, 2))
    with pytest.raises(ValueError, match="dimensions"):
        observe(scene)


def test_missing_template_rejected(scene):
    scene[0]["phase_templates"][0]["template"] = "missing.png"
    with pytest.raises(FileNotFoundError):
        observe(scene)


def test_profile_roundtrip_relative_templates(scene):
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    path = scene[3] / "profile.json"
    profile.save(path)
    loaded = VisionProfile.load(path)
    assert loaded == profile
    assert loaded.buttons["roll"] == (1, 15)
    assert loaded.shop[0].portrait.center == (2, 7)
    assert Perceptor(loaded, ocr=scene[2]).observe(scene[1]).gold == 10


@pytest.mark.parametrize("key,value", [("version", 2), ("version", True), ("image_size", [40.0, 20]), ("image_size", [0, 20]), ("calibrated", "true"), ("unexpected", 1)])
def test_invalid_profile_rejected(scene, key, value):
    scene[0][key] = value
    with pytest.raises(ValueError):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("rect", [[-1, 0, 4, 4], [39, 0, 4, 4], [0, 0, 0, 4], [0, 0, 4.0, 4], [0, 0, 4]])
def test_invalid_regions_rejected(scene, rect):
    scene[0]["hud"]["gold"] = rect
    with pytest.raises(ValueError):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("name", ["../private.png", "/tmp/private.png", "", None])
def test_template_paths_cannot_escape_profile_directory(scene, name):
    scene[0]["phase_templates"][0]["template"] = name
    with pytest.raises(ValueError, match="template paths"):
        VisionProfile.from_dict(scene[0], base_dir=scene[3])


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_matching_thresholds_rejected(scene, threshold):
    scene[0]["phase_templates"][0]["max_distance"] = threshold
    with pytest.raises(ValueError, match="max_distance"):
        VisionProfile.from_dict(scene[0])


def test_uncalibrated_profile_can_be_inspected_but_not_enabled(scene):
    scene[0]["calibrated"] = False
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    with pytest.raises(ValueError, match="explicitly calibrated"):
        profile.validate(require_calibrated=True)
    assert Perceptor(profile).observe(scene[1]).phase == Phase.SHOP


def test_distance_uses_signed_arithmetic():
    dark = np.zeros((2, 2, 3), dtype=np.uint8)
    light = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert image_distance(dark, light) == image_distance(light, dark) == 1
    assert image_distance(dark, dark) == 0


def test_crop_does_not_allow_negative_or_overflow_geometry():
    with pytest.raises(ValueError):
        Rect(-1, 0, 1, 1).crop(np.zeros((2, 2, 3), dtype=np.uint8))
