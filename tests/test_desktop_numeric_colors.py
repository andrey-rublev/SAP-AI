"""Explicit numeric ink filters on synthetic pixels, with no native OCR/input."""
import copy
from dataclasses import FrozenInstanceError, replace
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from sap_ai.desktop.runtime import prepare_numeric_crop, tesseract_ocr
from sap_ai.desktop.vision import NumericColor, Perceptor, VisionProfile


INK = (40, 150, 80)  # Arbitrary synthetic color, not a game calibration.


def color_record(field="gold"):
    return {"field": field, "min_rgb": [20, 120, 60], "max_rgb": [60, 180, 100]}


def numeric_pixels():
    crop = np.full((30, 36, 3), 25, dtype=np.uint8)
    crop[5:25, 6:11] = crop[5:25, 20:25] = INK
    crop[:, 0] = crop[:, -1] = 255  # Neutral badge edges, outside the ink bounds.
    return crop


@pytest.fixture
def scene(tmp_path):
    Image.new("RGB", (4, 4), (200, 200, 200)).save(tmp_path / "shop.png")
    config = {"version": 1, "image_size": [96, 80],
              "phase_templates": [{"phase": "shop", "region": [80, 0, 4, 4],
                                   "template": "shop.png"}],
              "hud": {"gold": [0, 0, 36, 30]}, "numeric_colors": [color_record()]}
    frame = np.zeros((80, 96, 3), dtype=np.uint8)
    frame[:30, :36] = numeric_pixels()
    frame[:4, 80:84] = 200
    return config, frame, tmp_path


def observe(scene, ocr):
    config, frame, directory = scene
    return Perceptor(VisionProfile.from_dict(config, base_dir=directory), ocr=ocr).observe(frame)


@pytest.mark.parametrize("collection", [None, {}, (), "gold", True])
def test_color_collection_requires_an_explicit_list_of_records(scene, collection):
    scene[0]["numeric_colors"] = collection
    with pytest.raises(ValueError, match="numeric_colors"):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("record", [None, 1, [], {},
    {"field": "gold", "min_rgb": [0, 0, 0]},
    {**color_record(), "threshold": 1},
    {**color_record(), "max_rbg": [255, 255, 255]},
])
def test_color_records_reject_unknown_or_missing_keys(scene, record):
    scene[0]["numeric_colors"] = [record]
    with pytest.raises(ValueError, match="numeric color"):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("field", [None, 0, [], {}, "", " gold ", "shop.9.level",
                                  "shop.0.level", "food.0.attack", "food.0.cost", "turn"])
def test_color_field_requires_an_existing_calibrated_numeric_region(scene, field):
    scene[0]["numeric_colors"] = [color_record(field)]
    with pytest.raises(ValueError):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("name", ["min_rgb", "max_rgb"])
@pytest.mark.parametrize("value", [None, "rgb", [], [0, 0], [0, 0, 0, 0],
                                   [True, 0, 0], [0., 0, 0], [-1, 0, 0], [256, 0, 0]])
def test_color_bounds_require_three_exact_bytes(scene, name, value):
    scene[0]["numeric_colors"][0][name] = value
    with pytest.raises(ValueError, match="numeric color"):
        VisionProfile.from_dict(scene[0])


def test_reversed_channel_bounds_duplicate_fields_and_untyped_objects_are_rejected(scene):
    config = copy.deepcopy(scene[0])
    config["numeric_colors"][0]["min_rgb"] = [61, 120, 60]
    with pytest.raises(ValueError, match="min_rgb"):
        VisionProfile.from_dict(config)
    config["numeric_colors"] = [color_record(), color_record()]
    with pytest.raises(ValueError, match="duplicate"):
        VisionProfile.from_dict(config)
    profile = VisionProfile.from_dict(scene[0])
    with pytest.raises(ValueError, match="NumericColor"):
        replace(profile, numeric_colors=(color_record(),)).validate()


@pytest.mark.parametrize("field", ["gold", "shop.0.attack", "team.0.level", "food.0.cost"])
def test_color_paths_cover_hud_pet_and_food_numeric_regions(scene, field):
    config = copy.deepcopy(scene[0])
    config["shop"] = [{"portrait": [0, 40, 4, 4], "attack": [5, 40, 12, 20]}]
    config["team"] = [{"portrait": [20, 40, 4, 4], "level": [25, 40, 12, 20]}]
    config["food"] = [{"portrait": [40, 40, 4, 4], "cost": [45, 40, 12, 20]}]
    config["numeric_colors"] = [color_record(field)]
    assert VisionProfile.from_dict(config).numeric_colors[0].field == field


def test_colors_roundtrip_immutably_and_empty_configuration_preserves_legacy_schema(scene):
    config, _, directory = scene
    profile = VisionProfile.from_dict(config, base_dir=directory)
    assert profile.numeric_colors == (NumericColor("gold", (20, 120, 60), (60, 180, 100)),)
    config["numeric_colors"][0]["min_rgb"][0] = 0
    assert profile.numeric_colors[0].min_rgb == (20, 120, 60)
    with pytest.raises(FrozenInstanceError):
        profile.numeric_colors[0].field = "turn"
    path = directory / "profile.json"
    profile.save(path)
    assert VisionProfile.load(path) == profile
    assert VisionProfile.load(path).to_dict() == profile.to_dict()
    legacy = copy.deepcopy(config)
    legacy.pop("numeric_colors")
    baseline = VisionProfile.from_dict(legacy)
    assert "numeric_colors" not in baseline.to_dict()
    assert VisionProfile.from_dict({**legacy, "numeric_colors": []}).to_dict() == baseline.to_dict()


def test_only_configured_field_ocr_is_masked_with_inclusive_rgb_bounds(scene):
    config, frame, _ = scene
    config["hud"]["turn"] = [40, 0, 36, 30]
    frame[:30, 40:76] = numeric_pixels()
    frame[5:15, 6:11] = config["numeric_colors"][0]["min_rgb"]
    frame[15:25, 6:11] = config["numeric_colors"][0]["max_rgb"]
    for x, value in enumerate(((19, 150, 80), (61, 150, 80), (40, 119, 80),
                               (40, 181, 80), (40, 150, 59), (40, 150, 101))):
        frame[2, 5 + x] = value
    original = frame.copy()
    ocr = Mock(return_value="7")
    board = observe(scene, ocr)
    assert (board.gold, board.turn) == (7, 7)
    assert ocr.call_count == 2
    masked = ocr.call_args_list[0].args[0]
    assert masked.dtype == np.uint8 and masked.shape == (30, 36, 3)
    expected = np.zeros_like(masked)
    expected[5:25, 6:11] = expected[5:25, 20:25] = 255
    np.testing.assert_array_equal(masked, expected)
    np.testing.assert_array_equal(ocr.call_args_list[1].args[0], original[:30, 40:76])
    np.testing.assert_array_equal(frame, original)


def test_absent_colors_preserve_original_ocr_bytes_and_readings(scene):
    config, frame, _ = scene
    config.pop("numeric_colors")
    ocr = Mock(return_value="7")
    assert observe(scene, ocr).gold == 7
    ocr.assert_called_once()
    np.testing.assert_array_equal(ocr.call_args.args[0], frame[:30, :36])


def add_reference(scene, value=7, name="gold.png"):
    config, frame, directory = scene
    Image.fromarray(frame[:30, :36]).save(directory / name)
    config.setdefault("numeric_templates", []).append(
        {"field": "gold", "value": value, "template": name, "max_distance": 0})


@pytest.mark.parametrize("raw,expected", [("7", 7), ("?", 7), ("8", None)])
def test_reference_matching_uses_original_rgb_and_keeps_ocr_disagreement(scene, raw, expected):
    add_reference(scene)
    ocr = Mock(return_value=raw)
    assert observe(scene, ocr).gold == expected
    ocr.assert_called_once()
    assert set(np.unique(ocr.call_args.args[0])) == {0, 255}


def test_ambiguous_original_references_block_colored_ocr_and_reference_only_mode_still_works(scene):
    add_reference(scene)
    assert observe(scene, None).gold == 7
    add_reference(scene, 8, "eight.png")
    ocr = Mock(side_effect=AssertionError("ambiguous references must block OCR"))
    assert observe(scene, ocr).gold is None
    ocr.assert_not_called()


def test_color_filter_does_not_swallow_engine_errors_even_with_a_reference(scene):
    add_reference(scene)
    ocr = Mock(side_effect=TimeoutError("engine timed out"))
    with pytest.raises(TimeoutError, match="engine timed out"):
        observe(scene, ocr)
    ocr.assert_called_once()


@pytest.mark.parametrize("mask", ["blank", "filled", "half", "dense_with_dark_hole"])
@pytest.mark.parametrize("reference", [False, True])
def test_empty_or_dense_masks_abstain_without_reading_dark_holes(scene, mask, reference):
    _, frame, _ = scene
    crop = frame[:30, :36]
    crop[:] = 25 if mask in {"blank", "half"} else INK
    if mask == "half":
        crop[:15] = INK
    elif mask == "dense_with_dark_hole":
        crop[7:23, 10:26] = 25
    if reference:
        add_reference(scene)
    ocr = Mock(side_effect=AssertionError("declined color masks must not invoke OCR"))
    assert observe(scene, ocr).gold == (7 if reference else None)
    ocr.assert_not_called()


@pytest.fixture
def fake_engine(monkeypatch):
    engine = SimpleNamespace(image_to_string=Mock(return_value="12"), pytesseract=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pytesseract", engine)
    monkeypatch.setattr("sap_ai.desktop.runtime._tesseract_executable", lambda: "mock-tesseract")
    return engine.image_to_string


def test_explicit_color_mask_removes_neutral_border_without_loosening_numeric_preparation(scene, fake_engine):
    assert prepare_numeric_crop(scene[1][:30, :36]) is None
    ocr = Mock(wraps=tesseract_ocr)
    assert observe(scene, ocr).gold == 12
    ocr.assert_called_once()
    fake_engine.assert_called_once()
    args, kwargs = fake_engine.call_args
    assert args[0].mode == "L"
    assert kwargs["timeout"] == 3.


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_matching_clipped_ink_still_rejects_whole_value_without_engine_call(scene, fake_engine, edge):
    crop = scene[1][:30, :36]
    if edge == "left":
        crop[5:25, :7] = INK
    elif edge == "right":
        crop[5:25, 24:] = INK
    elif edge == "top":
        crop[:6, 6:11] = INK
    else:
        crop[24:, 6:11] = INK
    ocr = Mock(wraps=tesseract_ocr)
    assert observe(scene, ocr).gold is None
    ocr.assert_called_once()
    assert prepare_numeric_crop(ocr.call_args.args[0]) is None
    fake_engine.assert_not_called()
