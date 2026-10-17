"""Synthetic title crops and injected OCR only; no captures or OCR processes."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

import desktop_tooltip as tooltip


def title_crop(text="Fish"):
    image = Image.new("RGB", (120, 30), (25, 25, 25))
    ImageDraw.Draw(image).text((6, 6), text, fill=(245, 170, 35), font=ImageFont.load_default())
    return np.asarray(image).copy()


def test_readable_title_requires_agreement_from_two_distinct_preparations():
    image = title_crop()
    original = image.copy()
    engine = Mock(side_effect=[" Fish ", "FISH"])
    assert tooltip.read_pet_title(image, ["fish", "ant"], ocr=engine) == "fish"
    assert engine.call_count == 2
    raw, foreground = (call.args[0] for call in engine.call_args_list)
    assert raw.ndim == 3 and raw.shape[2] == 3
    assert foreground.ndim == 2 and set(np.unique(foreground)) == {0, 255}
    assert raw.shape[:2] == foreground.shape
    assert np.array_equal(image, original)


@pytest.mark.parametrize("readings,canonical", [(["Woolly - Mammoth", "woolly mammoth"], "woolly mammoth"),
                                                (["sea\t lion", "SEA-LION"], "sea lion"),
                                                (["FISH", "fish"], "Fish")])
def test_normalization_matches_exact_words_and_returns_provided_vocabulary_spelling(readings, canonical):
    engine = Mock(side_effect=readings)
    assert tooltip.read_pet_title(title_crop(), [canonical], ocr=engine) == canonical
    assert engine.call_count == 2


@pytest.mark.parametrize("reading", [None, 5, [], "", "F1sh", "fish!", "Fi sh", "Fish Ant", "fish/ant",
                                     "fish\nant", "fish\n", "fish\rbody", "fish gives friends attack",
                                     "unicorn", "fiſh", "fi\u200bsh", "sea--lion"])
def test_unknown_nonword_bodytext_multiline_or_nonstring_never_becomes_a_known_name(reading):
    engine = Mock(return_value=reading)
    assert tooltip.read_pet_title(title_crop(), ["fish", "ant", "sea lion"], ocr=engine) is None
    assert engine.call_count == 1


@pytest.mark.parametrize("second", ["ant", None, "fish!", "fish\nant", "unicorn"])
def test_second_reading_cannot_repair_or_override_first(second):
    engine = Mock(side_effect=["fish", second])
    assert tooltip.read_pet_title(title_crop(), ["fish", "ant"], ocr=engine) is None
    assert engine.call_count == 2


def test_normalized_vocabulary_collision_is_ambiguous():
    engine = Mock(return_value="sea lion")
    assert tooltip.read_pet_title(title_crop(), ["sea lion", "sea-lion"], ocr=engine) is None
    assert engine.call_count == 1


@pytest.mark.parametrize("background", [(0, 0, 0), (255, 255, 255), (120, 120, 120), (200, 80, 10)])
def test_blank_or_no_colored_evidence_does_not_call_even_an_optimistic_ocr(background):
    engine = Mock(return_value="fish")
    image = np.full((30, 120, 3), background, dtype=np.uint8)
    assert tooltip.read_pet_title(image, ["fish"], ocr=engine) is None
    engine.assert_not_called()


def test_grayscale_text_and_isolated_colored_noise_are_not_colored_title_evidence():
    image = Image.new("RGB", (120, 30), "black")
    ImageDraw.Draw(image).text((6, 6), "Fish", fill="white", font=ImageFont.load_default())
    pixels = np.asarray(image).copy()
    pixels[0, 0] = (245, 170, 35)
    engine = Mock(return_value="fish")
    assert tooltip.read_pet_title(pixels, ["fish"], ocr=engine) is None
    engine.assert_not_called()


def test_separate_colored_title_lines_are_rejected_before_ocr():
    image = np.concatenate((title_crop(), title_crop("Ant")), axis=0)
    engine = Mock(return_value="fish")
    assert tooltip.read_pet_title(image, ["fish", "ant"], ocr=engine) is None
    engine.assert_not_called()


@pytest.mark.parametrize("image", [np.zeros((3, 4), dtype=np.uint8), np.zeros((3, 4, 4), dtype=np.uint8),
                                   np.zeros((0, 4, 3), dtype=np.uint8), np.zeros((3, 0, 3), dtype=np.uint8),
                                   np.zeros((3, 4, 3), dtype=np.float32), np.array(0, dtype=np.uint8)])
def test_invalid_rgb_input_fails_without_ocr(image):
    engine = Mock()
    with pytest.raises(ValueError, match="uint8 RGB"):
        tooltip.read_pet_title(image, ["fish"], ocr=engine)
    engine.assert_not_called()


def test_empty_vocabulary_makes_no_claim_and_invalid_vocabulary_is_explicit():
    engine = Mock()
    assert tooltip.read_pet_title(title_crop(), [], ocr=engine) is None
    with pytest.raises(TypeError, match="iterable"):
        tooltip.read_pet_title(title_crop(), "fish", ocr=engine)
    with pytest.raises(ValueError, match="alphabetic"):
        tooltip.read_pet_title(title_crop(), ["fish", 5], ocr=engine)
    with pytest.raises(TypeError, match="callable"):
        tooltip.read_pet_title(title_crop(), ["fish"], ocr=None)
    engine.assert_not_called()


def test_injected_engine_timeout_is_not_retried_or_reported_as_a_reading():
    engine = Mock(side_effect=TimeoutError("test timeout"))
    with pytest.raises(TimeoutError, match="test timeout"):
        tooltip.read_pet_title(title_crop(), ["fish"], ocr=engine)
    assert engine.call_count == 1


def fake_tesseract(monkeypatch, result="Fish\n"):
    engine = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=""), image_to_string=Mock(return_value=result))
    monkeypatch.setitem(sys.modules, "pytesseract", engine)
    monkeypatch.setattr(tooltip.shutil, "which", lambda name: "test-tesseract")
    return engine


def test_text_engine_uses_bounded_single_call_without_erasing_invalid_characters(monkeypatch):
    engine = fake_tesseract(monkeypatch, "  Fish2!\n")
    assert tooltip.tesseract_text_ocr(title_crop()) == "Fish2!"
    assert engine.image_to_string.call_count == 1
    options = engine.image_to_string.call_args.kwargs
    assert 0 < options["timeout"] <= 3
    assert "whitelist" not in options["config"]


def test_missing_python_ocr_dependency_has_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    with pytest.raises(tooltip.TooltipOCRUnavailable, match="pytesseract and Pillow"):
        tooltip.tesseract_text_ocr(title_crop())


def test_missing_executable_has_clear_error_and_does_not_call_engine(monkeypatch):
    engine = fake_tesseract(monkeypatch)
    monkeypatch.setattr(tooltip.shutil, "which", lambda name: None)
    monkeypatch.setattr(tooltip.Path, "is_file", lambda path: False)
    with pytest.raises(tooltip.TooltipOCRUnavailable, match="executable"):
        tooltip.tesseract_text_ocr(title_crop())
    engine.image_to_string.assert_not_called()


@pytest.mark.parametrize("failure,error,match", [(RuntimeError("Tesseract process timeout"), TimeoutError, "3s timeout"),
                                                (RuntimeError("engine failed"), tooltip.TooltipOCRError, "OCR failed"),
                                                (OSError("missing executable"), tooltip.TooltipOCRUnavailable, "installation")])
def test_engine_failures_remain_explicit_without_retry(monkeypatch, failure, error, match):
    engine = fake_tesseract(monkeypatch)
    engine.image_to_string.side_effect = failure
    with pytest.raises(error, match=match):
        tooltip.tesseract_text_ocr(title_crop())
    assert engine.image_to_string.call_count == 1
