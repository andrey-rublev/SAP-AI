"""Exact-crop OCR memoization; no subprocesses, timers, or desktop access."""
from unittest.mock import Mock

import numpy as np
import pytest

from desktop_runtime import CachedOCR
from desktop_vision import Perceptor, Rect, VisionProfile


def crop(value=0):
    return np.full((2, 3, 3), value, dtype=np.uint8)


@pytest.mark.parametrize("raw", [" 12\n", "", "?", "not a number"])
def test_identical_rgb_copies_reuse_raw_strings_including_blank(raw):
    engine = Mock(return_value=raw)
    cached = CachedOCR(engine)
    assert cached(crop()) == raw
    assert cached(crop().copy()) == raw
    engine.assert_called_once()


def test_one_changed_channel_forces_fresh_ocr_and_original_remains_reusable():
    engine = Mock(side_effect=["1", "2"])
    cached = CachedOCR(engine)
    original = crop()
    assert cached(original) == "1"
    original[0, 0, 2] = 1
    assert cached(original) == "2"
    assert cached(crop()) == "1"
    assert engine.call_count == 2


def test_same_bytes_with_different_shape_or_dtype_do_not_collide():
    engine = Mock(side_effect=["1", "2", "3"])
    cached = CachedOCR(engine)
    image = crop()
    assert cached(image) == "1"
    assert cached(image.reshape((3, 2, 3))) == "2"
    assert cached(image.view(np.int8)) == "3"
    assert engine.call_count == 3


def test_noncontiguous_crop_matches_equivalent_rgb_copy():
    engine = Mock(return_value="7")
    cached = CachedOCR(engine)
    image = np.arange(36, dtype=np.uint8).reshape(2, 6, 3)[:, ::2, :]
    assert not image.flags.c_contiguous
    assert cached(image) == cached(image.copy()) == "7"
    engine.assert_called_once()


def test_lru_eviction_preserves_recently_used_entry():
    engine = Mock(side_effect=["first", "second", "third", "second again"])
    cached = CachedOCR(engine, max_entries=2)
    assert cached(crop(1)) == "first"
    assert cached(crop(2)) == "second"
    assert cached(crop(1)) == "first"  # Promote entry one before inserting three.
    assert cached(crop(3)) == "third"
    assert cached(crop(1)) == "first"
    assert cached(crop(2)) == "second again"
    assert engine.call_count == 4
    assert len(cached._cache) == 2


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), RuntimeError("engine stopped")])
def test_failed_ocr_is_not_cached_and_retry_can_recover(failure):
    engine = Mock(side_effect=[failure, "5"])
    cached = CachedOCR(engine)
    with pytest.raises(type(failure)):
        cached(crop())
    assert cached(crop()) == cached(crop()) == "5"
    assert engine.call_count == 2


def test_non_string_result_is_not_cached():
    engine = Mock(side_effect=[None, "8"])
    cached = CachedOCR(engine)
    assert cached(crop()) is None
    assert cached(crop()) == "8"
    assert engine.call_count == 2


def test_cache_instances_do_not_share_previous_session_readings():
    engine = Mock(side_effect=["3", "4"])
    assert CachedOCR(engine)(crop()) == "3"
    assert CachedOCR(engine)(crop()) == "4"
    assert engine.call_count == 2


def test_perceptor_revalidates_number_range_on_every_cache_hit():
    engine = Mock(return_value="0")
    perceptor = Perceptor(VisionProfile(image_size=(3, 2)), ocr=CachedOCR(engine))
    frame, region = crop(), Rect(0, 0, 3, 2)
    assert perceptor._number(frame, region, 0, 99) == 0
    assert perceptor._number(frame, region, 1, 99) is None
    assert perceptor._number(frame, region, 0, 99) == 0
    engine.assert_called_once()


@pytest.mark.parametrize("max_entries", [0, -1, True, 1.5])
def test_invalid_cache_capacity_is_rejected(max_entries):
    with pytest.raises(ValueError, match="positive integer"):
        CachedOCR(Mock(), max_entries=max_entries)


def test_non_rgb_crop_is_rejected_without_ocr():
    engine = Mock()
    with pytest.raises(ValueError, match="RGB"):
        CachedOCR(engine)(np.zeros((2, 3), dtype=np.uint8))
    engine.assert_not_called()
