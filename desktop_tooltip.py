"""Offline exact pet-title OCR for an explicitly supplied RGB title crop.

This module never captures a screen, moves a pointer, or updates a profile.
Names contain ASCII letters with spaces or single ASCII hyphens between words.
Matching ignores case, collapses horizontal spaces, and treats a hyphen as a
space. It never repairs spelling, removes body text, or searches substrings.
The caller's vocabulary spelling is returned only when both preparations agree.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import re
import shutil

import numpy as np


TITLE_OCR_TIMEOUT_SECONDS = 3.0
_NAME = re.compile(r"[A-Za-z]+(?:(?:[ \t]+|[ \t]*-[ \t]*)[A-Za-z]+)*")
_PADDING = 12
_SCALE = 3


class TooltipOCRUnavailable(RuntimeError):
    """The optional text OCR dependency or executable is unavailable."""


class TooltipOCRError(RuntimeError):
    """The text OCR engine failed; no title was established."""


def _normalized_name(value):
    if not isinstance(value, str):
        return None
    value = value.strip(" \t")
    if _NAME.fullmatch(value) is None:
        return None
    return " ".join(value.replace("-", " ").split()).casefold()


def _rgb_crop(image):
    crop = np.asarray(image)
    if (crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3
            or min(crop.shape[:2]) <= 0):
        raise ValueError("pet title needs a nonempty uint8 RGB crop")
    return crop.copy()


def _preparations(crop):
    # A fixed saturation/value gate isolates colored title letters. Grayscale
    # body text and uniform blank/background crops cannot supply evidence.
    maximum = crop.max(axis=2).astype(np.int16)
    minimum = crop.min(axis=2).astype(np.int16)
    foreground = (maximum >= 100) & (maximum - minimum >= 50)
    count = int(foreground.sum())
    if count < 12 or not .005 <= count / foreground.size <= .45:
        return None
    ys, xs = np.where(foreground)
    if xs.max() - xs.min() < 5 or ys.max() - ys.min() < 4:
        return None
    # A title is one line; separated bands of colored text are not a title crop.
    occupied_rows = foreground.any(axis=1)
    starts = occupied_rows & ~np.r_[False, occupied_rows[:-1]]
    if int(starts.sum()) != 1:
        return None
    raw = np.pad(crop, ((_PADDING, _PADDING), (_PADDING, _PADDING), (0, 0)), constant_values=255)
    binary = np.pad(np.where(foreground, 0, 255).astype(np.uint8), _PADDING, constant_values=255)
    return tuple(np.repeat(np.repeat(image, _SCALE, axis=0), _SCALE, axis=1) for image in (raw, binary))


def read_pet_title(image, vocabulary: Iterable[str], *, ocr: Callable[[np.ndarray], str]):
    """Return one exact vocabulary name after two agreeing text OCR calls.

    ``image`` is the caller-selected title crop, never an entire tooltip or a
    screenshot to search. Ambiguous normalized vocabulary entries, non-string
    OCR, multiline output, unknown names, and disagreement return ``None``.
    Blank/no-colored evidence makes no OCR calls. An invalid first reading
    stops after one call; engine errors propagate without retries.
    """
    crop = _rgb_crop(image)
    if not callable(ocr):
        raise TypeError("text ocr must be callable")
    if isinstance(vocabulary, (str, bytes)):
        raise TypeError("vocabulary must be an iterable of canonical names")
    candidates = {}
    for name in vocabulary:
        normalized = _normalized_name(name)
        if normalized is None:
            raise ValueError("vocabulary names must contain alphabetic words separated by spaces or hyphens")
        candidates.setdefault(normalized, set()).add(name)
    if not candidates:
        return None
    prepared = _preparations(crop)
    if prepared is None:
        return None
    first = _normalized_name(ocr(prepared[0]))
    names = candidates.get(first, ())
    if len(names) != 1:
        return None
    second = _normalized_name(ocr(prepared[1]))
    return next(iter(names)) if second == first else None


def tesseract_text_ocr(image):
    """Read one prepared text image with one bounded Tesseract call.

    This is separate from numeric OCR and deliberately has no character
    whitelist that could erase invalid punctuation or digits from a title.
    Tesseract's surrounding output whitespace is stripped; internal lines stay.
    """
    crop = np.asarray(image)
    if (crop.dtype != np.uint8 or crop.ndim not in (2, 3) or min(crop.shape[:2], default=0) <= 0
            or (crop.ndim == 3 and crop.shape[2] != 3)):
        raise ValueError("text OCR needs a nonempty uint8 RGB or grayscale image")
    try:
        import pytesseract
    except ImportError as exc:
        raise TooltipOCRUnavailable("pet title OCR requires pytesseract and Pillow in the current Python environment") from exc
    executable = shutil.which("tesseract")
    installed = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if executable is None and installed.is_file():
        executable = str(installed)
    if executable is None:
        raise TooltipOCRUnavailable("pet title OCR requires the Tesseract executable on PATH or in C:\\Program Files\\Tesseract-OCR")
    pytesseract.pytesseract.tesseract_cmd = executable
    try:
        result = pytesseract.image_to_string(crop, config="--psm 7", timeout=TITLE_OCR_TIMEOUT_SECONDS)
    except RuntimeError as exc:
        if str(exc) == "Tesseract process timeout":
            raise TimeoutError(f"pet title Tesseract OCR exceeded {TITLE_OCR_TIMEOUT_SECONDS:g}s timeout") from exc
        raise TooltipOCRError("pet title Tesseract OCR failed") from exc
    except OSError as exc:
        raise TooltipOCRUnavailable("cannot run pet title Tesseract OCR; check its executable installation") from exc
    return result.strip() if isinstance(result, str) else result
