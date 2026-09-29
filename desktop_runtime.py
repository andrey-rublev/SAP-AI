"""Window-bound desktop IO. Imported without capturing or controlling anything."""
from __future__ import annotations

import ctypes
from collections import OrderedDict
from ctypes import wintypes
import os
from pathlib import Path
import shutil

import numpy as np

from desktop_state import Action, Phase, legal_action


OCR_TIMEOUT_SECONDS = 3.0


class CachedOCR:
    """Bounded, instance-local raw OCR cache for exactly identical RGB crops.

    Numeric validation remains the perceptor's responsibility on every frame.
    Errors and non-string results are never cached, so a failed engine can be
    retried without retaining an invented reading.
    """

    def __init__(self, ocr, *, max_entries=128):
        if not callable(ocr):
            raise TypeError("ocr must be callable")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.ocr, self.max_entries = ocr, max_entries
        self._cache = OrderedDict()

    def __call__(self, image):
        crop = np.asarray(image)
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("OCR cache requires an RGB crop")
        key = (crop.dtype.str, crop.shape, crop.tobytes())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self.ocr(crop)
        if isinstance(result, str):
            self._cache[key] = result
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return result


class DesktopUnavailable(RuntimeError):
    pass


class WindowsGameWindow:
    """Capture/control one foreground game client in physical screen pixels.

    Never activates a window or clicks through another foreground application.
    All stored profile coordinates are relative to the client, not the monitor.
    """

    def __init__(self, title="Super Auto Pets"):
        if os.name != "nt":
            raise DesktopUnavailable("live window control currently requires Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        # Make coordinates consistent on high-DPI monitors before capturing.
        try:
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            self.user32.SetProcessDPIAware()
        self.user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        self.user32.FindWindowW.restype = wintypes.HWND
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.title = title
        self.handle = self.user32.FindWindowW(None, title)
        if not self.handle:
            raise DesktopUnavailable(f"open the {title!r} game window first")
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        self.mouse = pyautogui

    def geometry(self):
        if not self.user32.IsWindow(self.handle) or self.user32.IsIconic(self.handle):
            raise DesktopUnavailable("game window was closed or minimized")
        if self.user32.GetForegroundWindow() != self.handle:
            raise DesktopUnavailable("game lost focus; no desktop input was sent")
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.user32.GetClientRect(self.handle, ctypes.byref(rect)):
            raise DesktopUnavailable("cannot read game client bounds")
        if not self.user32.ClientToScreen(self.handle, ctypes.byref(origin)):
            raise DesktopUnavailable("cannot locate game client on the screen")
        return origin.x, origin.y, rect.right, rect.bottom

    def capture(self):
        from PIL import ImageGrab
        x, y, width, height = self.geometry()
        frame = np.asarray(ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True).convert("RGB"))
        # Focus/geometry might change while the capture is in flight.
        if self.geometry() != (x, y, width, height):
            raise DesktopUnavailable("game moved during capture; retry from a fresh frame")
        return frame

    def _screen_point(self, point, expected_size):
        x, y, width, height = self.geometry()
        if (width, height) != tuple(expected_size):
            raise DesktopUnavailable("game size differs from the calibrated profile")
        px, py = point
        if not 0 <= px < width or not 0 <= py < height:
            raise ValueError("input point is outside the game client")
        return x + px, y + py

    def click(self, point, expected_size):
        target = self._screen_point(point, expected_size)
        self.mouse.moveTo(*target, duration=0.15)
        if self._screen_point(point, expected_size) != target:
            raise DesktopUnavailable("game moved before click")
        self.mouse.click(*target)

    def drag(self, source, target, expected_size):
        start = self._screen_point(source, expected_size)
        end = self._screen_point(target, expected_size)
        self.mouse.moveTo(*start, duration=0.15)
        if self._screen_point(source, expected_size) != start:
            raise DesktopUnavailable("game moved before drag")
        try:
            self.mouse.mouseDown(*start, button="left")
            if self._screen_point(target, expected_size) != end:
                raise DesktopUnavailable("game moved during drag")
            self.mouse.moveTo(*end, duration=0.35)
        finally:
            # A corner abort also prevents ordinary mouseUp. Suppress that check
            # only for release, then restore it even if the release raises.
            failsafe = self.mouse.FAILSAFE
            try:
                self.mouse.FAILSAFE = False
                self.mouse.mouseUp(button="left", _pause=False)
            finally:
                self.mouse.FAILSAFE = failsafe


def prepare_numeric_crop(image):
    """Isolate complete numeric glyphs from contrasting HUD/stat backgrounds.

    Tight calibration must leave a margin around each glyph. Border-connected
    shapes are removed; a substantial clipped glyph rejects the entire crop so
    a truncated two-digit value cannot silently become its remaining digit.
    This is a deterministic image transform, not a correction of OCR answers.
    """
    from PIL import Image, ImageOps

    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("numeric OCR requires a nonempty RGB uint8 crop")
    gray = np.asarray(ImageOps.grayscale(Image.fromarray(image)))
    mask = gray < 90 if np.median(gray) > 170 else gray > 180
    height, width = mask.shape
    visited = np.zeros_like(mask)
    foreground = np.zeros_like(mask)
    for y, x in zip(*np.where(mask)):
        if visited[y, x]:
            continue
        pending, component = [(y, x)], []
        visited[y, x] = True
        touches_edge = False
        while pending:
            cy, cx = pending.pop()
            component.append((cy, cx))
            touches_edge |= cx in (0, width - 1) or cy in (0, height - 1)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        pending.append((ny, nx))
        ys, xs = zip(*component)
        tall = max(ys) - min(ys) >= height * 0.3
        # A glyph can join a full-width frame line when clipped. Its width
        # cannot safely distinguish that combined shape from a UI border.
        if touches_edge and tall:
            return None
        if not touches_edge and tall and len(component) >= height * width * 0.015:
            foreground[ys, xs] = True
    if not foreground.any():
        return None
    ys, xs = np.where(foreground)
    foreground = foreground[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = Image.fromarray(np.where(foreground, 0, 255).astype(np.uint8))
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
    return ImageOps.expand(crop, 36, 255)


def tesseract_ocr(image):
    """Read isolated glyphs once; blank/clipped crops stay unknown."""
    crop = prepare_numeric_crop(image)
    if crop is None:
        return ""
    import pytesseract

    if not shutil.which("tesseract"):
        executable = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if executable.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(executable)
    try:
        return pytesseract.image_to_string(
            crop, config="--psm 13 -c tessedit_char_whitelist=0123456789",
            timeout=OCR_TIMEOUT_SECONDS,
        ).strip()
    except RuntimeError as exc:
        # Pytesseract terminates its child process, then raises an untyped
        # RuntimeError. Distinguish that timeout from an unreadable OCR result.
        if str(exc) == "Tesseract process timeout":
            raise TimeoutError(f"Tesseract OCR exceeded {OCR_TIMEOUT_SECONDS:g}s timeout") from exc
        raise


class DesktopRuntime:
    """Connect calibrated perception and a window, with a last-moment recheck."""

    def __init__(self, profile, perceptor, window, *, execute=False):
        profile.validate(require_calibrated=True)
        self.profile, self.perceptor, self.window = profile, perceptor, window
        self.execute = execute
        self.last_board = None
        self.last_frame = None

    def phase_actions(self):
        """Enable only transitions with both a recognized phase and all points."""
        requirements = {
            Phase.NAMING: ("choose_name", ("name_adjective", "name_noun")),
            Phase.NAMING_READY: ("confirm_name", ("confirm_name",)),
            Phase.ROUND_RESULT: ("continue_round", ("continue_round",)),
            Phase.TIER_UNLOCK: ("dismiss_tier", ("dismiss_tier",)),
        }
        recognized = {item.phase for item in self.profile.phase_templates}
        return {phase: Action(kind) for phase, (kind, points) in requirements.items()
                if phase in recognized and all(point in self.profile.buttons for point in points)}

    def observe(self):
        self.last_frame = self.window.capture()
        self.last_board = self.perceptor.observe(self.last_frame)
        return self.last_board

    def act(self, action):
        if not self.execute:
            raise DesktopUnavailable("desktop input is disabled in preview mode")
        before = self.last_board
        if before is None or not legal_action(before, action):
            raise ValueError("action is not legal for the last observed board")
        self.last_frame = self.window.capture()
        current = self.perceptor.observe(self.last_frame)
        if current.fingerprint() != before.fingerprint():
            raise DesktopUnavailable("board changed before input; observe again")
        size = self.profile.image_size
        if action.kind in ("buy", "merge"):
            # The desktop client selects a shop pet, then places it with a
            # second click. Each click rechecks focus and client geometry;
            # any failure propagates without retrying or selecting again.
            self.window.click(self.profile.shop[action.slot].portrait.center, size)
            self.window.click(self.profile.team[action.target].portrait.center, size)
        elif action.kind == "choose_name":
            # Validate the whole sequence before selecting its first option.
            names = ("name_adjective", "name_noun")
            if any(name not in self.profile.buttons for name in names):
                raise ValueError("profile needs both calibrated name options")
            for name in names:
                self.window.click(self.profile.buttons[name], size)
        elif action.kind == "sell":
            if "sell" not in self.profile.buttons:
                raise ValueError("profile has no calibrated sell point")
            # Selecting a teammate reveals the client's Sell button. Validate
            # that point before selection; each click checks focus/geometry.
            self.window.click(self.profile.team[action.slot].portrait.center, size)
            self.window.click(self.profile.buttons["sell"], size)
        else:
            if action.kind not in self.profile.buttons:
                raise ValueError(f"profile has no calibrated {action.kind} button")
            self.window.click(self.profile.buttons[action.kind], size)
