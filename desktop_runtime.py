"""Window-bound desktop IO. Imported without capturing or controlling anything."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil

import numpy as np

from desktop_state import legal_action


OCR_TIMEOUT_SECONDS = 3.0


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


def tesseract_ocr(image):
    """Read a numeric crop with a bounded subprocess; timeouts stop perception."""
    import pytesseract
    from PIL import Image, ImageOps

    if not shutil.which("tesseract"):
        executable = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if executable.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(executable)
    crop = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    crop = crop.resize((crop.width * 3, crop.height * 3))
    crop = ImageOps.grayscale(crop)
    try:
        return pytesseract.image_to_string(
            crop, config="--psm 7 -c tessedit_char_whitelist=0123456789",
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

    def observe(self):
        self.last_board = self.perceptor.observe(self.window.capture())
        return self.last_board

    def act(self, action):
        if not self.execute:
            raise DesktopUnavailable("desktop input is disabled in preview mode")
        before = self.last_board
        if before is None or not legal_action(before, action):
            raise ValueError("action is not legal for the last observed board")
        current = self.perceptor.observe(self.window.capture())
        if current.fingerprint() != before.fingerprint():
            raise DesktopUnavailable("board changed before input; observe again")
        size = self.profile.image_size
        if action.kind in ("buy", "merge"):
            # The desktop client selects a shop pet, then places it with a
            # second click. Each click rechecks focus and client geometry;
            # any failure propagates without retrying or selecting again.
            self.window.click(self.profile.shop[action.slot].portrait.center, size)
            self.window.click(self.profile.team[action.target].portrait.center, size)
        elif action.kind == "sell":
            if "sell" not in self.profile.buttons:
                raise ValueError("profile has no calibrated sell point")
            self.window.drag(self.profile.team[action.slot].portrait.center,
                             self.profile.buttons["sell"], size)
        else:
            if action.kind not in self.profile.buttons:
                raise ValueError(f"profile has no calibrated {action.kind} button")
            self.window.click(self.profile.buttons[action.kind], size)
