"""Window-bound desktop IO. Imported without capturing or controlling anything."""
from __future__ import annotations

import ctypes
from collections import OrderedDict
from ctypes import wintypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from desktop_state import Action, BoardChangedBeforeInput, Phase, legal_action
from desktop_vision import Rect


OCR_TIMEOUT_SECONDS = 3.0
DEPENDENCY_TIMEOUT_SECONDS = 5.0
VK_ESCAPE = 0x1B


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


class GameWindowNotFound(DesktopUnavailable):
    """No game client exists yet; other window failures must not relaunch it."""

    def __init__(self, message, *, stop_requested=None):
        super().__init__(message)
        self.stop_requested = stop_requested


def _tesseract_executable():
    executable = shutil.which("tesseract")
    if executable:
        return executable
    installed = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return str(installed) if installed.is_file() else None


def validate_live_dependencies():
    """Check optional packages and the OCR executable without desktop IO.

    Finding package specifications deliberately avoids importing pyautogui,
    whose import can initialize native desktop access. The only child process
    is a bounded, hidden Tesseract version query.
    """
    missing = []
    for module, package in (("PIL", "Pillow"), ("pyautogui", "pyautogui"), ("pytesseract", "pytesseract")):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    if missing:
        raise DesktopUnavailable(
            f"missing live Python dependencies: {', '.join(missing)}; "
            'use .\\.venv\\Scripts\\python.exe -m pip install -e ".[live]" from the project folder'
        )
    executable = _tesseract_executable()
    if executable is None:
        raise DesktopUnavailable("install Tesseract OCR and add it to PATH or use C:\\Program Files\\Tesseract-OCR")
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            timeout=DEPENDENCY_TIMEOUT_SECONDS, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise DesktopUnavailable(f"Tesseract dependency check exceeded {DEPENDENCY_TIMEOUT_SECONDS:g}s") from exc
    except OSError as exc:
        raise DesktopUnavailable("cannot run Tesseract OCR; check its installation and executable") from exc
    if result.returncode != 0 or not result.stdout.lstrip().lower().startswith("tesseract "):
        raise DesktopUnavailable("Tesseract OCR version check failed; check its installation and executable")
    return executable


def _require_running(should_stop):
    if should_stop is not None and should_stop():
        raise DesktopUnavailable("stop requested or runtime deadline reached; no further input sent")


class WindowsGameWindow:
    """Capture/control one foreground game client in physical screen pixels.

    Startup activation is explicit; capture/input never reactivate the client
    or click through another foreground application.
    All stored profile coordinates are relative to the client, not the monitor.
    """

    should_stop = None

    def __init__(self, title="Super Auto Pets"):
        if os.name != "nt":
            raise DesktopUnavailable("live window control currently requires Windows")
        self._escape_stopped = False
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        # Make coordinates consistent on high-DPI monitors before capturing.
        try:
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            self.user32.SetProcessDPIAware()
        self.user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        self.user32.FindWindowW.restype = wintypes.HWND
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        self.user32.SetForegroundWindow.restype = wintypes.BOOL
        self.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.user32.PostMessageW.restype = wintypes.BOOL
        self.user32.GetAsyncKeyState.argtypes = [wintypes.INT]
        self.user32.GetAsyncKeyState.restype = wintypes.SHORT
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.title = title
        self.handle = self.user32.FindWindowW(None, title)
        if not self.handle:
            raise GameWindowNotFound(f"open the {title!r} game window first",
                                     stop_requested=self.stop_requested)
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        self.mouse = pyautogui

    def stop_requested(self):
        """Latch a held Escape key without hooks, logging, or sending input.

        The high bit reports the current key state independently of the
        foreground application's message queue. A brief tap between polls can
        be missed; hold Escape until the session stops. In-flight OCR finishes
        before the session can check this predicate again.
        """
        if not self._escape_stopped:
            self._escape_stopped = bool(self.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)
        return self._escape_stopped

    def ready_for_profile(self, expected_size):
        """Read startup client geometry without activation or capture.

        Unity can expose its final title before applying its saved fullscreen
        size. A newly launched client must reach calibration before startup
        activation; existing clients still use activate's immediate checks.
        """
        if not self.user32.IsWindow(self.handle) or self.user32.IsIconic(self.handle):
            return False
        rect = wintypes.RECT()
        return bool(self.user32.GetClientRect(self.handle, ctypes.byref(rect))
                    and (rect.right, rect.bottom) == tuple(expected_size))

    def activate(self, expected_size):
        """Request foreground ownership once, before the session starts.

        A minimized client must be restored by the user first. Its restored
        dimensions cannot be checked before changing the desktop, so startup
        only accepts a visible client with the calibrated size. Windows may
        deny the focus request; actual foreground ownership is always checked.
        """
        _require_running(self.should_stop)
        if not self.user32.IsWindow(self.handle):
            raise DesktopUnavailable("game window was closed before startup")
        if self.user32.IsIconic(self.handle):
            raise DesktopUnavailable("restore the minimized game window before startup")
        rect = wintypes.RECT()
        if not self.user32.GetClientRect(self.handle, ctypes.byref(rect)):
            raise DesktopUnavailable("cannot read game client bounds")
        expected_size = tuple(expected_size)
        if (rect.right, rect.bottom) != expected_size:
            raise DesktopUnavailable("game size differs from the calibrated profile")
        if self.user32.GetForegroundWindow() != self.handle:
            _require_running(self.should_stop)
            self.user32.SetForegroundWindow(self.handle)
        _require_running(self.should_stop)
        geometry = self.geometry()
        if geometry[2:] != expected_size:
            raise DesktopUnavailable("game size changed during startup activation")
        return geometry

    def close_for_restart(self, expected_size, *, timeout=90., clock=time.monotonic, sleep=time.sleep):
        """Request normal application shutdown once, then verify disappearance.

        The caller must first verify a completed terminal result. This method
        never kills a process, retries a close, changes focus, or sends keys.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("close timeout must be finite and positive")
        deadline = clock() + timeout
        if self.should_stop is not None and self.should_stop():
            return False
        if self.geometry()[2:] != tuple(expected_size):
            raise DesktopUnavailable("game size differs from the calibrated profile; no close requested")
        if self.should_stop is not None and self.should_stop():
            return False
        if clock() >= deadline:
            raise DesktopUnavailable("timed out before requesting normal game shutdown")
        if self.user32.GetForegroundWindow() != self.handle:
            raise DesktopUnavailable("game lost focus before shutdown; no close requested")
        if not self.user32.PostMessageW(self.handle, 0x0010, 0, 0):  # WM_CLOSE
            raise DesktopUnavailable("game rejected normal shutdown; no restart attempted")
        while True:
            if self.should_stop is not None and self.should_stop():
                return False
            if clock() >= deadline:
                raise DesktopUnavailable("timed out waiting for game shutdown; no restart attempted")
            if not self.user32.IsWindow(self.handle):
                return True
            if self.user32.GetForegroundWindow() != self.handle:
                # Closing can invalidate the handle between these two reads.
                if not self.user32.IsWindow(self.handle):
                    continue
                raise DesktopUnavailable("game lost focus during shutdown; no restart attempted")
            sleep(min(.1, max(0., deadline - clock())))

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
        _require_running(self.should_stop)
        self.mouse.moveTo(*target, duration=0.15)
        if self._screen_point(point, expected_size) != target:
            raise DesktopUnavailable("game moved before click")
        _require_running(self.should_stop)
        self.mouse.click(*target)

    def hover(self, point, expected_size):
        """Move within the calibrated foreground client without pressing buttons."""
        _require_running(self.should_stop)
        target = self._screen_point(point, expected_size)
        _require_running(self.should_stop)
        self.mouse.moveTo(*target, duration=0.15)
        if self._screen_point(point, expected_size) != target:
            raise DesktopUnavailable("game moved during hover; no further input sent")
        _require_running(self.should_stop)

    def drag(self, source, target, expected_size):
        start = self._screen_point(source, expected_size)
        end = self._screen_point(target, expected_size)
        _require_running(self.should_stop)
        self.mouse.moveTo(*start, duration=0.15)
        if self._screen_point(source, expected_size) != start:
            raise DesktopUnavailable("game moved before drag")
        try:
            _require_running(self.should_stop)
            self.mouse.mouseDown(*start, button="left")
            if self._screen_point(target, expected_size) != end:
                raise DesktopUnavailable("game moved during drag")
            _require_running(self.should_stop)
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
    try:
        import pytesseract
    except ImportError as exc:
        raise DesktopUnavailable("Tesseract OCR requires pytesseract in the current Python environment") from exc

    executable = _tesseract_executable()
    if executable is None:
        raise DesktopUnavailable("install Tesseract OCR and add it to PATH or use C:\\Program Files\\Tesseract-OCR")
    pytesseract.pytesseract.tesseract_cmd = executable
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

    def __init__(self, profile, perceptor, window, *, execute=False, should_stop=None, identifier=None):
        profile.validate(require_calibrated=True)
        self.profile, self.perceptor, self.window = profile, perceptor, window
        self.execute = execute
        self.should_stop = should_stop
        if isinstance(window, WindowsGameWindow) and should_stop is not None:
            window.should_stop = should_stop
        # Optional desktop_identify.SpeciesIdentifier; hovering needs a park point.
        if identifier is not None and getattr(profile, "pointer_rest", None) is None:
            raise ValueError("species identification needs a calibrated pointer_rest")
        self.identifier = identifier
        self.last_board = None
        self.last_frame = None
        self._pointer_dirty = True
        self._last_raw_fingerprint = None

    def phase_actions(self, *, start_arena=False):
        """Enable calibrated transitions; entering a new arena is opt-in."""
        requirements = {
            Phase.NAMING: ("choose_name", ("name_adjective", "name_noun")),
            Phase.NAMING_READY: ("confirm_name", ("confirm_name",)),
            Phase.ROUND_RESULT: ("continue_round", ("continue_round",)),
            Phase.TIER_UNLOCK: ("dismiss_tier", ("dismiss_tier",)),
            Phase.LIFE_REWARD: ("dismiss_life_reward", ("dismiss_life_reward",)),
            Phase.END_TURN_CONFIRM: ("confirm_end_turn", ("confirm_end_turn",)),
        }
        if start_arena:
            requirements.update({
                Phase.MAIN_MENU: ("open_play", ("open_play",)),
                Phase.PLAY_MENU: ("open_arena", ("open_arena",)),
                Phase.ARENA_SETUP: ("start_arena", ("start_arena",)),
            })
        recognized = {item.phase for item in self.profile.phase_templates}
        return {phase: Action(kind) for phase, (kind, points) in requirements.items()
                if phase in recognized and all(point in self.profile.buttons for point in points)}

    def observe(self):
        rest = getattr(self.profile, "pointer_rest", None)
        if self.execute and rest is not None and self._pointer_dirty:
            _require_running(self.should_stop)
            self.window.hover(rest, self.profile.image_size)
            _require_running(self.should_stop)
            self._pointer_dirty = False
        self.last_frame = self.window.capture()
        board = self.perceptor.observe(self.last_frame)
        raw = board.fingerprint()
        stable, self._last_raw_fingerprint = raw == self._last_raw_fingerprint, raw
        if self.identifier is not None:
            board = self.identifier.annotate(board)
            # Hover only a settled shop, so a tooltip binds to the pet it names.
            if self.execute and stable and board.phase == Phase.SHOP:
                board = self._identify(board)
        self.last_board = board
        return self.last_board

    def _identify(self, board):
        size, rest = self.profile.image_size, self.profile.pointer_rest
        points = {(row, index): slot.portrait.center
                  for row in ("shop", "team") for index, slot in enumerate(getattr(self.profile, row))}

        def hover(point):
            _require_running(self.should_stop)
            self._pointer_dirty = True
            self.window.hover(point, size)

        def park():
            self.window.hover(rest, size)
            self.identifier.sleep(.15)
            self._pointer_dirty = False

        return self.identifier.identify(board, self.last_frame, points=points, hover=hover,
                                        capture=self.window.capture, park=park)

    def act(self, action):
        if not self.execute:
            raise DesktopUnavailable("desktop input is disabled in preview mode")
        _require_running(self.should_stop)
        before = self.last_board
        if before is None or not legal_action(before, action):
            raise ValueError("action is not legal for the last observed board")
        self.last_frame = self.window.capture()
        current = self.perceptor.observe(self.last_frame)
        if self.identifier is not None:
            current = self.identifier.annotate(current)
        _require_running(self.should_stop)
        if current.fingerprint() != before.fingerprint():
            # This exception is reserved for a proven no-input rejection.
            # Capture/OCR, focus and partial-click failures still fail closed.
            raise BoardChangedBeforeInput("board changed before input; observe again",
                                          preflight_board=current)
        size = self.profile.image_size
        def click(point):
            _require_running(self.should_stop)
            self._pointer_dirty = True
            self.window.click(point, size)

        def portrait_point(slots, index, missing_message, kind):
            if not isinstance(slots, (list, tuple)) or index >= len(slots):
                raise ValueError(missing_message)
            region = getattr(slots[index], "portrait", None)
            if not isinstance(region, Rect):
                raise ValueError(f"{kind} input needs calibrated Rect regions")
            region.validate(size)
            return region.center

        if action.kind in ("buy", "merge"):
            missing = "profile needs calibrated shop source and team target"
            points = (portrait_point(getattr(self.profile, "shop", ()), action.slot, missing, "pet"),
                      portrait_point(getattr(self.profile, "team", ()), action.target, missing, "pet"))
        elif action.kind == "buy_food":
            missing = "profile needs calibrated food source and team target"
            points = (portrait_point(getattr(self.profile, "food", ()), action.slot, missing, "food"),
                      portrait_point(getattr(self.profile, "team", ()), action.target, missing, "food"))
        elif action.kind == "choose_name":
            names = ("name_adjective", "name_noun")
            if any(name not in self.profile.buttons for name in names):
                raise ValueError("profile needs both calibrated name options")
            points = tuple(self.profile.buttons[name] for name in names)
        elif action.kind == "sell":
            if "sell" not in self.profile.buttons:
                raise ValueError("profile has no calibrated sell point")
            points = (portrait_point(getattr(self.profile, "team", ()), action.slot,
                                     "profile needs a calibrated team source", "sell"),
                      self.profile.buttons["sell"])
        else:
            if action.kind not in self.profile.buttons:
                raise ValueError(f"profile has no calibrated {action.kind} button")
            points = (self.profile.buttons[action.kind],)
        # Freeze and validate every destination before selecting anything.
        # A later lookup, malformed point, or changed mutable button must not
        # strand the client after the first click of a partial sequence.
        plan = []
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(type(value) is not int for value in point)):
                raise ValueError("input point must contain two integer client coordinates")
            x, y = point
            if not 0 <= x < size[0] or not 0 <= y < size[1]:
                raise ValueError("input point is outside the game client")
            plan.append((x, y))
        # Each native click retains focus/geometry and stop checks. Any partial
        # input failure propagates; the pending session action is not retried.
        try:
            for point in plan:
                click(point)
        finally:
            # Even a partial sequence may have moved pets; re-identify them.
            if self.identifier is not None:
                self.identifier.forget(action)
