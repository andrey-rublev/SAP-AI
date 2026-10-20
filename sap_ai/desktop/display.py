"""Choose and apply a game display mode the calibrated profile can read.

The profile is calibrated at one size (2560x1440). When the screen can host
that exactly, the game runs as it is. Otherwise the game is switched to a
window with the profile's aspect ratio that fits the screen, and the profile is
scaled to match (see :mod:`sap_ai.desktop.scaling`). Unity reads these
preferences at startup and writes them back on exit, so they are only changed
while the game is closed. The original values are backed up once.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from sap_ai.desktop.scaling import fit_size

GAME_KEY = r"Software\Team Wood\Super Auto Pets"
WIDTH = "Screenmanager Resolution Width_h182942802"
HEIGHT = "Screenmanager Resolution Height_h2627697771"
NATIVE = "Screenmanager Resolution Use Native_h1405027254"
MODE = "Screenmanager Fullscreen mode_h3630240806"
WINDOWED = 3  # UnityEngine.FullScreenMode.Windowed
# Title bar and borders of a windowed client, which must also fit the work area.
WINDOW_CHROME = (16, 40)


def _user32():
    user32 = ctypes.WinDLL("user32")
    try:  # physical pixels, matching captures and profile coordinates
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        user32.SetProcessDPIAware()
    return user32


def work_area() -> tuple[int, int]:
    """Usable desktop size in physical pixels (screen minus the taskbar)."""
    user32 = _user32()
    rect = (ctypes.c_long * 4)()
    if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
        return rect[2] - rect[0], rect[3] - rect[1]
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def screen_size() -> tuple[int, int]:
    user32 = _user32()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def choose_client_size(profile_size, screen, area) -> tuple[tuple[int, int], bool]:
    """Return (client size, windowed) for this desktop.

    Fullscreen at the calibrated size when the screen is exactly that size;
    otherwise the largest same-aspect window that fits the work area.
    """
    profile_size = tuple(profile_size)
    if tuple(screen) == profile_size:
        return profile_size, False
    bounds = (area[0] - WINDOW_CHROME[0], area[1] - WINDOW_CHROME[1])
    return fit_size(profile_size, bounds), True


def read_prefs() -> dict:
    import winreg
    values = {}
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, GAME_KEY) as key:
        for name in (WIDTH, HEIGHT, NATIVE, MODE):
            try:
                values[name] = winreg.QueryValueEx(key, name)[0]
            except FileNotFoundError:
                pass
    return values


def write_prefs(values: dict) -> None:
    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, GAME_KEY) as key:
        for name, value in values.items():
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, int(value))


def apply_windowed(size, backup_path) -> bool:
    """Set windowed ``size`` for the next launch; returns True if anything changed."""
    if os.name != "nt":
        return False
    current = read_prefs()
    wanted = {WIDTH: size[0], HEIGHT: size[1], NATIVE: 0, MODE: WINDOWED}
    if all(current.get(name) == value for name, value in wanted.items()):
        return False
    backup = Path(backup_path)
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    write_prefs(wanted)
    return True


def restore(backup_path) -> bool:
    """Put back the display preferences saved before the first change."""
    backup = Path(backup_path)
    if not backup.exists():
        return False
    write_prefs(json.loads(backup.read_text(encoding="utf-8")))
    return True
