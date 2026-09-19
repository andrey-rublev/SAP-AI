"""Bridge between a trained policy and the real Super Auto Pets client.

Super Auto Pets runs on the **Steam desktop client** (there is no web version),
so the only reliable way to observe and control a live match is screen capture +
OCR + synthetic mouse input.  This module wraps that:

* :func:`capture_screen`  - grab a region of the screen as an image
* :func:`read_text` / :func:`read_number` - OCR helpers (gold, hearts, ...)
* :func:`buy`, :func:`roll`, :func:`end_turn` - high-level in-game actions
* :func:`perform_action` - map an agent's discrete action onto the client

Coordinates and regions are resolution-dependent.  Calibrate :data:`LAYOUT` for
your monitor before relying on the click helpers - run this file directly to
print what the OCR currently reads so you can line things up.

Requires: pyautogui, pytesseract (+ the Tesseract binary), opencv-python, numpy.
The heavy dependencies are imported lazily so the rest of the project still
imports fine on a machine that only wants the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import pyautogui

    pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
    pyautogui.PAUSE = 0.1
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pyautogui = None

try:
    import pytesseract

    # Point pytesseract at the default Windows Tesseract install, if present.
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytesseract = None


@dataclass
class Layout:
    """Pixel regions/points for one screen resolution.

    Regions are ``(left, top, width, height)``; points are ``(x, y)``.
    These defaults are placeholders - calibrate them to your setup.
    """

    gold_region: tuple = (95, 40, 90, 50)
    hearts_region: tuple = (10, 40, 80, 50)
    shop_slots: list = field(
        default_factory=lambda: [(300, 620), (430, 620), (560, 620)]
    )
    team_slots: list = field(
        default_factory=lambda: [
            (700, 380), (820, 380), (940, 380), (1060, 380), (1180, 380)
        ]
    )
    roll_button: tuple = (170, 620)
    end_turn_button: tuple = (1500, 800)


LAYOUT = Layout()


def _require(dep, name: str) -> None:
    if dep is None:
        raise RuntimeError(
            f"{name} is not installed; run `pip install -r requirements.txt`"
        )


# --- perception -------------------------------------------------------------
def capture_screen(region=None) -> np.ndarray:
    """Screenshot the screen (or a sub-region) and return a BGR numpy image."""
    _require(pyautogui, "pyautogui")
    shot = np.array(pyautogui.screenshot(region=region))
    if cv2 is not None:
        return cv2.cvtColor(shot, cv2.COLOR_RGB2BGR)
    return shot[:, :, ::-1]  # RGB -> BGR without OpenCV


def read_text(image) -> str:
    """Run OCR over an image and return the recognised text."""
    _require(pytesseract, "pytesseract")
    return pytesseract.image_to_string(image).strip()


def read_number(region, default: int = 0) -> int:
    """OCR a region and return the first integer found (gold, hearts, ...)."""
    text = read_text(capture_screen(region))
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else default


def read_state() -> dict:
    """Read the parts of the live game state we can OCR reliably."""
    return {
        "gold": read_number(LAYOUT.gold_region),
        "hearts": read_number(LAYOUT.hearts_region),
    }


# --- control ----------------------------------------------------------------
def click(x: int, y: int) -> None:
    _require(pyautogui, "pyautogui")
    pyautogui.moveTo(x, y, duration=0.15)
    pyautogui.click()


def buy(slot: int) -> None:
    """Drag a shop pet onto the board (the SAP "buy" gesture)."""
    _require(pyautogui, "pyautogui")
    sx, sy = LAYOUT.shop_slots[slot]
    tx, ty = LAYOUT.team_slots[slot]
    pyautogui.moveTo(sx, sy, duration=0.15)
    pyautogui.dragTo(tx, ty, duration=0.3, button="left")


def roll() -> None:
    click(*LAYOUT.roll_button)


def end_turn() -> None:
    click(*LAYOUT.end_turn_button)


# Maps the discrete actions from game.SuperAutoPetsEnv onto the live client.
# Action 4 (sell) needs a sell-zone drag that depends on which pet to sell, so it
# is intentionally left out until the board is being tracked.
ACTION_TABLE = {
    0: lambda: buy(0),
    1: lambda: buy(1),
    2: lambda: buy(2),
    3: roll,
    5: end_turn,
}


def perform_action(action: int) -> None:
    """Execute one discrete agent action against the live Steam client."""
    fn = ACTION_TABLE.get(action)
    if fn is None:
        raise ValueError(f"no client mapping for action {action}")
    fn()


if __name__ == "__main__":
    # Calibration helper: show what the OCR currently reads.
    print("Reading live game state (calibrate LAYOUT if these look wrong):")
    print(" ", read_state())
