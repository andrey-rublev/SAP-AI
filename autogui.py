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
    sell_zone: tuple = (960, 900)  # drag a team pet here to sell it


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


def sell(slot: int) -> None:
    """Sell the team pet in ``slot`` by dragging it to the sell zone."""
    _require(pyautogui, "pyautogui")
    tx, ty = LAYOUT.team_slots[slot]
    zx, zy = LAYOUT.sell_zone
    pyautogui.moveTo(tx, ty, duration=0.15)
    pyautogui.dragTo(zx, zy, duration=0.3, button="left")


def end_turn() -> None:
    click(*LAYOUT.end_turn_button)


def _weakest_team_slot(obs):
    """Index of the weakest (lowest-strength) occupied team slot, or None."""
    if obs is None:
        return None
    team = np.asarray(obs)[5:10]  # team slots occupy obs[5:10]
    occupied = np.where(team > 0)[0]
    if occupied.size == 0:
        return None
    return int(occupied[int(np.argmin(team[occupied]))])


def perform_action(action: int, obs=None) -> None:
    """Execute one discrete agent action against the live Steam client.

    ``obs`` is the current observation (see :class:`game.SuperAutoPetsEnv`); it is
    only needed for the sell action, which must know which team slot is weakest.
    """
    if action in (0, 1, 2):
        buy(action)
    elif action == 3:
        roll()
    elif action == 4:
        slot = _weakest_team_slot(obs)
        if slot is None:
            raise ValueError("sell action needs a team observation to pick a slot")
        sell(slot)
    elif action == 5:
        end_turn()
    else:
        raise ValueError(f"no client mapping for action {action}")


if __name__ == "__main__":
    # Calibration helper: show what the OCR currently reads.
    print("Reading live game state (calibrate LAYOUT if these look wrong):")
    print(" ", read_state())
