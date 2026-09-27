"""Experimental, opt-in screen/OCR bridge, not a real SAP game engine.

Real pets have species, health, levels and abilities. Equal OCR stats do not
establish merge compatibility, so live purchases use empty team slots only.
Empty slots must currently produce an explicit zero: unreadable OCR aborts.

No screen dependencies or desktop actions run on import. Create a layout with
python autogui.py --write-template layout.json, edit coordinates and resolution,
and mark calibrated true after verifying them. --layout PATH --inspect reads
without clicking.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

# Loaded lazily, and easily replaced by mocks.
pyautogui = None
pytesseract = None
_CONTROL_ENABLED = False


class PerceptionError(RuntimeError):
    """The screen cannot be read confidently enough to propose an action."""


@dataclass
class Layout:
    """One calibrated screen. Regions are (left, top, width, height)."""

    calibrated: bool = False
    screen_size: tuple = (1920, 1080)
    gold_region: tuple = (95, 40, 90, 50)
    hearts_region: tuple = (10, 40, 80, 50)
    shop_slots: list = field(default_factory=lambda: [(300, 620), (430, 620), (560, 620)])
    team_slots: list = field(default_factory=lambda: [
        (700, 380), (820, 380), (940, 380), (1060, 380), (1180, 380),
    ])
    roll_button: tuple = (170, 620)
    end_turn_button: tuple = (1500, 800)
    sell_zone: tuple = (960, 900)
    shop_stat_regions: list = field(default_factory=lambda: [
        (300, 665, 60, 30), (430, 665, 60, 30), (560, 665, 60, 30),
    ])
    team_stat_regions: list = field(default_factory=lambda: [
        (700, 425, 55, 28), (820, 425, 55, 28), (940, 425, 55, 28),
        (1060, 425, 55, 28), (1180, 425, 55, 28),
    ])

    def validate(self, *, require_calibrated=True, actual_size=None):
        if type(self.calibrated) is not bool:
            raise ValueError("layout calibrated must be a boolean")
        if require_calibrated and not self.calibrated:
            raise ValueError("layout is an uncalibrated template; verify coordinates before live use")

        def coordinates(value, length, name):
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ValueError(f"{name} must contain {length} integer coordinates")
            if any(type(v) is not int for v in value):
                raise ValueError(f"{name} must contain integer coordinates")
            return value

        width, height = coordinates(self.screen_size, 2, "screen_size")
        if width <= 0 or height <= 0:
            raise ValueError("screen_size must be positive")
        if actual_size is not None and tuple(actual_size) != tuple(self.screen_size):
            raise ValueError(f"screen resolution {tuple(actual_size)} differs from calibrated {tuple(self.screen_size)}")

        def point(value, name):
            x, y = coordinates(value, 2, name)
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError(f"{name} lies outside screen_size")

        def region(value, name):
            x, y, w, h = coordinates(value, 4, name)
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError(f"{name} must be a positive region inside screen_size")

        for name in ("roll_button", "end_turn_button", "sell_zone"):
            point(getattr(self, name), name)
        for name in ("gold_region", "hearts_region"):
            region(getattr(self, name), name)
        for name, count, check in (
            ("shop_slots", 3, point), ("team_slots", 5, point),
            ("shop_stat_regions", 3, region), ("team_stat_regions", 5, region),
        ):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or len(values) != count:
                raise ValueError(f"{name} must contain exactly {count} entries")
            for value in values:
                check(value, name)
            if len({tuple(value) for value in values}) != count:
                raise ValueError(f"{name} contains duplicate entries")
        return self

    def save(self, path):
        self.validate(require_calibrated=False)
        Path(path).write_text(json.dumps({"version": 1, **asdict(self)}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.pop("version", None) != 1:
            raise ValueError("unsupported layout format; expected version 1")
        expected = {f.name for f in fields(cls)}
        if set(data) != expected:
            raise ValueError(f"layout fields do not match template; missing={sorted(expected - set(data))}, unknown={sorted(set(data) - expected)}")
        return cls(**data).validate()


LAYOUT = Layout()


def configure_layout(path, *, enable_control=False):
    """Load validated coordinates; explicitly enable mouse control if requested."""
    global LAYOUT, _CONTROL_ENABLED
    _CONTROL_ENABLED = False
    LAYOUT = Layout.load(path)
    _CONTROL_ENABLED = bool(enable_control)
    return LAYOUT


def disable_control():
    """Revoke mouse access after a live session ends or fails."""
    global _CONTROL_ENABLED
    _CONTROL_ENABLED = False


def _dependency(name):
    dep = globals()[name]
    if dep is None:
        try:
            dep = importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"{name} is unavailable; install live dependencies and use a desktop session: {exc}") from exc
        if name == "pyautogui":
            dep.FAILSAFE = True
            dep.PAUSE = 0.1
        elif name == "pytesseract" and not shutil.which("tesseract"):
            candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if candidate.is_file():
                dep.pytesseract.tesseract_cmd = str(candidate)
        globals()[name] = dep
    return dep


def _desktop(*, control=False):
    LAYOUT.validate()
    if control and not _CONTROL_ENABLED:
        raise RuntimeError("mouse control is disabled; configure a calibrated layout with enable_control=True")
    desktop = _dependency("pyautogui")
    LAYOUT.validate(actual_size=desktop.size())
    return desktop


def capture_screen(region=None) -> np.ndarray:
    """Capture RGB pixels only after layout and resolution validation."""
    return np.asarray(_desktop().screenshot(region=region))


def read_text(image) -> str:
    """Run OCR on RGB pixels. Missing Tesseract is a descriptive failure."""
    try:
        return _dependency("pytesseract").image_to_string(image, config="--psm 7").strip()
    except RuntimeError:
        raise
    except Exception as exc:
        raise PerceptionError(f"OCR failed; check the Tesseract installation: {exc}") from exc


def parse_int(text, default: int = 0) -> int:
    """Legacy pure text helper; live perception uses stricter read_number."""
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else default


def _strict_number(text, region):
    text = text.strip()
    if re.fullmatch(r"[0-9]+", text) is None:
        raise PerceptionError(f"unreadable numeric region {region}: {text!r}; no action taken")
    return int(text)


def read_number(region, default=None) -> int:
    """Read one integer. Live calls never treat blank/ambiguous text as empty."""
    text = read_text(capture_screen(region))
    try:
        return _strict_number(text, region)
    except PerceptionError:
        if default is not None:
            return default
        raise


def read_state() -> dict:
    return {"gold": read_number(LAYOUT.gold_region), "hearts": read_number(LAYOUT.hearts_region)}


def _validate_observation(obs):
    value = np.asarray(obs, dtype=float)
    if value.ndim != 1 or value.size not in (10, 12, 13):
        raise ValueError("observation must contain 10, 12 or 13 scalar values")
    if not np.isfinite(value).all() or (value < 0).any() or not np.equal(value, np.floor(value)).all():
        raise ValueError("observation must contain finite nonnegative integers")
    return value


def build_observation(gold, shop, team, turn: int = 0) -> np.ndarray:
    """Legacy [gold, turn, shop(3), team(5)]; no fabricated simulator metadata."""
    shop, team = list(shop)[:3], list(team)[:5]
    shop += [0] * (3 - len(shop))
    team += [0] * (5 - len(team))
    return _validate_observation([gold, turn, *shop, *team]).astype(np.float32)


def read_board(turn: int = 0) -> np.ndarray:
    """Read one snapshot, failing closed on blank or ambiguous OCR.

    Taking one snapshot avoids mixing values from separate animation frames.
    The caller supplies turn because game-phase detection is not implemented.
    """
    pixels = capture_screen()

    def number(region):
        x, y, w, h = region
        return _strict_number(read_text(pixels[y:y + h, x:x + w]), region)

    gold = number(LAYOUT.gold_region)
    shop = [number(r) for r in LAYOUT.shop_stat_regions]
    team = [number(r) for r in LAYOUT.team_stat_regions]
    if gold > 99 or any(value > 50 for value in shop + team):
        raise PerceptionError("OCR values exceed the supported scalar bridge limits")
    return build_observation(gold, shop, team, turn)


def _buy_target_slot(slot, obs, *, allow_scalar_merge=False):
    """Choose a destination; shop index and team index are unrelated.

    allow_scalar_merge reproduces ONLY the toy simulator's equality rule.
    Real pet merging requires species-aware perception, which is absent.
    """
    value = _validate_observation(obs)
    if type(slot) is not int or slot not in (0, 1, 2) or value[2 + slot] <= 0:
        raise ValueError("buy requires a nonempty shop slot 0, 1 or 2")
    team = value[5:10]
    if allow_scalar_merge:
        matches = np.flatnonzero(team == value[2 + slot])
        if matches.size:
            return int(matches[0])
    empty = np.flatnonzero(team == 0)
    if empty.size:
        return int(empty[0])
    raise ValueError("no empty team slot; scalar stats cannot establish real pet merge compatibility")


def _weakest_team_slot(obs):
    if obs is None:
        return None
    team = _validate_observation(obs)[5:10]
    occupied = np.flatnonzero(team > 0)
    return int(occupied[np.argmin(team[occupied])]) if occupied.size else None


def live_action_mask(obs):
    """Conservative scalar mask: real pet merges are unsupported."""
    from game import SuperAutoPetsEnv

    value = _validate_observation(obs)
    mask = SuperAutoPetsEnv.mask_from_obs(value).copy()
    if not (value[5:10] == 0).any():
        mask[:3] = False
    return mask


def click(x: int, y: int) -> None:
    desktop = _desktop(control=True)
    width, height = LAYOUT.screen_size
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError("click is outside the calibrated screen")
    desktop.moveTo(x, y, duration=0.15)
    desktop.click()


def buy(slot: int, team_slot: int) -> None:
    if slot not in range(3) or team_slot not in range(5):
        raise ValueError("invalid buy source or destination slot")
    desktop = _desktop(control=True)
    desktop.moveTo(*LAYOUT.shop_slots[slot], duration=0.15)
    desktop.dragTo(*LAYOUT.team_slots[team_slot], duration=0.3, button="left")


def roll() -> None:
    click(*LAYOUT.roll_button)


def sell(slot: int) -> None:
    if slot not in range(5):
        raise ValueError("invalid sell slot")
    desktop = _desktop(control=True)
    desktop.moveTo(*LAYOUT.team_slots[slot], duration=0.15)
    desktop.dragTo(*LAYOUT.sell_zone, duration=0.3, button="left")


def end_turn() -> None:
    click(*LAYOUT.end_turn_button)


def perform_action(action: int, obs=None) -> None:
    """Execute a legal observed action only with explicitly enabled control."""
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or action not in range(6):
        raise ValueError(f"no client mapping for action {action}")
    if obs is None:
        raise ValueError("all live actions require a current board observation")
    if not live_action_mask(obs)[action]:
        raise ValueError(f"action {action} is not supported for the observed board")
    _desktop(control=True)
    action = int(action)
    if action < 3:
        buy(action, _buy_target_slot(action, obs))
    elif action == 3:
        roll()
    elif action == 4:
        sell(_weakest_team_slot(obs))
    else:
        end_turn()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-template", metavar="PATH", help="write placeholder JSON without reading the screen")
    parser.add_argument("--layout", metavar="PATH", help="calibrated JSON layout")
    parser.add_argument("--inspect", action="store_true", help="explicitly read the screen; never click")
    args = parser.parse_args(argv)
    if args.write_template:
        path = Path(args.write_template)
        if path.exists():
            parser.error(f"refusing to overwrite existing layout {path}")
        Layout().save(path)
        print(f"Wrote uncalibrated template: {path}")
    elif args.inspect:
        if not args.layout:
            parser.error("--inspect requires --layout PATH")
        configure_layout(args.layout)
        print(read_state())
        print(read_board())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
