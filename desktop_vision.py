"""Calibrated, read-only perception of a single game-client RGB screenshot.

All coordinates are pixels relative to the exact captured client image. There
are deliberately no built-in game coordinates, species images, or OCR guesses.
User-created template images stay beside the local JSON calibration profile.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from desktop_state import Board, PetSlot, Phase


def _coordinates(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count or any(type(v) is not int for v in value):
        raise ValueError(f"{name} must contain {count} integer coordinates")
    return tuple(value)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_list(cls, value):
        return cls(*_coordinates(value, 4, "region"))

    def to_list(self):
        return [self.x, self.y, self.width, self.height]

    def validate(self, size):
        _coordinates(self.to_list(), 4, "region")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0 or self.x + self.width > size[0] or self.y + self.height > size[1]:
            raise ValueError("region must have positive dimensions and lie inside image_size")

    @property
    def center(self):
        return self.x + self.width // 2, self.y + self.height // 2

    def crop(self, image):
        self.validate((image.shape[1], image.shape[0]))
        return image[self.y:self.y + self.height, self.x:self.x + self.width].copy()


def _thresholds(max_distance, margin):
    for name, value in (("max_distance", max_distance), ("margin", margin)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number between 0 and 1")


@dataclass(frozen=True)
class PhaseTemplate:
    phase: Phase
    region: Rect
    template: str
    max_distance: float = 0.06
    margin: float = 0.015
    match_mode: str = "rgb"
    white_text: dict[str, int | float] | None = None

    def matching_options(self):
        if type(self.match_mode) is not str or self.match_mode not in {"rgb", "white_text"}:
            raise ValueError("phase match_mode must be 'rgb' or 'white_text'")
        if self.match_mode == "rgb":
            if self.white_text is not None:
                raise ValueError("white_text options require match_mode 'white_text'")
            return None
        options = {"min_channel": 220, "max_channel_spread": 25,
                   "min_ink_pixels": 8, "min_ink_fraction": 0.01,
                   "max_ink_fraction": 0.6}
        if self.white_text is not None:
            if not isinstance(self.white_text, dict) or set(self.white_text) - set(options):
                raise ValueError("white_text must contain only supported matching options")
            options.update(self.white_text)
        for name in ("min_channel", "max_channel_spread"):
            value = options[name]
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"white_text {name} must be an integer between 0 and 255")
        if type(options["min_ink_pixels"]) is not int or options["min_ink_pixels"] < 1:
            raise ValueError("white_text min_ink_pixels must be a positive integer")
        for name in ("min_ink_fraction", "max_ink_fraction"):
            value = options[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"white_text {name} must be finite and strictly between 0 and 1")
        if options["min_ink_fraction"] >= options["max_ink_fraction"]:
            raise ValueError("white_text min_ink_fraction must be below max_ink_fraction")
        return options

    @classmethod
    def from_dict(cls, data):
        allowed = {"phase", "region", "template", "max_distance", "margin", "match_mode", "white_text"}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("phase template must contain only supported fields")
        return cls(Phase(data["phase"]), Rect.from_list(data["region"]), data["template"], data.get("max_distance", 0.06), data.get("margin", 0.015), data.get("match_mode", "rgb"), data.get("white_text"))

    def to_dict(self):
        data = {"phase": self.phase.value, "region": self.region.to_list(), "template": self.template, "max_distance": self.max_distance, "margin": self.margin}
        if self.match_mode != "rgb":
            data["match_mode"] = self.match_mode
        if self.white_text is not None:
            data["white_text"] = dict(self.white_text)
        return data


@dataclass(frozen=True)
class SlotConfig:
    portrait: Rect
    attack: Rect | None = None
    health: Rect | None = None
    level: Rect | None = None
    empty_template: str | None = None
    species_templates: dict[str, str] = field(default_factory=dict)
    max_distance: float = 0.06
    margin: float = 0.015
    available_from_turn: int = 1

    @classmethod
    def from_dict(cls, data):
        regions = {name: Rect.from_list(data[name]) if data.get(name) is not None else None for name in ("attack", "health", "level")}
        return cls(portrait=Rect.from_list(data["portrait"]), **regions, empty_template=data.get("empty_template"), species_templates=data.get("species_templates", {}), max_distance=data.get("max_distance", 0.06), margin=data.get("margin", 0.015), available_from_turn=data.get("available_from_turn", 1))

    def to_dict(self):
        return {"portrait": self.portrait.to_list(), **{name: getattr(self, name).to_list() if getattr(self, name) else None for name in ("attack", "health", "level")}, "empty_template": self.empty_template, "species_templates": self.species_templates, "max_distance": self.max_distance, "margin": self.margin, "available_from_turn": self.available_from_turn}


@dataclass(frozen=True)
class NumberTemplate:
    """One manually verified numeric crop, never a default for unreadable text."""
    field: str
    value: int
    template: str
    max_distance: float = 0.01
    margin: float = 0.01

    @classmethod
    def from_dict(cls, data):
        return cls(data["field"], data["value"], data["template"], data.get("max_distance", 0.01), data.get("margin", 0.01))

    def to_dict(self):
        return {"field": self.field, "value": self.value, "template": self.template, "max_distance": self.max_distance, "margin": self.margin}


@dataclass(frozen=True)
class VisionProfile:
    image_size: tuple[int, int]
    phase_templates: tuple[PhaseTemplate, ...] = ()
    hud: dict[str, Rect] = field(default_factory=dict)
    shop: tuple[SlotConfig, ...] = ()
    team: tuple[SlotConfig, ...] = ()
    buttons: dict[str, tuple[int, int]] = field(default_factory=dict)
    calibrated: bool = False
    base_dir: Path = field(default_factory=lambda: Path("."), repr=False, compare=False)
    numeric_templates: tuple[NumberTemplate, ...] = ()

    def numeric_region(self, name):
        """Resolve an explicit HUD/slot field and its allowed integer range."""
        if name in {"gold", "turn", "wins", "lives"}:
            return self.hud.get(name), 1 if name == "turn" else 0, 99
        match = re.fullmatch(r"(shop|team)\.([0-4])\.(attack|health|level)", name) if isinstance(name, str) else None
        if match is None:
            raise ValueError(f"unknown numeric field: {name!r}")
        row, index, stat = match.groups()
        slots = getattr(self, row)
        if int(index) >= len(slots):
            raise ValueError(f"numeric field refers to missing slot: {name!r}")
        return getattr(slots[int(index)], stat), 0 if stat == "attack" else 1, 3 if stat == "level" else 99

    def validate(self, *, require_calibrated=False):
        width, height = _coordinates(self.image_size, 2, "image_size")
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        if type(self.calibrated) is not bool or (require_calibrated and not self.calibrated):
            raise ValueError("profile must be explicitly calibrated before desktop control")
        if set(self.hud) - {"gold", "turn", "wins", "lives"}:
            raise ValueError("unknown HUD field")
        if set(self.buttons) - {"roll", "end_turn", "sell", "continue", "name_adjective", "name_noun", "confirm_name", "continue_round", "dismiss_tier", "dismiss_life_reward", "confirm_end_turn", "open_play", "open_arena", "start_arena"}:
            raise ValueError("unknown button")
        for region in self.hud.values():
            region.validate(self.image_size)
        for point in self.buttons.values():
            x, y = _coordinates(point, 2, "button")
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError("button lies outside image_size")
        for item in self.phase_templates:
            if not isinstance(item.phase, Phase) or item.phase == Phase.UNKNOWN:
                raise ValueError("phase templates must identify a known phase")
            item.region.validate(self.image_size)
            self.template_path(item.template)
            _thresholds(item.max_distance, item.margin)
            item.matching_options()
        for row in (self.shop, self.team):
            if len(row) > 5:
                raise ValueError("shop and team may each contain at most five slots")
        for slot in (*self.shop, *self.team):
            if type(slot.available_from_turn) is not int or slot.available_from_turn < 1:
                raise ValueError("available_from_turn must be a positive integer")
            for name in ("portrait", "attack", "health", "level"):
                if getattr(slot, name) is not None:
                    getattr(slot, name).validate(self.image_size)
            _thresholds(slot.max_distance, slot.margin)
            if slot.empty_template is not None:
                self.template_path(slot.empty_template)
            if not isinstance(slot.species_templates, dict):
                raise ValueError("species_templates must map species names to local images")
            for species, path in slot.species_templates.items():
                if not isinstance(species, str) or not species.strip():
                    raise ValueError("species names must be nonempty strings")
                self.template_path(path)
        if any(slot.available_from_turn != 1 for slot in self.team):
            raise ValueError("team slots must be available from turn 1")
        if any(left.available_from_turn > right.available_from_turn
               for left, right in zip(self.shop, self.shop[1:])):
            raise ValueError("shop available_from_turn values must be nondecreasing")
        for item in self.numeric_templates:
            region, minimum, maximum = self.numeric_region(item.field)
            if region is None:
                raise ValueError(f"numeric reference requires a calibrated region: {item.field}")
            if type(item.value) is not int or not minimum <= item.value <= maximum:
                raise ValueError(f"numeric reference value is outside the field range: {item.field}")
            self.template_path(item.template)
            _thresholds(item.max_distance, item.margin)
        return self

    def template_path(self, name):
        if not isinstance(name, str) or not name.strip() or Path(name).is_absolute():
            raise ValueError("template paths must be nonempty relative local paths")
        root = Path(self.base_dir).resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("template paths must remain inside the profile directory")
        return path

    @classmethod
    def from_dict(cls, data, *, base_dir=Path(".")):
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("unsupported vision profile format; expected version 1")
        allowed = {"version", "image_size", "phase_templates", "hud", "shop", "team", "buttons", "calibrated", "numeric_templates"}
        if set(data) - allowed:
            raise ValueError(f"unknown vision profile fields: {sorted(set(data) - allowed)}")
        try:
            return cls(image_size=_coordinates(data["image_size"], 2, "image_size"), phase_templates=tuple(PhaseTemplate.from_dict(item) for item in data.get("phase_templates", [])), hud={name: Rect.from_list(value) for name, value in data.get("hud", {}).items()}, shop=tuple(SlotConfig.from_dict(item) for item in data.get("shop", [])), team=tuple(SlotConfig.from_dict(item) for item in data.get("team", [])), buttons={name: _coordinates(value, 2, "button") for name, value in data.get("buttons", {}).items()}, calibrated=data.get("calibrated", False), base_dir=Path(base_dir), numeric_templates=tuple(NumberTemplate.from_dict(item) for item in data.get("numeric_templates", []))).validate()
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid vision profile: {exc}") from exc

    def to_dict(self):
        return {"version": 1, "image_size": list(self.image_size), "calibrated": self.calibrated, "phase_templates": [item.to_dict() for item in self.phase_templates], "hud": {name: value.to_list() for name, value in self.hud.items()}, "shop": [slot.to_dict() for slot in self.shop], "team": [slot.to_dict() for slot in self.team], "buttons": {name: list(point) for name, point in self.buttons.items()}, "numeric_templates": [item.to_dict() for item in self.numeric_templates]}

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), base_dir=path.parent)

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def read_number(text, minimum, maximum):
    """Accept a complete integer only; never turn unreadable OCR into zero."""
    if not isinstance(text, str) or not re.fullmatch(r"[0-9]{1,3}", text.strip()):
        return None
    value = int(text.strip())
    return value if minimum <= value <= maximum else None


def image_distance(left, right):
    if left.shape != right.shape:
        raise ValueError("template dimensions differ from calibrated crop")
    return float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255)


def _white_text_distance(left, right, options):
    """Symmetric mask error; insufficient or filled crops cannot match."""
    if left.shape != right.shape:
        raise ValueError("template dimensions differ from calibrated crop")
    masks = []
    for image in (left, right):
        darkest = image.min(axis=2)
        spread = image.max(axis=2).astype(np.int16) - darkest.astype(np.int16)
        mask = (darkest >= options["min_channel"]) & (spread <= options["max_channel_spread"])
        ink = int(mask.sum())
        fraction = ink / mask.size
        if (ink < options["min_ink_pixels"] or fraction < options["min_ink_fraction"]
                or fraction > options["max_ink_fraction"]):
            return float("inf")
        masks.append(mask)
    # Normalizing by the union keeps blank background from diluting missing
    # strokes, and penalizes extra white strokes as well as absent ones.
    return float(np.count_nonzero(masks[0] ^ masks[1]) / np.count_nonzero(masks[0] | masks[1]))


class Perceptor:
    """Read one atomic frame. The injected OCR callable accepts an RGB array."""

    def __init__(self, profile: VisionProfile, *, ocr: Callable[[np.ndarray], str] | None = None):
        self.profile = profile.validate()
        self.ocr = ocr
        self._templates = {}
        self._numeric_templates = {}
        for item in profile.numeric_templates:
            self._numeric_templates.setdefault(item.field, []).append(item)

    def _template(self, name, region):
        if name not in self._templates:
            from PIL import Image
            with Image.open(self.profile.template_path(name)) as image:
                self._templates[name] = np.asarray(image.convert("RGB"))
        template = self._templates[name]
        if template.shape != (region.height, region.width, 3):
            raise ValueError(f"template {name!r} dimensions differ from calibrated crop")
        return template

    def _distance(self, frame, region, name):
        return image_distance(region.crop(frame), self._template(name, region))

    def _reference_number(self, frame, region, field_name):
        scores = {}
        for item in self._numeric_templates.get(field_name, ()):
            score = self._distance(frame, region, item.template)
            if item.value not in scores or score < scores[item.value][0]:
                scores[item.value] = (score, item)
        ordered = sorted(scores.values(), key=lambda pair: pair[0])
        if not ordered or ordered[0][0] > ordered[0][1].max_distance:
            return None, False
        score, item = ordered[0]
        if len(ordered) > 1 and ordered[1][0] - score <= item.margin:
            return None, True
        return item.value, True

    def _number(self, frame, region, minimum, maximum, field_name=None):
        if region is None:
            return None
        reference, matched = self._reference_number(frame, region, field_name)
        if matched and reference is None:
            return None
        if self.ocr is None:
            return reference
        try:
            value = read_number(self.ocr(region.crop(frame)), minimum, maximum)
        except TimeoutError:
            # Abort this observation instead of timing out again on every crop.
            raise
        except Exception:
            # OCR process failures must never authorize a mouse action.
            return None
        if matched:
            return reference if value is None or value == reference else None
        return value

    def _phase(self, frame):
        modes = {}
        for item in self.profile.phase_templates:
            options = item.matching_options()
            score = (self._distance(frame, item.region, item.template) if options is None
                     else _white_text_distance(item.region.crop(frame), self._template(item.template, item.region), options))
            scores = modes.setdefault(item.match_mode, {})
            if item.phase not in scores or score < scores[item.phase][0]:
                scores[item.phase] = (score, item)
        winners = set()
        # RGB and Jaccard errors have different meanings. Compare competitors
        # within each mode; every mode with eligible evidence must confidently
        # agree. A mode abstains only if none of its phase scores meets its own
        # threshold, so an ambiguous eligible rival cannot be bypassed.
        for scores in modes.values():
            ordered = sorted(scores.values(), key=lambda pair: pair[0])
            if not any(score <= item.max_distance for score, item in ordered):
                continue
            score, item = ordered[0]
            runner_up = ordered[1][0] if len(ordered) > 1 else float("inf")
            if score > item.max_distance or runner_up - score <= item.margin:
                return Phase.UNKNOWN
            winners.add(item.phase)
        return next(iter(winners)) if len(winners) == 1 else Phase.UNKNOWN

    def _slot(self, frame, slot, field_prefix=None):
        attack = self._number(frame, slot.attack, 0, 99, f"{field_prefix}.attack")
        health = self._number(frame, slot.health, 1, 99, f"{field_prefix}.health")
        stats_present = attack is not None and health is not None
        scores = sorted((self._distance(frame, slot.portrait, path), name) for name, path in slot.species_templates.items())
        empty_score = self._distance(frame, slot.portrait, slot.empty_template) if slot.empty_template is not None else float("inf")
        if empty_score <= slot.max_distance:
            # Blank OCR alone cannot resolve competing visual evidence. Empty
            # must beat every known pet reference by the configured margin.
            competing_pet = bool(scores) and scores[0][0] - empty_score <= slot.margin
            conflicting_stats = attack is not None or health is not None
            return PetSlot(occupied=None if competing_pet or conflicting_stats else False)
        if not stats_present:
            return PetSlot(occupied=None)
        species = None
        if scores and scores[0][0] <= slot.max_distance and (len(scores) == 1 or scores[1][0] - scores[0][0] > slot.margin):
            species = scores[0][1]
        return PetSlot(occupied=True, species=species, attack=attack, health=health, level=self._number(frame, slot.level, 1, 3, f"{field_prefix}.level"))

    def observe(self, frame):
        frame = np.asarray(frame)
        width, height = self.profile.image_size
        if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError("frame must be an RGB uint8 image matching calibrated image_size exactly")
        phase = self._phase(frame)
        if phase != Phase.SHOP:
            return Board(phase=phase)
        limits = {"gold": (0, 99), "turn": (1, 99), "wins": (0, 99), "lives": (0, 99)}
        hud = {name: self._number(frame, self.profile.hud.get(name), *bounds, field_name=name) for name, bounds in limits.items()}
        turn = hud["turn"]
        if turn is None and any(slot.available_from_turn > 1 for slot in self.profile.shop):
            # The active geometry is unknown. Never authorize input from a
            # partially observed shop or infer the turn from visible pet count.
            shop = tuple(PetSlot(None) for _ in self.profile.shop)
        else:
            shop = tuple(self._slot(frame, slot, f"shop.{i}")
                         for i, slot in enumerate(self.profile.shop)
                         if turn is None or slot.available_from_turn <= turn)
        return Board(phase=phase, **hud, shop=shop, team=tuple(self._slot(frame, slot, f"team.{i}") for i, slot in enumerate(self.profile.team)))
