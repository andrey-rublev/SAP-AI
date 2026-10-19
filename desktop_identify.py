"""Identify pet species in the live shop by hovering and reading tooltip titles.

Portrait templates only cover a handful of species, so most pets were observed
with ``species=None`` and merges never fired. Hovering a pet shows a tooltip
whose title is the pet's name; :func:`desktop_tooltip.read_pet_title` reads it
against a fixed vocabulary (two agreeing OCR passes, exact names only).

Binding a tooltip to the hovered slot relies on two observations: the parked
frame has no tooltip candidate, and the settled hover frame has exactly one,
with identical clipped white regions (static HUD art touching the search edge)
in both frames, so nothing else appeared. Names are
cached per slot and keyed by the slot's observed stats, so each pet is hovered
once; actions that move or replace pets invalidate the affected slots.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from desktop_state import Action, Board, PetSlot
from desktop_tooltip import read_pet_title, tesseract_text_ocr
from desktop_tooltip_layout import TooltipLayout, TooltipPanelInventory, inventory_tooltip_panels


# Pets across the standard packs plus common summoned tokens. Matching is exact,
# so extra names never cause a wrong reading; a missing name only abstains.
PET_NAMES = (
    "Ant", "Beaver", "Beetle", "Bluebird", "Chinchilla", "Cricket", "Duck", "Duckling",
    "Fish", "Frog", "Hummingbird", "Horse", "Iguana", "Kiwi", "Ladybug", "Mosquito",
    "Otter", "Pig", "Pigeon", "Sloth", "Bat", "Bear", "Bison", "Blowfish",
    "Boar", "Buffalo", "Bus", "Butterfly", "Camel", "Cat", "Caterpillar", "Chick",
    "Chicken", "Cow", "Crab", "Crocodile", "Crow", "Deer", "Dodo", "Dog", "Dolphin",
    "Dragon", "Eagle", "Elephant", "Ferret", "Fly", "Flamingo", "Fox", "Giraffe",
    "Goat", "Gorilla", "Hatching Chick", "Hedgehog", "Hippo", "Hyena", "Jellyfish",
    "Kangaroo", "Koala", "Lion", "Llama", "Lobster", "Mammoth", "Microbe", "Mole",
    "Monkey", "Moose", "Mouse", "Octopus", "Orangutan", "Owl", "Ox", "Panda",
    "Panther", "Parrot", "Peacock", "Penguin", "Pony", "Poodle", "Porcupine",
    "Puppy", "Rabbit", "Ram", "Rat", "Dirty Rat", "Rhino", "Rooster", "Salmon",
    "Saber Tooth Tiger", "Scorpion", "Seagull", "Seal", "Shark", "Sheep", "Shrimp",
    "Skunk", "Snail", "Snake", "Spider", "Squid", "Squirrel", "Stegosaurus", "Stork",
    "Swan", "Tabby Cat", "Tapir", "Tiger", "Toad", "Tropical Fish", "Turkey",
    "Turtle", "Tyrannosaurus", "Velociraptor", "Walrus", "Whale", "Wolverine",
    "Worm", "Zombie Cricket", "Zombie Fly", "Bee", "Gecko", "Guinea Pig",
    "Lizard", "Leopard", "Sauropod", "Silkmoth", "Snapping Turtle", "Woodpecker",
)


def load_tooltip_layout(path, image_size) -> TooltipLayout:
    """Load a private tooltip calibration written for this exact client size."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or set(data) != {"version", "calibrated", "image_size", "layout"}
            or data["version"] != 1 or data["calibrated"] is not True
            or data["image_size"] != list(image_size) or not isinstance(data["layout"], dict)):
        raise ValueError("tooltip layout needs version 1, calibrated=true and the profile's image_size")
    try:
        layout = TooltipLayout(**data["layout"])
    except TypeError as exc:
        raise ValueError("unsupported tooltip layout fields") from exc
    x, y, width, height = layout.search_rect
    if x + width > image_size[0] or y + height > image_size[1]:
        raise ValueError("tooltip search rectangle lies outside the game client")
    return layout


@dataclass
class _Entry:
    key: tuple
    species: str | None = None
    attempts: int = 0


def _only_static_clipping(inventory: TooltipPanelInventory) -> bool:
    return set(inventory.unresolved_reasons) <= {"search_boundary"}


def _key(pet: PetSlot):
    return (pet.attack, pet.health, pet.level)


class SpeciesIdentifier:
    """Per-slot species cache plus the hover/read protocol.

    ``annotate`` is pure; ``identify`` performs IO through the supplied
    ``hover``/``capture``/``park`` callables so the runtime keeps every focus,
    geometry and stop check.
    """

    def __init__(self, layout: TooltipLayout, *, ocr: Callable[[np.ndarray], str] = tesseract_text_ocr,
                 vocabulary=PET_NAMES, settle_seconds=.75, max_hovers=3, max_attempts=2,
                 sleep=time.sleep):
        if not isinstance(layout, TooltipLayout):
            raise TypeError("layout must be a TooltipLayout")
        if type(max_hovers) is not int or max_hovers < 1 or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_hovers and max_attempts must be positive integers")
        self.layout, self.ocr, self.sleep = layout, ocr, sleep
        self.vocabulary = tuple(dict.fromkeys(vocabulary))
        self.settle_seconds, self.max_hovers, self.max_attempts = settle_seconds, max_hovers, max_attempts
        self._slots: dict[tuple[str, int], _Entry] = {}
        self._turn = None
        self.hovers = self.identified = 0

    # -- cache ---------------------------------------------------------------
    def _sync_turn(self, board: Board) -> None:
        if board.turn is not None and board.turn != self._turn:
            # A new turn brings a new shop; team pets keep their identities.
            self._turn = board.turn
            self._slots = {slot: entry for slot, entry in self._slots.items() if slot[0] == "team"}

    def _entry(self, row: str, index: int, pet: PetSlot) -> _Entry | None:
        entry = self._slots.get((row, index))
        if entry is None or pet.occupied is not True or entry.key != _key(pet):
            return None
        return entry

    def annotate(self, board: Board) -> Board:
        """Fill unread species from the cache; never overrides a template reading."""
        if not isinstance(board, Board):
            return board
        self._sync_turn(board)

        def fill(row):
            pets = []
            for index, pet in enumerate(getattr(board, row)):
                entry = self._entry(row, index, pet)
                if pet.species is None and entry is not None and entry.species is not None:
                    pet = replace(pet, species=entry.species)
                pets.append(pet)
            return tuple(pets)

        return replace(board, shop=fill("shop"), team=fill("team"))

    def targets(self, board: Board) -> list[tuple[str, int]]:
        """Occupied slots with complete stats, no species and attempts left."""
        result = []
        for row in ("team", "shop"):
            for index, pet in enumerate(getattr(board, row)):
                if pet.occupied is not True or pet.species is not None or pet.strength is None:
                    continue
                entry = self._entry(row, index, pet)
                if entry is None or (entry.species is None and entry.attempts < self.max_attempts):
                    result.append((row, index))
        return result

    def record(self, row: str, index: int, pet: PetSlot, species: str | None) -> None:
        entry = self._entry(row, index, pet) or _Entry(_key(pet))
        entry.attempts += 1
        if species is not None:
            entry.species = " ".join(species.split()).casefold()
            self.identified += 1
        self._slots[(row, index)] = entry

    def forget(self, action: Action) -> None:
        """Invalidate slots whose occupant an executed action may have changed."""
        if action.kind in {"roll", "buy", "merge", "buy_food"}:
            self._slots = {slot: entry for slot, entry in self._slots.items() if slot[0] != "shop"}
        if action.kind in {"buy", "merge", "buy_food"} and action.target is not None:
            self._slots.pop(("team", action.target), None)
        if action.kind == "sell" and action.slot is not None:
            self._slots.pop(("team", action.slot), None)

    # -- IO protocol ----------------------------------------------------------
    def identify(self, board: Board, parked_frame, *, points, hover, capture, park) -> Board:
        """Hover up to ``max_hovers`` unknown pets and return the annotated board.

        ``points[(row, index)]`` is each slot's client point. Nothing is hovered
        unless the parked frame is free of tooltips, which binds each later
        tooltip to the pet under the pointer. The pointer is parked afterwards.
        """
        self._sync_turn(board)
        targets = self.targets(board)[:self.max_hovers]
        if not targets:
            return board
        parked = inventory_tooltip_panels(np.asarray(parked_frame), self.layout)
        if parked.candidates or not _only_static_clipping(parked):
            return board
        try:
            for row, index in targets:
                pet = getattr(board, row)[index]
                hover(points[(row, index)])
                self.sleep(self.settle_seconds)
                self.hovers += 1
                found = inventory_tooltip_panels(np.asarray(capture()), self.layout)
                name = None
                if (len(found.candidates) == 1 and _only_static_clipping(found)
                        and found.boundary_rects == parked.boundary_rects):
                    name = read_pet_title(found.candidates[0].title_crop, self.vocabulary, ocr=self.ocr)
                self.record(row, index, pet, name)
        finally:
            park()
        return self.annotate(board)
