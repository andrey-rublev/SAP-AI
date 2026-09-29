"""Observed desktop state and a conservative policy for the real game.

This schema deliberately differs from the scalar toy simulator. A missing
reading stays ``None``; in particular, an unreadable slot is never an empty
slot. The initial policy values observed attack and health, not pet abilities.
It needs no desktop, OCR, NumPy, or model-training dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class Phase(str, Enum):
    UNKNOWN = "unknown"
    SHOP = "shop"
    BATTLE = "battle"
    NAMING = "naming"
    NAMING_READY = "naming_ready"
    ROUND_RESULT = "round_result"
    TIER_UNLOCK = "tier_unlock"
    RESULT = "result"


def _integer(value: int | None, name: str, minimum: int = 0, maximum: int | None = None) -> None:
    if value is None:
        return
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        limit = f" through {maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be an integer {minimum}{limit}, or None")


@dataclass(frozen=True)
class PetSlot:
    occupied: bool | None
    species: str | None = None
    attack: int | None = None
    health: int | None = None
    level: int | None = None

    def __post_init__(self) -> None:
        if self.occupied is not None and type(self.occupied) is not bool:
            raise ValueError("occupied must be True, False, or None")
        if self.species is not None:
            if not isinstance(self.species, str) or not self.species.strip():
                raise ValueError("species must be a nonempty string or None")
            object.__setattr__(self, "species", " ".join(self.species.split()).casefold())
        _integer(self.attack, "attack")
        _integer(self.health, "health", minimum=1)
        _integer(self.level, "level", minimum=1, maximum=3)
        if self.occupied is False and any(value is not None for value in (
            self.species, self.attack, self.health, self.level,
        )):
            raise ValueError("an empty slot cannot contain pet metadata")

    @property
    def strength(self) -> int | None:
        """Return observed attack + health, never a fabricated default."""
        if self.occupied is not True or self.attack is None or self.health is None:
            return None
        return self.attack + self.health


@dataclass(frozen=True)
class Board:
    phase: Phase
    gold: int | None = None
    turn: int | None = None
    wins: int | None = None
    lives: int | None = None
    shop: tuple[PetSlot, ...] = ()
    team: tuple[PetSlot, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", Phase(self.phase))
        for name in ("gold", "turn", "wins", "lives"):
            _integer(getattr(self, name), name)
        for name in ("shop", "team"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or len(values) > 5:
                raise ValueError(f"{name} must contain at most five PetSlot values")
            if not all(isinstance(slot, PetSlot) for slot in values):
                raise ValueError(f"{name} must contain PetSlot values")
            object.__setattr__(self, name, tuple(values))

    def fingerprint(self) -> tuple:
        """A stable, hashable state signature for repeated-frame detection."""
        def slots(values: tuple[PetSlot, ...]) -> tuple:
            return tuple((pet.occupied, pet.species, pet.attack, pet.health, pet.level) for pet in values)

        return (self.phase.value, self.gold, self.turn, self.wins, self.lives,
                slots(self.shop), slots(self.team))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phase"] = self.phase.value
        result["shop"] = list(result["shop"])
        result["team"] = list(result["team"])
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Board:
        values = dict(data)
        for name in ("shop", "team"):
            if name in values:
                values[name] = tuple(PetSlot(**slot) for slot in values[name])
        return cls(**values)


@dataclass(frozen=True)
class Action:
    kind: str
    slot: int | None = None
    target: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"buy", "roll", "sell", "end_turn", "merge", "continue",
                             "choose_name", "confirm_name", "continue_round", "dismiss_tier"}:
            raise ValueError(f"unknown desktop action: {self.kind!r}")
        _integer(self.slot, "slot", maximum=4)
        _integer(self.target, "target", maximum=4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Action:
        return cls(**dict(data))


def _known_shop(board: Board) -> bool:
    """Reject incomplete geometry and any ambiguous slot occupancy."""
    return (
        board.phase == Phase.SHOP
        and board.gold is not None
        and 1 <= len(board.shop) <= 5
        and len(board.team) == 5
        and all(pet.occupied is not None for pet in (*board.shop, *board.team))
    )


def legal_action(board: Board, action: Action) -> bool:
    """Validate an action against observations, without touching the desktop.

    A standard pet costs three gold. Combines require an identified level-one
    shop pet and an identified matching teammate below level three. Interstitial
    actions require their exact recognized phase and no slot arguments. The
    runtime separately requires calibrated points; generic continue is disabled.
    """
    if not isinstance(board, Board) or not isinstance(action, Action):
        return False
    transitions = {"choose_name": Phase.NAMING, "confirm_name": Phase.NAMING_READY,
                   "continue_round": Phase.ROUND_RESULT, "dismiss_tier": Phase.TIER_UNLOCK}
    if action.kind in transitions:
        return (board.phase == transitions[action.kind]
                and action.slot is None and action.target is None)
    if not _known_shop(board):
        return False
    if action.kind in {"roll", "end_turn"}:
        return action.slot is None and action.target is None and (
            action.kind == "end_turn" or board.gold >= 1
        )
    if action.kind == "sell":
        return (
            action.slot is not None and action.slot < len(board.team)
            and action.target is None and board.team[action.slot].occupied is True
        )
    if action.kind in {"buy", "merge"}:
        if (board.gold < 3 or action.slot is None or action.target is None
                or action.slot >= len(board.shop) or action.target >= len(board.team)):
            return False
        source, target = board.shop[action.slot], board.team[action.target]
        if source.occupied is not True:
            return False
        if action.kind == "buy":
            return target.occupied is False
        return (
            target.occupied is True
            and source.species is not None and source.species == target.species
            and source.level == 1 and target.level is not None and target.level < 3
        )
    return False


class DesktopPolicy:
    """A deterministic first desktop baseline using actual observed stats.

    Fill empty spaces, combine identified duplicates, and replace only known
    level-one pets for a substantial stat gain. This does not model abilities,
    food, frozen shops, or positioning, and never invokes the toy DQN.
    """

    def __init__(self, *, minimum_upgrade_gain: int = 4) -> None:
        _integer(minimum_upgrade_gain, "minimum_upgrade_gain", minimum=1)
        if minimum_upgrade_gain is None:
            raise ValueError("minimum_upgrade_gain must be an integer")
        self.minimum_upgrade_gain = minimum_upgrade_gain

    def choose_action(self, board: Board) -> Action | None:
        if not isinstance(board, Board) or not _known_shop(board):
            return None
        # Missing stats can change both the best purchase and the weakest pet.
        if any(pet.occupied and pet.strength is None for pet in (*board.shop, *board.team)):
            return None
        offers = [(index, pet) for index, pet in enumerate(board.shop) if pet.occupied]
        # The second key makes equal-strength choices deterministic by slot.
        offers.sort(key=lambda pair: (-pair[1].strength, pair[0]))
        empty = next((index for index, pet in enumerate(board.team) if pet.occupied is False), None)
        if board.gold >= 3 and offers:
            if empty is not None:
                return Action("buy", offers[0][0], empty)

            merges = [
                (index, target)
                for index, _ in offers
                for target in range(len(board.team))
                if legal_action(board, Action("merge", index, target))
            ]
            if merges:
                return Action("merge", *merges[0])

            # A positive margin ensures replacements increase observed stats.
            # Do not discard unknown-level or upgraded pets for stat-only gains.
            replaceable = [(index, pet) for index, pet in enumerate(board.team) if pet.level == 1]
            if replaceable:
                weakest, pet = min(replaceable, key=lambda pair: (pair[1].strength, pair[0]))
                if offers[0][1].strength - pet.strength >= self.minimum_upgrade_gain:
                    return Action("sell", weakest)

        # Preserve enough gold to use the new shop after a refresh.
        if board.gold >= 4:
            return Action("roll")
        return Action("end_turn")
