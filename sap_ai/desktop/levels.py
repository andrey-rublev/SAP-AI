"""Retain stable, directly observed team levels across brief unreadable badges.

No OCR, image or desktop dependencies. This remembers readings; it never
infers experience or treats an unreadable slot as an occupied one.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from sap_ai.desktop.state import Action, Board, PetSlot, Phase


@dataclass(frozen=True)
class _Reading:
    pet: PetSlot
    confirmations: int


def _matches(reading: _Reading, pet: PetSlot) -> bool:
    known = reading.pet
    return (pet.occupied is True and pet.strength is not None
            and (known.attack, known.health) == (pet.attack, pet.health)
            and (known.species is None or pet.species is None or known.species == pet.species)
            and (pet.level is None or pet.level == known.level))


class LevelMemory:
    """Keep a direct level after two matching readable observations.

    ``observe`` consumes one *raw* captured Board and may confirm a reading.
    ``annotate`` performs preflight lookup without confirming another frame.
    A gap before confirmation breaks the streak. A confirmed reading survives
    a missing badge only while occupancy, stats and known species still agree.
    """

    def __init__(self):
        self._readings: dict[int, _Reading] = {}
        self._turn = None

    def reset(self):
        self._readings.clear()
        self._turn = None

    def _sync(self, board):
        if board.turn is not None:
            if self._turn is not None and board.turn < self._turn:
                self.reset()
            self._turn = board.turn

    def observe(self, board: Board) -> Board:
        if board.phase != Phase.SHOP:
            return board
        self._sync(board)
        if len(board.team) != 5:
            self._readings.clear()
            return board
        for index, pet in enumerate(board.team):
            reading = self._readings.get(index)
            if reading is not None and not _matches(reading, pet):
                self._readings.pop(index)
                reading = None
            if pet.occupied is not True or pet.strength is None:
                continue
            if pet.level is not None:
                count = min(reading.confirmations + 1, 2) if reading is not None else 1
                self._readings[index] = _Reading(pet, count)
            elif reading is not None and reading.confirmations < 2:
                self._readings.pop(index)  # confirmation must be consecutive
        return self.annotate(board)

    def annotate(self, board: Board) -> Board:
        if board.phase != Phase.SHOP:
            return board
        self._sync(board)
        if len(board.team) != 5:
            self._readings.clear()
            return board
        team = []
        for index, pet in enumerate(board.team):
            reading = self._readings.get(index)
            if reading is not None and not _matches(reading, pet):
                self._readings.pop(index)
                reading = None
            if reading is not None and reading.confirmations == 2 and pet.level is None:
                pet = replace(pet, level=reading.pet.level)
            team.append(pet)
        return replace(board, team=tuple(team))

    def forget(self, action: Action):
        """An input may have changed its target even if its receipt later fails."""
        target = action.slot if action.kind == "sell" else action.target
        if action.kind in {"buy", "merge", "buy_food", "sell"} and target is not None:
            self._readings.pop(target, None)
