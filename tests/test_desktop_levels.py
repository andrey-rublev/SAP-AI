"""Observation memory for intermittent level badges; no native dependencies."""
from dataclasses import replace

import pytest

from sap_ai.desktop.levels import LevelMemory
from sap_ai.desktop.state import Action, Board, PetSlot, Phase


EMPTY = PetSlot(False)


def board(level=1, *, attack=3, species="fish", turn=4):
    pet = PetSlot(True, species, attack, 4, level)
    return Board(Phase.SHOP, gold=6, turn=turn, shop=(EMPTY,), team=(pet,) + (EMPTY,) * 4)


def test_two_direct_readings_recover_a_later_missing_badge():
    memory = LevelMemory()
    first, second = memory.observe(board()), memory.observe(board())
    assert first.team[0].level == second.team[0].level == 1
    raw = board(None)
    assert memory.observe(raw).team[0].level == 1
    assert raw.team[0].level is None


def test_preflight_lookups_cannot_confirm_the_same_capture_repeatedly():
    memory = LevelMemory()
    memory.observe(board())
    for _ in range(5):
        assert memory.annotate(board(None)).team[0].level is None
    assert memory.observe(board(None)).team[0].level is None


def test_a_gap_breaks_an_unconfirmed_streak():
    memory = LevelMemory()
    for value in (1, None, 1, None):
        assert memory.observe(board(value)).team[0].level == value


@pytest.mark.parametrize("change", [board(None, attack=5), board(None, species="pig"),
                                     replace(board(), team=(EMPTY,) * 5),
                                     replace(board(), team=(PetSlot(None),) + (EMPTY,) * 4)])
def test_changed_or_ambiguous_pet_evidence_revokes_a_confirmed_level(change):
    memory = LevelMemory()
    memory.observe(board()), memory.observe(board())
    memory.annotate(change)  # even a preflight contradiction revokes memory
    assert memory.observe(board(None)).team[0].level is None


def test_visible_contradiction_wins_and_must_confirm_again():
    memory = LevelMemory()
    memory.observe(board()), memory.observe(board())
    assert memory.observe(board(2)).team[0].level == 2
    assert memory.observe(board(None)).team[0].level is None
    memory.observe(board(2)), memory.observe(board(2))
    assert memory.observe(board(None)).team[0].level == 2


@pytest.mark.parametrize("action", [Action("buy", 0, 0), Action("merge", 0, 0),
                                     Action("buy_food", 0, 0), Action("sell", 0)])
def test_mutating_actions_forget_the_target_before_any_receipt(action):
    memory = LevelMemory()
    memory.observe(board()), memory.observe(board())
    memory.forget(action)
    assert memory.observe(board(None)).team[0].level is None


def test_rolls_and_round_transitions_preserve_unchanged_team_evidence():
    memory = LevelMemory()
    memory.observe(board()), memory.observe(board())
    memory.forget(Action("roll"))
    memory.observe(Board(Phase.BATTLE))
    assert memory.observe(board(None, turn=5)).team[0].level == 1
    assert memory.observe(board(None, turn=1)).team[0].level is None


def test_incomplete_shop_geometry_and_explicit_reset_clear_evidence():
    memory = LevelMemory()
    memory.observe(board()), memory.observe(board())
    memory.observe(replace(board(), team=board().team[:4]))
    assert memory.observe(board(None)).team[0].level is None
    memory.observe(board()), memory.observe(board())
    memory.reset()
    assert memory.observe(board(None)).team[0].level is None
