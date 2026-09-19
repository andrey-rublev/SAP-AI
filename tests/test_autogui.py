"""Tests for the pure logic in autogui (no real mouse/OCR involved)."""
import numpy as np
import pytest

import autogui


def obs(team):
    return np.array([10, 1, 6, 5, 4, *team], dtype=np.float32)


def test_weakest_slot_none_without_obs():
    assert autogui._weakest_team_slot(None) is None


def test_weakest_slot_picks_lowest_occupied():
    assert autogui._weakest_team_slot(obs([0, 3, 0, 7, 2])) == 4  # value 2 is lowest


def test_weakest_slot_none_when_empty_team():
    assert autogui._weakest_team_slot(obs([0, 0, 0, 0, 0])) is None


def test_perform_sell_requires_observation():
    with pytest.raises(ValueError):
        autogui.perform_action(4, obs=None)


def test_perform_action_rejects_unknown():
    with pytest.raises(ValueError):
        autogui.perform_action(99)
