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


def test_parse_int_extracts_digits():
    assert autogui.parse_int("A 3 / 5") == 3
    assert autogui.parse_int("no digits", default=7) == 7
    assert autogui.parse_int("") == 0


def test_build_observation_shape_and_values():
    obs = autogui.build_observation(gold=10, shop=[1, 2, 3], team=[4, 5, 0, 0, 0], turn=2)
    assert obs.shape == (10,)
    assert list(obs) == [10, 2, 1, 2, 3, 4, 5, 0, 0, 0]


def test_build_observation_pads_short_lists():
    obs = autogui.build_observation(gold=5, shop=[6], team=[], turn=0)
    assert list(obs) == [5, 0, 6, 0, 0, 0, 0, 0, 0, 0]


def test_build_observation_feeds_mask_from_obs():
    from game import SuperAutoPetsEnv

    obs = autogui.build_observation(gold=10, shop=[0, 4, 0], team=[0, 0, 0, 0, 0])
    mask = SuperAutoPetsEnv.mask_from_obs(obs)
    assert list(mask) == [False, True, False, True, False, True]
