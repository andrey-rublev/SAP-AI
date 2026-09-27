"""Unit tests for heuristic.HeuristicAgent."""
import numpy as np
import pytest

from game import SuperAutoPetsEnv
from heuristic import HeuristicAgent
from main import evaluate


def obs_from(gold, turn, shop, team):
    return np.array([gold, turn, *shop, *team], dtype=np.float32)


def test_action_is_always_valid():
    agent = HeuristicAgent()
    rng = np.random.default_rng(0)
    for _ in range(100):
        obs = rng.integers(0, 7, size=10).astype(np.float32)
        assert agent.choose_action(obs) in range(6)


def test_prefers_larger_strength_gain_over_combine():
    agent = HeuristicAgent()
    # shop slot 2 (value 4) matches team pet 4; slots 0/1 are stronger but no match.
    obs = obs_from(gold=10, turn=1, shop=[6, 5, 4], team=[4, 0, 0, 0, 0])
    assert agent.choose_action(obs) == 0  # six strength is better than a +1 combine


def test_fills_empty_slot_with_strongest():
    agent = HeuristicAgent()
    obs = obs_from(gold=10, turn=1, shop=[2, 6, 3], team=[0, 0, 0, 0, 0])
    assert agent.choose_action(obs) == 1  # strongest shop pet


def test_ends_turn_when_broke():
    agent = HeuristicAgent()
    obs = obs_from(gold=1, turn=1, shop=[6, 6, 6], team=[3, 0, 0, 0, 0])
    assert agent.choose_action(obs) == 5


def test_beats_random_by_a_wide_margin():
    env = SuperAutoPetsEnv()
    rand_rate, _ = evaluate(env, agent=None, episodes=100)
    heur_rate, _ = evaluate(env, agent=HeuristicAgent(), episodes=100)
    assert heur_rate > rand_rate + 0.5


def test_respects_supplied_mask():
    agent = HeuristicAgent()
    obs = obs_from(10, 0, [2, 6, 3], [0, 0, 0, 0, 0])
    assert agent.choose_action(obs, mask=[1, 0, 0, 0, 0, 1]) == 0
    assert agent.choose_action(obs, mask=[0, 0, 0, 0, 0, 1]) == 5
    assert agent.choose_action(obs, mask=[0, 0, 0, 1, 0, 0]) == 3


@pytest.mark.parametrize("mask", [[False] * 6, [True] * 5, [2] * 6])
def test_rejects_empty_or_malformed_masks(mask):
    with pytest.raises(ValueError):
        HeuristicAgent().choose_action(obs_from(10, 0, [2, 6, 3], [0] * 5), mask=mask)


def test_sells_for_affordable_upgrade_including_sale_proceeds():
    agent = HeuristicAgent()
    obs = obs_from(2, 2, [6, 0, 0], [1, 2, 3, 4, 5])
    assert agent.choose_action(obs) == 4


def test_does_not_sell_for_a_purchase_that_would_only_combine():
    agent = HeuristicAgent()
    obs = obs_from(2, 2, [6, 0, 0], [2, 3, 4, 5, 6])
    assert agent.choose_action(obs) == 5


def test_rolls_empty_shop_when_purchase_would_remain_affordable():
    obs = obs_from(4, 1, [0, 0, 0], [6, 0, 0, 0, 0])
    assert HeuristicAgent().choose_action(obs) == 3


def test_extended_observation_metadata_is_not_a_pet():
    legacy = obs_from(3, 2, [1, 0, 0], [2, 3, 4, 5, 6])
    extended = np.concatenate((legacy, [1, 5, 0]))
    assert HeuristicAgent().choose_action(legacy) == 5
    assert HeuristicAgent().choose_action(extended) == 5


def test_skips_unneeded_shop_actions_when_remaining_victories_are_guaranteed():
    legacy = obs_from(10, 8, [6, 5, 4], [7, 7, 6, 5, 0])
    extended = np.concatenate((legacy, [8, 5, 0]))
    assert HeuristicAgent().choose_action(extended) == 5


def test_guaranteed_win_shortcut_still_honors_external_mask():
    legacy = obs_from(10, 8, [6, 5, 4], [7, 7, 6, 5, 0])
    extended = np.concatenate((legacy, [8, 5, 0]))
    assert HeuristicAgent().choose_action(extended, mask=[0, 0, 1, 0, 0, 0]) == 2


def test_entire_heuristic_runs_use_only_legal_actions():
    env, agent = SuperAutoPetsEnv(), HeuristicAgent()
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        while True:
            action = agent.choose_action(obs, mask=info["action_mask"])
            assert info["action_mask"][action]
            obs, _, terminated, truncated, info = env.step(action)
            assert info["action_valid"]
            if terminated or truncated:
                assert not truncated
                break
