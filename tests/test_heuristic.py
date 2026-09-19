"""Unit tests for heuristic.HeuristicAgent."""
import numpy as np

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


def test_prefers_combine_over_empty_slot():
    agent = HeuristicAgent()
    # shop slot 2 (value 4) matches team pet 4; slots 0/1 are stronger but no match.
    obs = obs_from(gold=10, turn=1, shop=[6, 5, 4], team=[4, 0, 0, 0, 0])
    assert agent.choose_action(obs) == 2


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
