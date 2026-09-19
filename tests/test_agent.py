"""Unit tests for agent.QLearningAgent."""
import numpy as np
import pytest

from agent import QLearningAgent


def test_choose_action_is_valid():
    agent = QLearningAgent([0, 1, 2], seed=0)
    for _ in range(20):
        assert agent.choose_action("s") in agent.actions


def test_greedy_picks_argmax():
    agent = QLearningAgent([0, 1, 2], epsilon=1.0, seed=0)
    agent.q["s"] = {0: 1.0, 1: 5.0, 2: 2.0}
    assert agent.choose_action("s", greedy=True) == 1


def test_choose_action_respects_mask_when_exploring():
    agent = QLearningAgent([0, 1, 2], epsilon=1.0, seed=0)  # always explore
    for _ in range(30):
        assert agent.choose_action("s", mask=[False, True, False]) == 1


def test_greedy_respects_mask():
    agent = QLearningAgent([0, 1, 2], seed=0)
    agent.q["s"] = {0: 5.0, 1: 1.0, 2: 2.0}
    # action 0 has the highest value but is masked out -> best legal is action 2
    assert agent.choose_action("s", greedy=True, mask=[False, True, True]) == 2


def test_learn_terminal_update():
    agent = QLearningAgent([0, 1], alpha=0.5, gamma=0.9, seed=0)
    agent.learn("s", 0, reward=10.0, next_state="s2", done=True)
    # target = reward (no bootstrap when done); new = 0 + 0.5 * (10 - 0)
    assert agent.q["s"][0] == pytest.approx(5.0)


def test_learn_bootstraps_from_next_state():
    agent = QLearningAgent([0, 1], alpha=1.0, gamma=0.5, seed=0)
    agent.q["s2"] = {0: 4.0, 1: 8.0}
    agent.learn("s", 0, reward=1.0, next_state="s2", done=False)
    # target = 1 + 0.5 * max(4, 8) = 5; alpha=1 -> q becomes the target
    assert agent.q["s"][0] == pytest.approx(5.0)


def test_epsilon_decays_to_floor():
    agent = QLearningAgent([0], epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


def test_state_fn_is_applied():
    calls = []

    def state_fn(s):
        calls.append(s)
        return "bucket"

    agent = QLearningAgent([0, 1], state_fn=state_fn, seed=0)
    agent.choose_action(np.array([1.0, 2.0]), greedy=True)  # greedy path uses _key
    assert calls  # state_fn was invoked
    assert "bucket" in agent.q  # keyed by the abstracted state


def test_numpy_state_is_hashable_key():
    agent = QLearningAgent([0, 1], seed=0)
    agent.choose_action(np.array([1.4, 2.6, 3.0]), greedy=True)
    assert (1, 3, 3) in agent.q  # rounded to ints and tupled


def test_save_load_round_trip_preserves_types(tmp_path):
    agent = QLearningAgent([0, 1, 2], seed=0)
    agent.q[(1, 2)] = {0: 0.5, 1: -0.5, 2: 1.5}
    path = tmp_path / "q.json"
    agent.save(path)

    loaded = QLearningAgent([]).load(path)
    assert loaded.actions == [0, 1, 2]
    row = loaded.q[(1, 2)]
    assert set(map(type, row)) == {int}  # action keys stay ints
    assert row == {0: 0.5, 1: -0.5, 2: 1.5}
    assert isinstance(loaded.choose_action((1, 2), greedy=True), int)


def test_unpack_step_handles_both_apis():
    assert QLearningAgent._unpack_step(("o", 1.0, True, False, {})) == ("o", 1.0, True, {})
    assert QLearningAgent._unpack_step(("o", 1.0, True, {})) == ("o", 1.0, True, {})


def test_unpack_reset_handles_both_apis():
    assert QLearningAgent._unpack_reset(("o", {})) == "o"
    assert QLearningAgent._unpack_reset("o") == "o"


def test_train_returns_history_and_learns():
    from game import SuperAutoPetsEnv

    env = SuperAutoPetsEnv()
    agent = QLearningAgent(list(range(env.action_space.n)), seed=0)
    history = agent.train(env, num_episodes=50, max_steps=100)
    assert len(history) == 50
    assert agent.epsilon < 1.0  # epsilon decayed during training
