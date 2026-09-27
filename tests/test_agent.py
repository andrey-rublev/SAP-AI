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
    assert agent._key(np.array([1.0, 2.0])) == "bucket"
    assert not agent.q  # inference does not grow the learned table


def test_numpy_state_is_hashable_key():
    agent = QLearningAgent([0, 1], seed=0)
    agent.choose_action(np.array([1.4, 2.6, 3.0]), greedy=True)
    assert agent._key(np.array([1.4, 2.6, 3.0])) == (1, 3, 3)


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


def test_bootstrap_ignores_illegal_next_actions():
    agent = QLearningAgent([0, 1, 2], alpha=1.0, gamma=0.5)
    agent.q["next"] = {0: 999.0, 1: -4.0, 2: -2.0}
    agent.learn("state", 0, 1.0, "next", False, next_mask=[False, True, True])
    assert agent.q["state"][0] == pytest.approx(0.0)


@pytest.mark.parametrize("terminated", [False, True])
def test_empty_next_mask_has_no_bootstrap(terminated):
    agent = QLearningAgent([0, 1], alpha=1.0)
    agent.q["next"] = {0: 999.0, 1: 1000.0}
    agent.learn("state", 0, 3.0, "next", terminated, next_mask=[False, False])
    assert agent.q["state"][0] == 3.0


@pytest.mark.parametrize("greedy", [False, True])
def test_empty_policy_mask_is_rejected(greedy):
    with pytest.raises(ValueError, match="no legal actions"):
        QLearningAgent([0, 1]).choose_action("state", greedy=greedy, mask=[False, False])


@pytest.mark.parametrize("classic", [False, True])
@pytest.mark.parametrize("terminated", [False, True])
def test_train_bootstraps_truncations_but_not_terminations(classic, terminated):
    class OneStepEnv:
        def reset(self):
            return "start", {"action_mask": [True, False]}

        def step(self, action):
            assert action == 0
            info = {"action_mask": [False, True], "TimeLimit.truncated": not terminated}
            if classic:
                return "next", 1.0, True, info
            return "next", 1.0, terminated, not terminated, info

    agent = QLearningAgent([0, 1], alpha=1.0, gamma=0.5)
    agent.q["next"] = {0: 1000.0, 1: 8.0}
    assert agent.train(OneStepEnv(), num_episodes=1) == [1.0]
    assert agent.q["start"][0] == (1.0 if terminated else 5.0)


def test_training_seed_is_reproducible():
    from game import SuperAutoPetsEnv
    first = QLearningAgent(range(6), seed=13)
    second = QLearningAgent(range(6), seed=13)
    assert first.train(SuperAutoPetsEnv(), num_episodes=6) == second.train(SuperAutoPetsEnv(), num_episodes=6)
    assert dict(first.q) == dict(second.q)


def test_seed_is_only_passed_to_first_reset():
    class Env:
        seeds = []

        def reset(self, *, seed=None):
            self.seeds.append(seed)
            return "state", {}

        def step(self, action):
            return "end", 1.0, True, False, {}

    env = Env()
    QLearningAgent([0], seed=7).train(env, num_episodes=3)
    assert env.seeds == [7, None, None]


def test_checkpoint_restores_rng_and_typed_states(tmp_path):
    agent = QLearningAgent(["buy", "roll"], seed=7)
    agent.q[("pet", 3)] = {"buy": 2.0, "roll": 3.0}
    agent.q[b"board"] = {"buy": 1.0, "roll": -1.0}
    for _ in range(8):
        agent.choose_action("state")
    path = tmp_path / "nested" / "agent.json"
    agent.save(path)
    clone = QLearningAgent([]).load(path)
    assert dict(clone.q) == dict(agent.q)
    assert [clone.choose_action("state") for _ in range(30)] == [agent.choose_action("state") for _ in range(30)]


def test_checkpoint_encoder_mismatch_is_reported(tmp_path):
    def encoder(state):
        return "state"

    encoder.version = 1
    path = tmp_path / "agent.json"
    QLearningAgent([0], state_fn=encoder).save(path)
    encoder.version = 2
    with pytest.raises(ValueError, match="state encoder"):
        QLearningAgent([0], state_fn=encoder).load(path)
    with pytest.raises(ValueError, match="state encoder"):
        QLearningAgent([0]).load(path)


def test_legacy_checkpoint_remains_loadable(tmp_path):
    import json
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"actions": [0, 1], "q": {"[1, 2]": [3.0, 4.0]}}))
    assert QLearningAgent([]).load(path).q[(1, 2)] == {0: 3.0, 1: 4.0}


def test_invalid_checkpoint_does_not_partially_replace_agent(tmp_path):
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"actions": [0, 1], "q": {'"state"': [3.0]}}))
    agent = QLearningAgent(["original"])
    with pytest.raises(ValueError, match="one finite value"):
        agent.load(path)
    assert agent.actions == ["original"]
