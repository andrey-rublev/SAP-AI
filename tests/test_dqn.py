"""Mechanics tests for the DQN agent (kept fast; no performance assertions)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dqn import DQNAgent, ReplayBuffer
from game import SuperAutoPetsEnv


@pytest.fixture(scope="module", autouse=True)
def small_network_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def env():
    return SuperAutoPetsEnv()


@pytest.fixture
def agent(env):
    return DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        obs_scale=env.observation_space.high,
        batch_size=8,
        seed=0,
    )


def test_choose_action_in_range(agent, env):
    obs, _ = env.reset(seed=0)
    assert agent.choose_action(obs) in range(6)
    assert agent.choose_action(obs, greedy=True) in range(6)


def test_choose_action_respects_mask(agent, env):
    obs, _ = env.reset(seed=0)
    only_end_turn = [False, False, False, False, False, True]
    for _ in range(20):
        assert agent.choose_action(obs, mask=only_end_turn) == 5
    assert agent.choose_action(obs, greedy=True, mask=only_end_turn) == 5


def test_learn_none_until_buffer_fills(agent, env):
    assert agent.learn() is None
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        a = agent.choose_action(obs)
        nxt, r, term, trunc, _ = env.step(a)
        agent.remember(obs, a, r, nxt, term or trunc)
        obs = nxt if not (term or trunc) else env.reset()[0]
    assert isinstance(agent.learn(), float)


def test_train_returns_history_and_decays_epsilon(agent, env):
    history = agent.train(env, num_episodes=8, max_steps=80)
    assert len(history) == 8
    assert agent.epsilon < 1.0


def test_save_load_reproduces_q_values(agent, env, tmp_path):
    agent.train(env, num_episodes=5, max_steps=80)
    path = tmp_path / "dqn.pt"
    agent.save(path)

    clone = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        obs_scale=env.observation_space.high,
    ).load(path)

    obs, _ = env.reset(seed=3)
    with torch.no_grad():
        q_original = agent.q(agent._tensor(obs).unsqueeze(0))
        q_loaded = clone.q(clone._tensor(obs).unsqueeze(0))
    assert torch.allclose(q_original, q_loaded)
    assert np.allclose(clone.obs_scale, env.observation_space.high)


def test_replay_buffer_capacity_and_sample():
    buf = ReplayBuffer(capacity=5, seed=0)
    for i in range(10):
        buf.push([i], i % 6, float(i), [i + 1], False, [True] * 6)
    assert len(buf) == 5  # oldest entries evicted
    s, a, r, ns, d, m = buf.sample(3)
    assert len(s) == len(a) == len(r) == len(m) == 3


def constant_outputs(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.net[-1].bias.copy_(torch.tensor(values))


@pytest.mark.parametrize("double_dqn", [False, True])
@pytest.mark.parametrize("terminated", [False, True])
def test_empty_successor_mask_has_finite_reward_only_target(double_dqn, terminated):
    agent = DQNAgent(2, 2, batch_size=1, hidden=8, seed=1, device="cpu", double_dqn=double_dqn)
    constant_outputs(agent.q, [0.0, 0.0])
    constant_outputs(agent.target, [999.0, 1000.0])
    agent.remember([0, 0], 0, 3.0, [1, 1], terminated, [False, False])
    # Huber(0, 3) = 3 - 0.5, regardless of termination or target output.
    assert agent.learn() == pytest.approx(2.5)
    assert all(torch.isfinite(p).all() for p in agent.q.parameters())


def test_double_dqn_selects_online_action_and_evaluates_target_network():
    agent = DQNAgent(2, 3, gamma=0.5, batch_size=1, hidden=8, seed=1, device="cpu")
    constant_outputs(agent.q, [2.0, 1.0, 1000.0])
    constant_outputs(agent.target, [4.0, 100.0, 2000.0])
    agent.remember([0, 0], 0, 1.0, [1, 1], False, [True, True, False])
    # Online legal argmax is 0; target = 1 + .5*4 = 3; current = 2.
    # A target-network max or an unmasked online max would produce a large error.
    assert agent.learn() == pytest.approx(0.5)


@pytest.mark.parametrize("greedy", [False, True])
def test_policy_rejects_empty_action_mask(greedy):
    agent = DQNAgent(2, 2, hidden=8)
    with pytest.raises(ValueError, match="no legal actions"):
        agent.choose_action([0, 0], greedy=greedy, mask=[False, False])


def test_replay_owns_input_memory():
    agent = DQNAgent(2, 2, hidden=8)
    state, successor, mask = np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([True, False])
    agent.remember(state, 0, 1.0, successor, False, mask)
    state[:] = 99
    successor[:] = 99
    mask[:] = False
    saved = agent.buffer.buffer[0]
    assert np.array_equal(saved[0], [1, 2])
    assert np.array_equal(saved[3], [3, 4])
    assert np.array_equal(saved[5], [True, False])


@pytest.mark.parametrize("classic", [False, True])
@pytest.mark.parametrize("terminated", [False, True])
def test_train_preserves_truncation_bootstrap_semantics(classic, terminated):
    class Env:
        def reset(self):
            return [0, 0], {"action_mask": [True, False]}

        def step(self, action):
            info = {"action_mask": [False, True], "TimeLimit.truncated": not terminated}
            if classic:
                return [1, 1], 1.0, True, info
            return [1, 1], 1.0, terminated, not terminated, info

    agent = DQNAgent(2, 2, hidden=8, seed=1)
    assert agent.train(Env(), num_episodes=1) == [1.0]
    assert agent.buffer.buffer[0][4] is terminated
    assert np.array_equal(agent.buffer.buffer[0][5], [False, True])


def test_seeded_initialization_does_not_reset_global_rngs():
    np.random.seed(33)
    torch.manual_seed(33)
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    first = DQNAgent(2, 2, hidden=8, seed=7, device="cpu")
    second = DQNAgent(2, 2, hidden=8, seed=7, device="cpu")
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert all(torch.equal(a, b) for a, b in zip(first.q.parameters(), second.q.parameters()))


def test_full_checkpoint_restores_architecture_optimizer_replay_and_rng(tmp_path):
    agent = DQNAgent(2, 3, obs_scale=[2, 4], hidden=11, batch_size=2, buffer_size=8,
                     gamma=0.8, epsilon=0.4, seed=17, target_update=3, device="cpu")
    for i in range(5):
        agent.remember([i, 1], i % 3, i * 0.5, [i + 1, 1], False, [True, False, True])
    agent.learn()
    agent.episodes_trained = 9
    path = tmp_path / "nested" / "model.pt"
    agent.save(path)
    # The complete payload can be deserialized with safe mode explicitly enabled.
    assert torch.load(path, weights_only=True)["format_version"] == 2
    clone = DQNAgent.from_checkpoint(path, device="cpu")
    assert clone._config() == agent._config()
    assert clone.episodes_trained == 9
    assert clone.learning_steps == agent.learning_steps
    assert len(clone.buffer) == len(agent.buffer)
    assert np.array_equal(clone.obs_scale, [2, 4])
    assert all(torch.equal(a, b) for a, b in zip(agent.target.parameters(), clone.target.parameters()))
    assert [clone.choose_action([1, 1]) for _ in range(20)] == [agent.choose_action([1, 1]) for _ in range(20)]
    # Same replay RNG, target weights and Adam moments produce the same next update.
    assert clone.learn() == pytest.approx(agent.learn())
    assert all(torch.equal(a, b) for a, b in zip(agent.q.parameters(), clone.q.parameters()))


def test_load_rebuilds_saved_architecture(tmp_path):
    original = DQNAgent(3, 4, hidden=11, seed=1)
    path = tmp_path / "model.pt"
    original.save(path)
    clone = DQNAgent(2, 2, hidden=8).load(path)
    assert (clone.obs_dim, clone.n_actions, clone.hidden) == (3, 4, 11)
    assert clone.choose_action([0, 0, 0]) in range(4)


def test_legacy_checkpoint_infers_hidden_width(tmp_path):
    original = DQNAgent(10, 6, hidden=17, seed=1)
    path = tmp_path / "legacy.pt"
    torch.save({"model": original.q.state_dict(), "obs_dim": 10, "n_actions": 6,
                "obs_scale": original.obs_scale.tolist()}, path)
    clone = DQNAgent.from_checkpoint(path, device="cpu")
    assert (clone.obs_dim, clone.n_actions, clone.hidden) == (10, 6, 17)
    assert all(torch.equal(a, b) for a, b in zip(original.q.parameters(), clone.q.parameters()))


def test_constructor_does_not_modify_scale_array():
    scale = np.array([0.0, 2.0], dtype=np.float32)
    agent = DQNAgent(2, 2, obs_scale=scale, hidden=8)
    assert np.array_equal(scale, [0, 2])
    assert np.array_equal(agent.obs_scale, [1, 2])


def test_training_is_reproducible_with_seed():
    env1, env2 = SuperAutoPetsEnv(), SuperAutoPetsEnv()
    config = dict(obs_dim=env1.observation_space.shape[0], n_actions=env1.action_space.n,
                  obs_scale=env1.observation_space.high, hidden=8, batch_size=4, seed=12, device="cpu")
    first, second = DQNAgent(**config), DQNAgent(**config)
    assert first.train(env1, num_episodes=2, max_steps=20) == second.train(env2, num_episodes=2, max_steps=20)
    assert all(torch.equal(a, b) for a, b in zip(first.q.parameters(), second.q.parameters()))
