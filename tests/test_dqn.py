"""Mechanics tests for the DQN agent (kept fast; no performance assertions)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dqn import DQNAgent, ReplayBuffer
from game import SuperAutoPetsEnv


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
        buf.push([i], i % 6, float(i), [i + 1], False)
    assert len(buf) == 5  # oldest entries evicted
    s, a, r, ns, d = buf.sample(3)
    assert len(s) == len(a) == len(r) == 3
