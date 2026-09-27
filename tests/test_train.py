"""Optional combat adapter contract tests; no external engine required."""
import numpy as np
import pytest

from train import ACTIONS, SAPEnvironment


class Team:
    def __init__(self, pets):
        self.pets = list(pets)


class Battle:
    winner = 0

    def __init__(self, team, enemy):
        assert len(team.pets) <= 5
        assert enemy.pets == ["sheep", "tiger"]

    def battle(self):
        np.random.random()  # upstream uses the global RNG
        return self.winner


def test_buys_use_team_constructor_and_have_costs():
    env = SAPEnvironment(engine=(Team, Battle))
    obs, info = env.reset(seed=1)
    assert info["action_mask"].all()
    obs, reward, term, trunc, info = env.step("buy_0")
    assert len(env.team.pets) == 1
    assert env.gold == 12
    assert env.shop[0] == 0
    assert not info["action_mask"][0]
    assert not term and not trunc
    assert reward < 0  # no positive shopping loop
    assert len(obs) == 9


@pytest.mark.parametrize("winner, reward", [(0, 1), (1, -1), (2, 0)])
def test_battle_results_and_global_rng_isolation(monkeypatch, winner, reward):
    monkeypatch.setattr(Battle, "winner", winner)
    env = SAPEnvironment(engine=(Team, Battle))
    env.reset(seed=10)
    state = np.random.get_state()
    _, got, term, trunc, info = env.step("battle")
    assert got == reward and term and not trunc
    assert not info["action_mask"].any()
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    with pytest.raises(RuntimeError, match="reset"):
        env.step("battle")


def test_drill_cannot_loop_forever_and_seeds_repeat():
    env = SAPEnvironment(max_steps=2, engine=(Team, Battle))
    initial, _ = env.reset(seed=7)
    env.step("roll")
    _, _, term, trunc, info = env.step("roll")
    assert not term and trunc
    assert info["truncation_reason"] == "step_limit"
    assert env.reset(seed=7)[0] == initial


def test_agent_can_train_optional_drill():
    from agent import QLearningAgent

    env = SAPEnvironment(engine=(Team, Battle))
    agent = QLearningAgent(ACTIONS, seed=3)
    history = agent.train(env, num_episodes=5, seed=3)
    assert len(history) == 5


def test_upstream_sapai_integration_when_installed():
    pytest.importorskip("sapai")
    env = SAPEnvironment()
    trajectories = []
    for _ in range(2):
        obs, _ = env.reset(seed=99)
        trajectory = [obs]
        for action in ("buy_0", "buy_1", "buy_2", "battle"):
            obs, reward, term, trunc, info = env.step(action)
            trajectory.append((obs, reward, term, trunc))
        assert term and not trunc
        assert info["battle_winner"] in (0, 1, 2)
        trajectories.append(trajectory)
    assert trajectories[0] == trajectories[1]
