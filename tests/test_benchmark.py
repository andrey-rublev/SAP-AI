"""Tests for the benchmark harness (small, fast; DQN excluded)."""
from benchmark import main, run


def test_run_returns_three_agent_rows():
    rows = run(episodes=150, eval_episodes=30, seed=0, include_dqn=False, dqn_episodes=0)
    names = [r[0] for r in rows]
    assert names == ["Random", "Heuristic", "Tabular Q"]
    for _, rate, wins in rows:
        assert 0.0 <= rate <= 1.0
        assert wins >= 0.0


def test_heuristic_outranks_random():
    rows = dict((r[0], r[1]) for r in run(150, 30, 0, False, 0))
    assert rows["Heuristic"] > rows["Random"]


def test_main_returns_rows():
    rows = main(["--episodes", "150", "--eval-episodes", "30"])
    assert len(rows) == 3
