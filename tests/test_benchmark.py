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


def test_report_repeats_and_uses_same_episode_seeds():
    from benchmark import run_report

    a = run_report(episodes=15, eval_episodes=4, seed=7, eval_seed=90)
    b = run_report(episodes=15, eval_episodes=4, seed=7, eval_seed=90)
    assert a == b
    for row in a["results"]:
        assert [ep["seed"] for ep in row["episodes"]] == [90, 91, 92, 93]
        low, high = row["win_rate_ci95"]
        assert low <= row["win_rate"] <= high


def test_main_exports_json(tmp_path):
    import json

    path = tmp_path / "reports/benchmark.json"
    main(["--episodes", "5", "--eval-episodes", "3", "--output", str(path)])
    report = json.loads(path.read_text())
    assert len(report["results"]) == 3
    assert report["config"]["evaluation_episodes"] == 3


def test_report_uses_requested_tabular_step_budget(monkeypatch):
    from benchmark import run_report
    from agent import QLearningAgent

    captured = []
    def train(self, env, **kwargs):
        captured.append(kwargs["max_steps"])
        return []
    monkeypatch.setattr(QLearningAgent, "train", train)
    run_report(episodes=2, eval_episodes=1, max_steps=7)
    assert captured == [7]
