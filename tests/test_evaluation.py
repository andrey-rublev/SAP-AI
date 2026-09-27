"""Regression tests for fair, repeatable evaluations and experiment outputs."""
import json

import numpy as np
import pytest

from agent import QLearningAgent
from evaluation import evaluate_policy
from game import SuperAutoPetsEnv
from main import compact_state, main, parse_args


def test_random_evaluation_is_repeatable_and_legal():
    class CheckedEnv(SuperAutoPetsEnv):
        def step(self, action):
            assert self._action_mask()[action]
            return super().step(action)

    first = evaluate_policy(CheckedEnv(), episodes=8, seed=50)
    second = evaluate_policy(CheckedEnv(), episodes=8, seed=50)
    assert first == second
    assert [ep.seed for ep in first.episodes] == list(range(50, 58))


def test_evaluation_preserves_policy_rng_and_table():
    agent = QLearningAgent(range(6), state_fn=compact_state, seed=42)
    before = agent._rng.getstate()
    first = evaluate_policy(SuperAutoPetsEnv(), agent, episodes=5, seed=20)
    assert agent._rng.getstate() == before
    assert not agent.q
    assert evaluate_policy(SuperAutoPetsEnv(), agent, episodes=5, seed=20) == first


def test_step_budget_is_reported_as_truncation():
    result = evaluate_policy(SuperAutoPetsEnv(), episodes=3, max_steps=1)
    assert result.truncations == 3
    assert result.win_rate == 0
    assert all(ep.stop_reason == "evaluation_step_limit" for ep in result.episodes)
    assert all(not ep.terminated for ep in result.episodes)
    low, high = result.win_rate_ci95
    assert low == pytest.approx(0)
    assert 0 < high < 1


@pytest.mark.parametrize("counts", [{"episodes": 0}, {"max_steps": 0}, {"seed": -1}])
def test_invalid_evaluation_settings_rejected(counts):
    with pytest.raises(ValueError):
        evaluate_policy(SuperAutoPetsEnv(), **counts)


def test_compact_state_preserves_shop_action_identity():
    obs = np.array([10, 0, 1, 6, 3, 0, 0, 0, 0, 0], dtype=np.float32)
    permuted = obs.copy()
    permuted[2:5] = [6, 1, 3]
    assert compact_state(obs) != compact_state(permuted)


def test_compact_state_distinguishes_combine_from_placement():
    obs = np.array([10, 1, 2, 4, 6, 1, 3, 4, 0, 0], dtype=np.float32)
    changed = obs.copy()
    changed[5:10] = [1, 2, 5, 0, 0]
    assert compact_state(obs) != compact_state(changed)


def test_main_writes_reproducible_artifacts(tmp_path):
    model, report = tmp_path / "models/q.json", tmp_path / "reports/run.json"
    args = ["--episodes", "10", "--eval-episodes", "4", "--seed", "7",
            "--output", str(model), "--report", str(report)]
    first = main(args)
    first_model = model.read_text()
    first_report = report.read_text()
    assert main(args) == first
    assert model.read_text() == first_model
    assert report.read_text() == first_report
    assert json.loads(first_report)["tabular"]["episode_count"] == 4


@pytest.mark.parametrize("args", [["--episodes", "0"], ["--eval-episodes", "-1"],
                                   ["--seed", "-1"], ["--epsilon-decay", "2"]])
def test_invalid_cli_counts_fail_early(args):
    with pytest.raises(SystemExit):
        parse_args(args)


def test_resume_rejects_incompatible_action_order(tmp_path):
    path = tmp_path / "reordered.json"
    QLearningAgent([5, 1, 2, 3, 4, 0], state_fn=compact_state).save(path)
    with pytest.raises(ValueError, match="action order"):
        main(["--resume", str(path), "--episodes", "1", "--eval-episodes", "1", "--no-save"])
