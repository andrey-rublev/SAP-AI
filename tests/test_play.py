"""Tests for the offline (dry-run) game loop in play.py."""
import pytest

from heuristic import HeuristicAgent
from play import load_policy, main, parse_args, run_dry


def test_dry_run_returns_one_info_per_episode():
    infos = main(["--dry-run", "--episodes", "3", "--max-steps", "300"])
    assert len(infos) == 3
    assert all("wins" in i and "lives" in i for i in infos)


def test_heuristic_solves_env_in_dry_run():
    infos = run_dry(HeuristicAgent(), episodes=5, max_steps=300, verbose=False)
    assert all(i["wins"] >= 10 for i in infos)


def test_qtable_policy_requires_a_path():
    with pytest.raises(SystemExit):
        load_policy(parse_args(["--dry-run", "--policy", "qtable"]))
