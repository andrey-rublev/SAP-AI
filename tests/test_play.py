"""Tests for the offline (dry-run) game loop in play.py."""
import pytest

from sap_ai.sim.heuristic import HeuristicAgent
from sap_ai.sim.play import load_policy, main, parse_args, run_dry


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


def test_simulator_reports_external_cutoffs():
    infos = main(["--quiet", "--max-steps", "1"])
    assert len(infos) == 1
    assert infos[0]["truncated"] and infos[0]["cutoff"]
    assert not infos[0]["terminated"]
    assert infos[0]["outcome"] == "truncated"
    assert infos[0]["steps"] == 1


@pytest.mark.parametrize("argv", [
    ["--live"], ["--episodes", "0"], ["--max-steps", "-1"], ["--seed", "-1"],
])
def test_invalid_cli_inputs_rejected(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_dqn_requires_a_checkpoint():
    with pytest.raises(SystemExit, match="checkpoint"):
        load_policy(parse_args(["--policy", "dqn"]))


def test_dqn_checkpoint_load_and_legacy_observation_playback(tmp_path):
    pytest.importorskip("torch")
    from sap_ai.sim.dqn import DQNAgent
    path = tmp_path / "legacy-dim.pt"
    DQNAgent(10, 6, hidden=16, device="cpu", seed=0).save(path)
    policy = load_policy(parse_args(["--policy", "dqn", "--checkpoint", str(path)]))
    assert policy.obs_dim == 10
    result = run_dry(policy, 1, 3, verbose=False)
    assert len(result) == 1 and result[0]["steps"] == 3


def test_console_entrypoint_does_not_return_result_as_exit_status(monkeypatch):
    import sap_ai.sim.play as play
    monkeypatch.setattr(play, "main", lambda: [{"outcome": "won"}])
    assert play.cli() is None
