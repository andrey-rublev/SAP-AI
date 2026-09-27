"""Tests for the offline (dry-run) game loop in play.py."""
import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

from heuristic import HeuristicAgent
from play import load_policy, main, parse_args, run_dry, run_live


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


def test_simulator_is_default_and_does_not_import_desktop(monkeypatch):
    import autogui
    monkeypatch.setattr(autogui, "_dependency", MagicMock(side_effect=AssertionError("desktop loaded")))
    infos = main(["--quiet", "--max-steps", "1"])
    assert len(infos) == 1
    assert infos[0]["truncated"] and infos[0]["cutoff"]
    assert not infos[0]["terminated"]
    assert infos[0]["outcome"] == "truncated"
    assert infos[0]["steps"] == 1


@pytest.mark.parametrize("argv", [
    ["--live"], ["--live", "--dry-run"], ["--episodes", "0"],
    ["--max-steps", "-1"], ["--delay", "nan"], ["--delay", "-1"],
    ["--turn", "-1"], ["--preview"], ["--seed", "-1"],
])
def test_invalid_cli_inputs_rejected(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.fixture
def live_bridge(monkeypatch):
    import autogui
    calls = SimpleNamespace(configure=MagicMock(), read=MagicMock(
        return_value=np.array([10, 0, 1, 2, 3, 0, 0, 0, 0, 0], dtype=np.float32)),
        act=MagicMock(), disable=MagicMock())
    monkeypatch.setattr(autogui, "configure_layout", calls.configure)
    monkeypatch.setattr(autogui, "read_board", calls.read)
    monkeypatch.setattr(autogui, "perform_action", calls.act)
    monkeypatch.setattr(autogui, "disable_control", calls.disable)
    return calls


def test_live_stops_at_end_turn_instead_of_clicking_through_battle(live_bridge):
    policy = SimpleNamespace(choose_action=MagicMock(return_value=5))
    assert run_live(policy, 100, 0, layout_path="example.json", turn=2) == [5]
    live_bridge.configure.assert_called_once_with("example.json", enable_control=True)
    live_bridge.read.assert_called_once_with(2)
    live_bridge.act.assert_called_once()
    live_bridge.disable.assert_called_once()


def test_live_preview_reads_once_and_never_acts(live_bridge):
    assert run_live(HeuristicAgent(), 100, 0, layout_path="example.json", preview=True) == [2]
    live_bridge.configure.assert_called_once_with("example.json", enable_control=False)
    live_bridge.read.assert_called_once()
    live_bridge.act.assert_not_called()


def test_live_perception_failure_stops_before_policy_or_control(live_bridge):
    import autogui
    live_bridge.read.side_effect = autogui.PerceptionError("blank board")
    policy = SimpleNamespace(choose_action=MagicMock())
    with pytest.raises(autogui.PerceptionError):
        run_live(policy, 10, 0, layout_path="example.json")
    policy.choose_action.assert_not_called()
    live_bridge.act.assert_not_called()
    live_bridge.disable.assert_called_once()


def test_live_rejects_incompatible_dqn_without_reading(live_bridge):
    policy = SimpleNamespace(obs_dim=13)
    with pytest.raises(ValueError, match="metadata"):
        run_live(policy, 10, 0, layout_path="example.json")
    live_bridge.configure.assert_not_called()
    live_bridge.read.assert_not_called()


def test_live_rejects_policy_action_even_if_policy_ignores_mask(live_bridge):
    policy = SimpleNamespace(choose_action=MagicMock(return_value=4))
    with pytest.raises(ValueError, match="unsupported"):
        run_live(policy, 10, 0, layout_path="example.json")
    live_bridge.act.assert_not_called()


def test_dqn_requires_a_checkpoint():
    with pytest.raises(SystemExit, match="checkpoint"):
        load_policy(parse_args(["--policy", "dqn"]))


def test_dqn_checkpoint_load_and_legacy_observation_playback(tmp_path):
    pytest.importorskip("torch")
    from dqn import DQNAgent
    path = tmp_path / "legacy-dim.pt"
    DQNAgent(10, 6, hidden=16, device="cpu", seed=0).save(path)
    policy = load_policy(parse_args(["--policy", "dqn", "--checkpoint", str(path)]))
    assert policy.obs_dim == 10
    result = run_dry(policy, 1, 3, verbose=False)
    assert len(result) == 1 and result[0]["steps"] == 3


def test_console_entrypoint_does_not_return_result_as_exit_status(monkeypatch):
    import play
    monkeypatch.setattr(play, "main", lambda: [{"outcome": "won"}])
    assert play.cli() is None
