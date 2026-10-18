"""One explicit completed-client bootstrap with shared budgets and mocked IO."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import desktop
from desktop_session import SessionResult
from desktop_state import Action, Board, Phase
from tools.export_desktop_replay import export_session


def terminal(actions=0, polls=1):
    return SessionResult(actions, actions, polls, "result", Board(Phase.RESULT))


@pytest.fixture
def probe(tmp_path, monkeypatch):
    args = desktop.parse_args(["play", "--restart-completed", "--log", str(tmp_path / "session.jsonl"),
                               "--max-polls", "20", "--max-actions", "5"])
    old, new = Mock(), Mock()
    old.close_for_restart.return_value = True
    perceptor = Mock()
    perceptor.observe.return_value = Board(Phase.RESULT)
    sessions = Mock(side_effect=[terminal(), terminal(actions=2, polls=5)])
    monkeypatch.setattr(desktop, "run_desktop_session", sessions)
    return SimpleNamespace(args=args, profile=SimpleNamespace(image_size=(100, 100)),
        old=old, new=new, perceptor=perceptor, sessions=sessions, open=Mock(return_value=new),
        stopped=False, info={"first_recognized_phase": "result"}, root=tmp_path)


def run(probe):
    return desktop.run_with_completed_restart(probe.args, probe.profile, probe.perceptor, probe.old,
        lambda: probe.stopped, probe.open, probe.info)


def test_bootstrap_preserves_zero_input_receipt_and_shares_remaining_poll_budget(probe):
    result, window, info = run(probe)
    assert (result.actions, result.acknowledgments, result.polls) == (2, 2, 5)
    assert window is probe.new and info["attempts"] == 1 and info["polls"] == 2
    remaining = probe.sessions.call_args_list[1].args[0]
    assert (remaining.max_actions, remaining.max_polls) == (5, 18)
    assert probe.info["first_recognized_phase"] is None
    probe.old.close_for_restart.assert_called_once_with((100, 100), timeout=90)
    probe.open.assert_called_once()
    events = [json.loads(line) for line in Path(info["log"]).read_text().splitlines()]
    assert events[0]["event"] == "bootstrap_started"
    assert events[0]["receipt"] == terminal().to_dict()
    assert events[-1]["event"] == "bootstrap_finished" and events[-1]["attempts"] == 1
    assert not Path(probe.args.log).exists()  # Mock sessions; bootstrap events use their own sidecar.


@pytest.mark.parametrize("first", [
    terminal(actions=1), SessionResult(0, 0, 2, "stopped"),
    SessionResult(0, 0, 2, "result", Board(Phase.UNKNOWN)),
    SessionResult(0, 0, 2, "result", Board(Phase.RESULT), error="capture failed"),
    SessionResult(0, 0, 2, "result", Board(Phase.RESULT), pending_action=Action("roll")),
])
def test_only_an_initial_clean_zero_action_terminal_can_bootstrap(probe, first):
    probe.sessions.side_effect = [first]
    assert run(probe)[0] is first
    probe.old.capture.assert_not_called()
    probe.old.close_for_restart.assert_not_called()
    probe.open.assert_not_called()


def test_restored_terminal_reports_diagnostic_without_second_close(probe):
    probe.sessions.side_effect = [terminal(), terminal()]
    result, _, info = run(probe)
    assert result.reason == "empty_terminal" and "no second bootstrap" in result.error
    assert info["attempts"] == 1 and info["polls"] + result.polls == 3
    probe.old.close_for_restart.assert_called_once()
    probe.new.close_for_restart.assert_not_called()
    probe.open.assert_called_once()


def test_insufficient_poll_budget_prevents_fresh_capture_close_or_relaunch(probe):
    probe.args.max_polls = 2
    result, _, info = run(probe)
    assert result.reason == "max_polls" and info["polls"] == 1 and info["attempts"] == 0
    probe.old.capture.assert_not_called()
    probe.old.close_for_restart.assert_not_called()
    probe.open.assert_not_called()


@pytest.mark.parametrize("when", ["initial_receipt", "fresh_check", "close"])
def test_stop_control_is_honored_at_every_bootstrap_boundary(probe, when):
    if when == "initial_receipt":
        probe.stopped = True
    elif when == "fresh_check":
        def observe(frame):
            probe.stopped = True
            return Board(Phase.RESULT)
        probe.perceptor.observe.side_effect = observe
    else:
        def close(*args, **kwargs):
            probe.stopped = True
            return True
        probe.old.close_for_restart.side_effect = close
    result, _, _ = run(probe)
    assert result.reason == "stopped"
    probe.open.assert_not_called()
    if when != "close":
        probe.old.close_for_restart.assert_not_called()


@pytest.mark.parametrize("failure", ["changed", "focus", "close", "launch"])
def test_bootstrap_errors_are_retained_without_control_retries(probe, failure):
    if failure == "changed":
        probe.perceptor.observe.return_value = Board(Phase.SHOP)
    elif failure == "focus":
        probe.old.capture.side_effect = RuntimeError("lost focus")
    elif failure == "close":
        probe.old.close_for_restart.side_effect = RuntimeError("normal shutdown failed")
    else:
        probe.open.side_effect = RuntimeError("launch failed")
    result, _, info = run(probe)
    assert result.reason == "bootstrap_error" and result.error
    assert probe.old.close_for_restart.call_count <= 1 and probe.open.call_count <= 1
    assert probe.sessions.call_count == 1
    assert info["attempts"] == int(failure in {"close", "launch"})


def test_actual_arena_exception_after_dispatch_propagates_without_fabricated_zero_receipt(probe):
    dispatches = []
    attempts = 0
    def runner(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return terminal()
        dispatches.append(Action("open_play"))
        raise OSError("actual arena log failed after dispatch")
    probe.sessions.side_effect = runner
    with pytest.raises(OSError, match="after dispatch"):
        run(probe)
    assert dispatches == [Action("open_play")] and probe.sessions.call_count == 2
    probe.old.close_for_restart.assert_called_once()
    probe.open.assert_called_once()
    probe.new.close_for_restart.assert_not_called()
    path = next(probe.root.glob("*-bootstrap-*.jsonl"))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["receipt"] == terminal().to_dict()
    assert events[-1]["result"] is None
    assert events[-1]["actual_session_counts_known"] is False
    assert "after dispatch" in events[-1]["actual_session_error"]
    assert not any(row.get("result", {}).get("reason") == "bootstrap_error"
                   for row in events if isinstance(row.get("result"), dict))


def test_batch_bootstrap_does_not_count_old_result_or_consume_normal_restart_allowance(probe):
    probe.args.arenas = 2
    probe.sessions.side_effect = [terminal(), terminal(actions=2, polls=5), terminal(actions=2, polls=5)]
    opens = Mock(side_effect=[probe.old, probe.new, Mock()])
    probe.new.close_for_restart.return_value = True
    summary = desktop.run_arena_batch(probe.args, probe.profile, probe.perceptor, opens, lambda: False)
    assert (summary["completed_arenas"], summary["actions"], summary["acknowledgments"], summary["polls"]) == (2, 4, 4, 13)
    assert summary["bootstrap_attempts"] == summary["max_bootstrap_attempts"] == 1
    assert summary["bootstrap_polls"] == 2
    assert summary["restart_attempts"] == summary["max_restart_attempts"] == 1
    assert len(summary["sessions"]) == 2 and opens.call_count == 3
    assert probe.sessions.call_args_list[1].args[0].max_polls == 18
    assert probe.sessions.call_args_list[2].args[0].max_polls == 12


def test_default_and_cmd_override_keep_bootstrap_opt_in():
    assert desktop.parse_args(["play"]).restart_completed is False
    assert desktop.parse_args(["play", "--restart-completed", "--no-restart-completed"]).restart_completed is False
    launcher = Path(desktop.__file__).with_name("play.cmd").read_text()
    assert "desktop.py play --restart-completed %*" in launcher


def test_cli_bootstrap_retains_two_exportable_receipts_and_runs_real_session(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "desktop_control_lease", Mock(side_effect=nullcontext))
    profile = Mock(image_size=(100, 100))
    perceptor = Mock()
    perceptor.observe.return_value = Board(Phase.RESULT)
    monkeypatch.setattr(desktop.VisionProfile, "load", Mock(return_value=profile))
    monkeypatch.setattr(desktop, "Perceptor", Mock(return_value=perceptor))
    monkeypatch.setattr(desktop, "validate_profile_assets", Mock())
    monkeypatch.setattr(desktop, "validate_live_dependencies", Mock())
    old, new = Mock(), Mock()
    old.stop_requested.return_value = new.stop_requested.return_value = False
    old.close_for_restart.return_value = True
    connect = Mock(side_effect=[old, new])
    monkeypatch.setattr(desktop, "connect_game_window", connect)
    frames = iter([[Board(Phase.RESULT)], [Board(Phase.MAIN_MENU)] * 2 + [Board(Phase.PLAY_MENU)] * 2 + [Board(Phase.RESULT)]])
    runtimes = []
    def runtime(*args, **kwargs):
        observations = iter(next(frames))
        result = SimpleNamespace(observe=lambda: next(observations), act=Mock(), last_frame=None,
            phase_actions=Mock(return_value={Phase.MAIN_MENU: Action("open_play")}))
        runtimes.append(result)
        return result
    monkeypatch.setattr(desktop, "DesktopRuntime", runtime)
    log = tmp_path / "session.jsonl"
    desktop.main(["play", "--restart-completed", "--max-polls", "8", "--log", str(log),
                  "--stop-file", str(tmp_path / "STOP")])
    first, actual = export_session(log.read_text(), session=1), export_session(log.read_text())
    assert first["expected"]["actions"] == first["expected"]["acknowledgments"] == 0
    assert first["expected"]["polls"] == 1
    assert actual["expected"] == {"actions": 1, "acknowledgments": 1, "polls": 5,
                                    "reason": "result", "pending_action": None}
    assert len(runtimes) == connect.call_count == 2
    runtimes[0].act.assert_not_called()
    runtimes[1].act.assert_called_once_with(Action("open_play"))
    old.close_for_restart.assert_called_once()
    new.close_for_restart.assert_not_called()


@pytest.mark.parametrize("flags", [["--no-launch"], ["--window", "custom"]])
def test_unrestartable_bootstrap_configuration_is_rejected_before_io(flags):
    with pytest.raises(SystemExit):
        desktop.parse_args(["play", "--restart-completed", *flags])
