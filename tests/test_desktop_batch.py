"""Bounded multi-arena collection without native desktop access."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import desktop
from desktop_runtime import DesktopUnavailable
from desktop_session import SessionResult
from desktop_state import Action, Board, PetSlot, Phase


def arguments(tmp_path, *flags):
    return ["play", "--arenas", "3", "--log", str(tmp_path / "session.jsonl"),
            "--stop-file", str(tmp_path / "STOP"), "--last-frame", str(tmp_path / "last.png"),
            "--max-actions", "10", "--max-polls", "30", *flags]


def terminal(actions=2, polls=5):
    return SessionResult(actions, actions, polls, "result", Board(Phase.RESULT))


@pytest.fixture
def batch(tmp_path, monkeypatch):
    args = desktop.parse_args(arguments(tmp_path))
    profile = SimpleNamespace(image_size=(100, 100))
    perceptor = Mock()
    perceptor.observe.return_value = Board(Phase.RESULT)
    windows = [Mock(), Mock(), Mock()]
    for window in windows:
        window.close_for_restart.return_value = True
    open_window = Mock(side_effect=windows)
    sessions = Mock(return_value=terminal())
    monkeypatch.setattr(desktop, "run_desktop_session", sessions)
    return SimpleNamespace(args=args, profile=profile, perceptor=perceptor, windows=windows,
                           open=open_window, sessions=sessions, stopped=False, tmp=tmp_path)


def run(batch):
    return desktop.run_arena_batch(batch.args, batch.profile, batch.perceptor, batch.open,
                                   lambda: batch.stopped)


def saved_summary(batch):
    return json.loads(next(batch.tmp.glob("*-batch-*/summary.json")).read_text())


def test_requested_arenas_end_without_an_extra_close_or_launch(batch):
    Path(batch.args.log).write_text("original append-only log\n")
    summary = run(batch)
    assert (summary["reason"], summary["completed_arenas"], summary["actions"], summary["polls"]) == (
        "completed", 3, 6, 17)
    assert summary["restart_attempts"] == summary["max_restart_attempts"] == 2
    assert batch.open.call_count == batch.sessions.call_count == 3
    for window in batch.windows[:2]:
        window.close_for_restart.assert_called_once_with((100, 100), timeout=90)
    batch.windows[-1].close_for_restart.assert_not_called()
    assert Path(batch.args.log).read_text() == "original append-only log\n"
    assert saved_summary(batch) == summary
    configs = [call.args[0] for call in batch.sessions.call_args_list]
    assert [(config.max_actions, config.max_polls) for config in configs] == [(10, 30), (8, 24), (6, 18)]
    assert len({config.log for config in configs}) == 3
    assert all(Path(config.log).parent.name == f"arena-{index:04d}"
               for index, config in enumerate(configs, 1))


@pytest.mark.parametrize("reason", ["max_actions", "max_polls", "stopped", "unknown_timeout", "policy_stopped"])
def test_only_a_completed_result_can_restart(batch, reason):
    batch.sessions.return_value = SessionResult(0, 0, 4, reason, Board(Phase.UNKNOWN))
    if reason == "unknown_timeout":
        with pytest.raises(RuntimeError, match="desktop batch stopped"):
            run(batch)
    else:
        assert run(batch)["reason"] == reason
    batch.windows[0].close_for_restart.assert_not_called()
    batch.windows[0].capture.assert_not_called()
    batch.open.assert_called_once()


@pytest.mark.parametrize("result", [
    SessionResult(1, 0, 5, "result", Board(Phase.RESULT), pending_action=Action("end_turn")),
    SessionResult(1, 0, 5, "result", Board(Phase.RESULT)),
    SessionResult(1, 1, 5, "result", Board(Phase.RESULT), error="failed input"),
    SessionResult(0, 0, 5, "result", Board(Phase.UNKNOWN)),
    SessionResult(0, 0, 5, "result"),
])
def test_pending_error_or_unverified_terminal_result_never_closes(batch, result):
    batch.sessions.return_value = result
    with pytest.raises(RuntimeError, match="desktop batch stopped"):
        run(batch)
    batch.windows[0].close_for_restart.assert_not_called()
    batch.open.assert_called_once()
    assert saved_summary(batch)["completed_arenas"] == 0


@pytest.mark.parametrize("cap", ["actions", "polls"])
def test_budget_exhausted_at_a_terminal_result_never_closes(batch, cap):
    if cap == "actions":
        batch.args.max_actions = 2
    else:
        batch.args.max_polls = 5
    summary = run(batch)
    assert summary["reason"] == f"max_{cap}"
    assert summary["completed_arenas"] == 1
    batch.windows[0].capture.assert_not_called()
    batch.windows[0].close_for_restart.assert_not_called()
    batch.open.assert_called_once()


def test_fresh_terminal_check_and_next_session_must_both_fit_poll_budget(batch):
    batch.args.max_polls = 6
    assert run(batch)["reason"] == "max_polls"
    batch.windows[0].capture.assert_not_called()
    batch.windows[0].close_for_restart.assert_not_called()


@pytest.mark.parametrize("cap", ["actions", "polls"])
def test_remaining_budgets_are_shared_across_arenas(batch, cap):
    if cap == "actions":
        batch.args.max_actions = 3
        second = SessionResult(1, 1, 3, "max_actions", Board(Phase.SHOP))
    else:
        batch.args.max_polls = 10
        second = SessionResult(1, 1, 4, "max_polls", Board(Phase.SHOP))
    batch.sessions.side_effect = [terminal(), second]
    summary = run(batch)
    assert summary["reason"] == f"max_{cap}"
    assert summary[cap] == getattr(batch.args, f"max_{cap}")
    assert summary["completed_arenas"] == 1 and summary["restart_attempts"] == 1
    second_args = batch.sessions.call_args_list[1].args[0]
    assert getattr(second_args, f"max_{cap}") == (1 if cap == "actions" else 4)
    batch.windows[1].close_for_restart.assert_not_called()
    assert batch.open.call_count == 2


@pytest.mark.parametrize("phase", [Phase.UNKNOWN, Phase.MAIN_MENU, Phase.BATTLE])
def test_changed_terminal_screen_cannot_close_or_relaunch(batch, phase):
    batch.perceptor.observe.return_value = Board(phase)
    with pytest.raises(RuntimeError, match="restart_error"):
        run(batch)
    batch.windows[0].close_for_restart.assert_not_called()
    batch.open.assert_called_once()
    summary = saved_summary(batch)
    assert summary["restart_polls"] == 1 and summary["restart_attempts"] == 0


def test_failed_close_attempt_is_retained_without_retry_or_relaunch(batch):
    batch.windows[0].close_for_restart.side_effect = DesktopUnavailable("normal shutdown rejected")
    with pytest.raises(RuntimeError, match="restart_error"):
        run(batch)
    batch.windows[0].close_for_restart.assert_called_once()
    batch.open.assert_called_once()
    summary = saved_summary(batch)
    assert summary["restart_attempts"] == 1 and summary["completed_arenas"] == 1
    assert "normal shutdown rejected" in summary["error"]
    events = [json.loads(line) for line in (Path(summary["directory"]) / "batch.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events].count("restart_attempted") == 1


def test_cold_reset_cannot_count_the_same_terminal_screen_as_another_arena(batch):
    batch.sessions.side_effect = [terminal(), terminal(actions=0, polls=1)]
    with pytest.raises(RuntimeError, match="desktop batch stopped"):
        run(batch)
    summary = saved_summary(batch)
    assert summary["reason"] == "empty_terminal"
    assert summary["completed_arenas"] == 1 and summary["restart_attempts"] == 1
    assert batch.open.call_count == 2
    batch.windows[1].close_for_restart.assert_not_called()


def test_initial_terminal_client_stops_without_counting_or_closing(batch):
    batch.sessions.return_value = terminal(actions=0, polls=1)
    with pytest.raises(RuntimeError, match="empty_terminal"):
        run(batch)
    summary = saved_summary(batch)
    assert summary["completed_arenas"] == summary["restart_attempts"] == 0
    batch.windows[0].close_for_restart.assert_not_called()
    batch.open.assert_called_once()


@pytest.mark.parametrize("stage", ["session", "fresh_check", "close"])
def test_stopping_between_arenas_never_relaunches(batch, stage):
    def stop_and_return(value):
        batch.stopped = True
        return value
    if stage == "session":
        batch.sessions.side_effect = lambda *args: stop_and_return(terminal())
    elif stage == "fresh_check":
        batch.perceptor.observe.side_effect = lambda frame: stop_and_return(Board(Phase.RESULT))
    else:
        batch.windows[0].close_for_restart.side_effect = lambda *args, **kwargs: stop_and_return(False)
    summary = run(batch)
    assert summary["reason"] == "stopped" and summary["error"] is None
    batch.open.assert_called_once()
    if stage != "close":
        batch.windows[0].close_for_restart.assert_not_called()


@pytest.fixture
def cli(tmp_path, monkeypatch):
    profile = Mock(image_size=(2, 2))
    perceptor = Mock()
    perceptor.observe.return_value = Board(Phase.RESULT)
    monkeypatch.setattr(desktop.VisionProfile, "load", Mock(return_value=profile))
    monkeypatch.setattr(desktop, "Perceptor", Mock(return_value=perceptor))
    monkeypatch.setattr(desktop, "validate_profile_assets", Mock())
    monkeypatch.setattr(desktop, "validate_live_dependencies", Mock())
    windows = [Mock(), Mock(), Mock()]
    for window in windows:
        window.stop_requested.return_value = False
        window.close_for_restart.return_value = True
    connect = Mock(side_effect=windows)
    monkeypatch.setattr(desktop, "connect_game_window", connect)
    runtimes = []
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = [Board(Phase.MAIN_MENU)] * 2 + [Board(Phase.PLAY_MENU)] * 2 + [Board(Phase.RESULT)]
    def runtime(*args, **kwargs):
        boards = iter(frames)
        created = SimpleNamespace(last_frame=None, act=Mock(),
                                  phase_actions=Mock(return_value={Phase.MAIN_MENU: Action("open_play")}))
        def observe():
            created.last_frame = frame
            return next(boards)
        created.observe = Mock(side_effect=observe)
        runtimes.append(created)
        return created
    monkeypatch.setattr(desktop, "DesktopRuntime", runtime)
    return SimpleNamespace(profile=profile, windows=windows, connect=connect, runtimes=runtimes,
                           tmp=tmp_path, perceptor=perceptor, frames=frames)


def test_cli_batch_has_independent_sessions_and_bounded_private_recordings(cli):
    log = cli.tmp / "session.jsonl"
    log.write_text("existing session log\n")
    desktop.main(arguments(cli.tmp, "--arenas", "2", "--record-dir", str(cli.tmp / "frames"), "--record-limit", "2"))
    summary = json.loads(next(cli.tmp.glob("*-batch-*/summary.json")).read_text())
    assert (summary["reason"], summary["completed_arenas"], summary["actions"], summary["polls"]) == (
        "completed", 2, 2, 11)
    assert log.read_text() == "existing session log\n"
    assert len(cli.runtimes) == cli.connect.call_count == 2
    for runtime in cli.runtimes:
        runtime.act.assert_called_once_with(Action("open_play"))
    assert len(list((cli.tmp / "frames").rglob("*.png"))) == 4
    frame_dirs = set()
    for session in summary["sessions"]:
        assert session["first_recognized_phase"] == "main_menu"
        events = [json.loads(line) for line in Path(session["log"]).read_text().splitlines()]
        observed = [row for row in events if row["event"] == "observed"]
        recorded = [row for row in observed if "frame_path" in row]
        assert len(recorded) == 2 and recorded[-1]["record_limit_reached"] is True
        assert events[-1]["result"]["reason"] == "result"
        frame_dirs.add(Path(recorded[0]["frame_path"]).parent)
        assert (Path(session["log"]).parent / "last.png").is_file()
    assert len(frame_dirs) == 2
    cli.windows[0].close_for_restart.assert_called_once()
    cli.windows[1].close_for_restart.assert_not_called()


def test_batch_reports_resumed_shop_instead_of_claiming_a_fresh_menu_start(cli):
    empty, fish = PetSlot(False), PetSlot(True, "fish", 2, 3)
    before = Board(Phase.SHOP, gold=3, turn=6, shop=(fish,), team=(empty,) * 5)
    after = Board(Phase.SHOP, gold=0, turn=6, shop=(empty,), team=(fish, empty, empty, empty, empty))
    cli.frames[:] = [before, before, after, after, Board(Phase.RESULT)]
    desktop.main(arguments(cli.tmp, "--arenas", "2"))
    summary = json.loads(next(cli.tmp.glob("*-batch-*/summary.json")).read_text())
    assert summary["completed_arenas"] == 2
    assert [session["first_recognized_phase"] for session in summary["sessions"]] == ["shop", "shop"]


@pytest.mark.parametrize("stop", ["stop_file", "deadline", "escape"])
@pytest.mark.parametrize("when", ["before_close", "after_close"])
def test_cli_batch_keeps_one_global_stop_closure_across_restarts(cli, monkeypatch, stop, when):
    clock = [10.]
    monkeypatch.setattr(desktop.time, "monotonic", lambda: clock[0])
    def trigger():
        if stop == "stop_file":
            (cli.tmp / "STOP").touch()
        elif stop == "deadline":
            clock[0] = 11.
        else:
            cli.windows[0].stop_requested.return_value = True
    if when == "before_close":
        original = desktop.run_desktop_session
        def session(*args):
            result = original(*args)
            trigger()
            return result
        monkeypatch.setattr(desktop, "run_desktop_session", session)
    else:
        def close(*args, **kwargs):
            trigger()
            return True
        cli.windows[0].close_for_restart.side_effect = close
    desktop.main(arguments(cli.tmp, "--max-seconds", "1"))
    cli.connect.assert_called_once()
    summary = json.loads(next(cli.tmp.glob("*-batch-*/summary.json")).read_text())
    assert summary["reason"] == "stopped" and summary["completed_arenas"] == 1
    if when == "before_close":
        cli.windows[0].close_for_restart.assert_not_called()


def test_default_one_arena_keeps_existing_session_paths_and_never_closes(cli):
    desktop.main(arguments(cli.tmp, "--arenas", "1"))
    assert not list(cli.tmp.glob("*-batch-*"))
    events = [json.loads(line) for line in (cli.tmp / "session.jsonl").read_text().splitlines()]
    assert events[-1]["result"]["reason"] == "result"
    cli.connect.assert_called_once()
    cli.windows[0].close_for_restart.assert_not_called()


@pytest.mark.parametrize("flags", [["--no-launch"], ["--window", "custom game"]])
def test_multi_arena_rejects_unrestartable_configuration_before_desktop_io(cli, flags):
    with pytest.raises(SystemExit):
        desktop.main(arguments(cli.tmp, *flags))
    desktop.VisionProfile.load.assert_not_called()
    desktop.validate_live_dependencies.assert_not_called()
    cli.connect.assert_not_called()


def test_batch_preflight_failure_prevents_any_connection(cli):
    desktop.validate_profile_assets.side_effect = ValueError("profile asset failed")
    with pytest.raises(ValueError, match="profile asset failed"):
        desktop.main(arguments(cli.tmp))
    cli.connect.assert_not_called()
    assert not list(cli.tmp.glob("*-batch-*"))


def test_default_arena_count_preserves_single_game_behavior():
    assert desktop.parse_args(["play"]).arenas == 1
