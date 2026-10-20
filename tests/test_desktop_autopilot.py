"""Autopilot recovery policy with fake sessions; never touches a desktop."""
import json

import pytest

import sap_ai.desktop.cli as desktop
from sap_ai.desktop.autopilot import Autopilot, AutopilotLimits, classify
from sap_ai.desktop.runtime import DesktopUnavailable
from sap_ai.desktop.session import SessionResult
from sap_ai.desktop.state import Board, Phase


def ended(reason, actions=0, acks=None, phase=Phase.SHOP, error=None):
    return SessionResult(actions, actions if acks is None else acks, 10, reason, Board(phase), error=error)


def game_over(trophies):
    return ended("result", actions=80, phase=Phase.RESULT), trophies


class Harness:
    def __init__(self, tmp_path, outcomes, *, limits=AutopilotLimits(), stop_after=None):
        self.outcomes, self.calls = list(outcomes), []
        self.restarts = self.focus_waits = 0
        self.stop_after = stop_after
        self.pilot = Autopilot(run_session=self.session, open_window=lambda: "window",
                               restart_client=self.restart, wait_for_focus=self.focus,
                               should_stop=self.stopped, directory=tmp_path / "pilot",
                               limits=limits, report=lambda message: None)

    def stopped(self):
        return not self.outcomes or (self.stop_after is not None and len(self.calls) >= self.stop_after)

    def session(self, window, number, info):
        self.calls.append(window)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, tuple):
            outcome, info["trophies"] = outcome
        return outcome

    def restart(self, window):
        self.restarts += 1
        return f"window-{self.restarts}"

    def focus(self, window):
        self.focus_waits += 1
        return window


def test_game_overs_start_new_arenas_and_record_trophies(tmp_path):
    run = Harness(tmp_path, [game_over(3), game_over(5), game_over(1)], limits=AutopilotLimits(arenas=2))
    summary = run.pilot.run()
    assert summary["status"] == "completed" and summary["completed_arenas"] == 2
    assert summary["trophies"] == [3, 5] and summary["mean_trophies"] == 4
    # No restart after the last requested arena.
    assert run.restarts == 1 and run.calls == ["window", "window-1"]
    saved = json.loads((tmp_path / "pilot" / "summary.json").read_text())
    assert saved["trophies"] == [3, 5]


def test_progress_then_stall_observes_again_before_restarting(tmp_path):
    run = Harness(tmp_path, [ended("action_timeout", actions=12, acks=11), game_over(2)],
                  limits=AutopilotLimits(arenas=1))
    summary = run.pilot.run()
    assert run.restarts == 0 and summary["recoveries"] == 1 and summary["completed_arenas"] == 1


def test_unrecognized_screen_without_progress_restarts_the_client(tmp_path):
    run = Harness(tmp_path, [ended("unknown_timeout"), game_over(4)], limits=AutopilotLimits(arenas=1))
    summary = run.pilot.run()
    assert run.restarts == 1 and summary["client_restarts"] == 1
    assert run.calls == ["window", "window-1"]


def test_second_stalled_session_escalates_to_a_restart(tmp_path):
    run = Harness(tmp_path, [ended("action_timeout"), ended("action_timeout"), game_over(0)],
                  limits=AutopilotLimits(arenas=1))
    run.pilot.run()
    assert run.calls[:3] == ["window", "window", "window-1"]


def test_repeated_failures_without_progress_give_up_with_evidence(tmp_path):
    run = Harness(tmp_path, [ended("unknown_timeout")] * 6, limits=AutopilotLimits(max_consecutive_failures=3))
    summary = run.pilot.run()
    assert summary["status"] == "error" and "3 sessions in a row" in summary["error"]
    assert len(run.calls) == 3


def test_lost_focus_waits_instead_of_counting_as_failure(tmp_path):
    lost = ended("observation_error", error="DesktopUnavailable: game lost focus; no desktop input was sent")
    run = Harness(tmp_path, [lost, DesktopUnavailable("restore the minimized game window before startup"),
                             game_over(2)], limits=AutopilotLimits(arenas=1))
    summary = run.pilot.run()
    assert run.focus_waits == 2 and summary["failures_in_a_row"] == 0 and summary["completed_arenas"] == 1


def test_user_stop_and_fatal_errors_end_the_run(tmp_path):
    assert Harness(tmp_path / "a", [ended("stopped")]).pilot.run()["status"] == "stopped"
    summary = Harness(tmp_path / "b", [ended("event_error", error="disk full")]).pilot.run()
    assert summary["status"] == "error" and "event_error" in summary["error"]


def test_classification_of_session_endings():
    assert classify(ended("result", actions=0, phase=Phase.RESULT)) == "terminal"
    assert classify(ended("result", actions=5, phase=Phase.RESULT)) == "completed"
    assert classify(ended("max_actions", actions=600)) == "recover"
    assert classify(None, "game size differs from the calibrated profile") == "recover"
    assert classify(ended("event_error")) == "fatal"


def test_continuous_cli_defaults_to_unlimited_arenas_and_time():
    args = desktop.parse_args(["play", "--continuous"])
    assert args.arenas == 0 and args.max_seconds > 365 * 24 * 3600
    capped = desktop.parse_args(["play", "--continuous", "--arenas", "3", "--max-seconds", "60"])
    assert (capped.arenas, capped.max_seconds) == (3, 60)
    with pytest.raises(SystemExit):
        desktop.parse_args(["play", "--continuous", "--no-launch"])


def test_arena_finished_by_a_recovery_session_counts_with_its_last_trophies(tmp_path):
    stalled = ended("unknown_timeout", actions=11, acks=10)
    run = Harness(tmp_path, [(stalled, 3), ended("result", actions=0, phase=Phase.RESULT)],
                  limits=AutopilotLimits(arenas=1))
    summary = run.pilot.run()
    assert summary["completed_arenas"] == 1 and summary["trophies"] == [3]


def test_stale_game_over_at_startup_is_not_an_arena(tmp_path):
    run = Harness(tmp_path, [ended("result", actions=0, phase=Phase.RESULT), game_over(2)],
                  limits=AutopilotLimits(arenas=1))
    summary = run.pilot.run()
    assert summary["completed_arenas"] == 1 and summary["trophies"] == [2] and run.restarts == 1
