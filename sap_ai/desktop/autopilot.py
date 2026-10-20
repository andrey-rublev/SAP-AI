"""Unattended play: keep playing arenas and recover instead of stopping.

A single session stops at the first thing it cannot explain (an unrecognized
screen, an unacknowledged click, lost focus). The autopilot treats those as
recoverable: it starts a fresh session from a new observation, and if that
makes no progress either, it restarts the game client, which reopens at the
main menu where the session navigates back into the arena. After each Game Over
it starts another arena. It only stops when asked (Escape, screen corner, STOP
file, deadline), after a configured number of arenas, or after repeated
failures that made no progress at all.

Every dependency is injected so each path is testable without a desktop.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

from sap_ai.desktop.runtime import DesktopUnavailable
from sap_ai.desktop.state import Phase


# Session endings a fresh observation or a client restart can plausibly fix.
RECOVERABLE = frozenset({
    "unknown_timeout", "action_timeout", "transition_timeout", "repeated_transition",
    "next_shop_timeout", "shop_observation_timeout", "policy_stopped", "illegal_action",
    "turn_unreadable", "transition_disabled", "observation_error", "action_error",
    "policy_error", "max_actions", "max_polls",
})
# Screens a fresh session cannot get past without restarting the client.
RESTART_FIRST = frozenset({"unknown_timeout", "transition_disabled", "repeated_transition", "transition_timeout"})
FOCUS_MARKERS = ("lost focus", "minimized")


@dataclass(frozen=True)
class AutopilotLimits:
    arenas: int = 0                      # 0 plays until stopped
    max_consecutive_failures: int = 5    # failures in a row with zero acknowledged actions
    max_restarts_without_progress: int = 3  # client restarts that changed nothing
    idle_seconds: float = 60.0           # user idle time before reclaiming focus
    poll_seconds: float = 1.0


def classify(result, error: str | None = None) -> str:
    """Map a session ending (or a pre-session error) to an autopilot response."""
    if error is not None:
        return "focus" if any(marker in error for marker in FOCUS_MARKERS) else "recover"
    if result is None or result.reason == "stopped" or "stop requested" in (result.error or ""):
        return "stopped"
    if result.reason == "result":
        played = result.actions > 0 and result.last_board is not None and result.last_board.phase == Phase.RESULT
        return "completed" if played else "terminal"
    if any(marker in (result.error or "") for marker in FOCUS_MARKERS):
        return "focus"
    return "recover" if result.reason in RECOVERABLE else "fatal"


class Autopilot:
    """Drive sessions until stopped; ``run`` returns the final summary."""

    def __init__(self, *, run_session: Callable, open_window: Callable, restart_client: Callable,
                 wait_for_focus: Callable, should_stop: Callable[[], bool], directory,
                 limits: AutopilotLimits = AutopilotLimits(), clock=time.monotonic,
                 report: Callable[[str], None] = print):
        self.run_session, self.open_window = run_session, open_window
        self.restart_client, self.wait_for_focus = restart_client, wait_for_focus
        self.should_stop, self.limits, self.clock, self.report = should_stop, limits, clock, report
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.summary = {"status": "running", "completed_arenas": 0, "trophies": [], "mean_trophies": None,
                        "sessions": 0, "actions": 0, "acknowledgments": 0, "recoveries": 0,
                        "client_restarts": 0, "focus_waits": 0, "failures_in_a_row": 0,
                        "last_reason": None, "error": None, "directory": str(self.directory.resolve())}
        self._started = clock()
        # An arena can span several sessions (recoveries); count it as a whole.
        self._arena_acks, self._arena_trophies = 0, None
        self._restarts_without_progress = 0

    # -- bookkeeping ------------------------------------------------------------
    def _event(self, kind, **fields):
        row = {"event": kind, "elapsed": round(self.clock() - self._started, 2), **fields}
        with (self.directory / "autopilot.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")

    def _save(self):
        self.summary["elapsed_seconds"] = round(self.clock() - self._started, 2)
        path = self.directory / "summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _arena_finished(self):
        self.summary["completed_arenas"] += 1
        if self._arena_trophies is not None:
            self.summary["trophies"].append(self._arena_trophies)
            trophies = self.summary["trophies"]
            self.summary["mean_trophies"] = round(sum(trophies) / len(trophies), 2)
        self.report(f"arena {self.summary['completed_arenas']} finished with "
                    f"{self._arena_trophies} trophies (mean {self.summary['mean_trophies']})")
        self._event("arena", number=self.summary["completed_arenas"], trophies=self._arena_trophies)
        self._arena_acks, self._arena_trophies = 0, None

    # -- main loop ----------------------------------------------------------------
    def run(self) -> dict:
        try:
            self._loop()
        except KeyboardInterrupt:
            self.summary["status"] = "stopped"
        except Exception as exc:  # unexpected: keep the evidence, then surface it
            self.summary.update(status="error", error=f"{type(exc).__name__}: {exc}")
            self._event("error", error=self.summary["error"])
            raise
        finally:
            self._save()
        return self.summary

    def _loop(self):
        window = self.open_window()
        restart_next = False
        while True:
            if window is None or self.should_stop():
                self.summary["status"] = "stopped"
                return
            if self.limits.arenas and self.summary["completed_arenas"] >= self.limits.arenas:
                self.summary["status"] = "completed"
                return
            if restart_next:
                restart_next = False
                if self._restarts_without_progress >= self.limits.max_restarts_without_progress:
                    self.summary.update(status="error", error=(
                        f"{self._restarts_without_progress} game restarts in a row changed nothing "
                        f"(last: {self.summary['last_reason']}); stopping instead of restarting again. "
                        f"See {self.directory}"))
                    return
                self._restarts_without_progress += 1
                self.summary["client_restarts"] += 1
                self._event("client_restart", number=self.summary["client_restarts"])
                self._save()
                window = self.restart_client(window)
                continue
            self.summary["sessions"] += 1
            info = {"first_recognized_phase": None, "trophies": None}
            result, error = None, None
            try:
                result = self.run_session(window, self.summary["sessions"], info)
            except DesktopUnavailable as exc:  # activation/geometry before any session input
                error = str(exc)
            outcome = classify(result, error)
            if result is not None:
                self.summary["actions"] += result.actions
                self.summary["acknowledgments"] += result.acknowledgments
                self._arena_acks += result.acknowledgments
            if info.get("trophies") is not None:
                self._arena_trophies = info["trophies"]
            if outcome == "terminal" and self._arena_acks > 0:
                outcome = "completed"  # Game Over reached by a recovery session
            reason = error or (result.reason if result is not None else "stopped")
            self.summary["last_reason"] = reason
            self._event("session", number=self.summary["sessions"], outcome=outcome, reason=reason,
                        actions=result.actions if result else 0,
                        acknowledgments=result.acknowledgments if result else 0,
                        error=result.error if result else error, trophies=info.get("trophies"))
            # Menu clicks repeat after every restart; only gameplay counts.
            progressed = result is not None and info.get("gameplay_acks", result.acknowledgments) > 0
            if progressed or outcome == "completed":
                self._restarts_without_progress = 0
            if outcome == "stopped":
                self.summary["status"] = "stopped"
                return
            if outcome == "fatal":
                self.summary.update(status="error", error=f"session ended with {reason}: {result.error}")
                return
            if outcome == "completed":
                self.summary["failures_in_a_row"] = 0
                self._arena_finished()
                restart_next = True  # Game Over: a fresh client opens a new arena
            elif outcome == "focus":
                self.summary["focus_waits"] += 1
                self.report("game is not in front; waiting until you are idle before resuming")
                window = self.wait_for_focus(window)
            else:  # recover / terminal
                self.summary["recoveries"] += 1
                failures = 0 if progressed else self.summary["failures_in_a_row"] + 1
                self.summary["failures_in_a_row"] = failures
                if failures >= self.limits.max_consecutive_failures:
                    self.summary.update(status="error", error=(
                        f"{failures} sessions in a row made no progress (last: {reason}); "
                        f"see {self.directory}"))
                    return
                # One fresh look first; restart when that also stalls or the
                # screen itself is the problem.
                restart_next = (outcome == "terminal" or failures >= 2
                                or (reason in RESTART_FIRST and not progressed))
                self.report(f"recovering from {reason} "
                            f"({'restarting the game' if restart_next else 'observing again'})")
            self._save()
