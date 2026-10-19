"""Bounded game startup, separate from observation and gameplay input."""
from __future__ import annotations

import math
import os
import time

from sap_ai.desktop.runtime import DesktopUnavailable, GameWindowNotFound


GAME_TITLE = "Super Auto Pets"
STEAM_GAME_URI = "steam://rungameid/1714040"


def launch_installed_game():
    """Ask the installed Steam handler to open SAP; never handle sign-in."""
    if os.name != "nt":
        raise DesktopUnavailable("automatic game launch requires Windows")
    try:
        os.startfile(STEAM_GAME_URI)
    except OSError as exc:
        raise DesktopUnavailable("Steam could not launch Super Auto Pets; open the game manually") from exc


def connect_game_window(title, window_factory, *, launch=True, timeout=90.,
                        should_stop=None, clock=time.monotonic, sleep=time.sleep,
                        launch_game=launch_installed_game, is_ready=None):
    """Connect once, or launch a missing standard client and wait within bounds.

    Only a missing window permits launch. Focus, geometry, dependency and other
    failures propagate immediately. For a newly launched client, ``is_ready``
    may wait for its startup geometry before returning it for activation. An
    already open client keeps the normal immediate geometry validation.
    The original missing client's Escape latch works without a window and
    remains active throughout startup.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("launch timeout must be finite and positive")
    escape_requested = None

    def stopped():
        return bool((should_stop is not None and should_stop()) or
                    (escape_requested is not None and escape_requested()))

    if stopped():
        return None
    try:
        window = window_factory(title)
    except GameWindowNotFound as exc:
        if not launch or title != GAME_TITLE:
            raise
        escape_requested = exc.stop_requested
    else:
        return None if stopped() else window
    if stopped():
        return None
    deadline = clock() + timeout
    launch_game()
    while True:
        if stopped():
            return None
        if clock() >= deadline:
            raise DesktopUnavailable(
                "timed out waiting for Super Auto Pets; check Steam and complete any sign-in yourself"
            )
        try:
            window = window_factory(title)
        except GameWindowNotFound:
            sleep(min(.5, max(0., deadline - clock())))
        else:
            if stopped():
                return None
            if clock() >= deadline:
                raise DesktopUnavailable("timed out waiting for Super Auto Pets; open the game manually")
            if is_ready is not None and not is_ready(window):
                sleep(min(.5, max(0., deadline - clock())))
                continue
            return window
