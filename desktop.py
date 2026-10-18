"""Inspect recorded game frames or run the calibrated desktop controller."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from desktop_runtime import (CachedOCR, DesktopRuntime, WindowsGameWindow,
                             tesseract_ocr, validate_live_dependencies)
from desktop_launch import GAME_TITLE, connect_game_window
from desktop_control import desktop_control_lease
from desktop_session import DesktopSession, SessionResult
from desktop_state import DesktopPolicy, Phase
from desktop_vision import Perceptor, Rect, VisionProfile
from evaluation import positive_int


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return number


def read_image(path):
    from PIL import Image
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def save_image(frame, path):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


class FrameRecorder:
    """Bounded local evidence, recording only changes to the observed board."""

    def __init__(self, directory, limit=200):
        if type(limit) is not int or limit < 1:
            raise ValueError("record limit must be a positive integer")
        self.directory = Path(directory).resolve()
        self.limit = limit
        self.run_directory = None
        self.saved = 0
        self._last_board = None

    def record(self, event, frame):
        if event.get("event") != "observed":
            return event
        board = event.get("board")
        if not isinstance(board, dict):
            raise ValueError("observed recording event requires a board dictionary")
        # Store serialized content so later mutations cannot change history.
        signature = json.dumps(board, sort_keys=True, separators=(",", ":"))
        if signature == self._last_board:
            return event
        if self.saved >= self.limit:
            self._last_board = signature
            return event
        if frame is None:
            raise RuntimeError("cannot record an observed board without its captured frame")
        if self.run_directory is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            directory = self.directory / f"{stamp}-{uuid4().hex}"
            directory.mkdir(parents=True, exist_ok=False)
            self.run_directory = directory
        path = self.run_directory / f"{self.saved + 1:06d}.png"
        save_image(frame, path)
        self.saved += 1
        self._last_board = signature
        event = {**event, "frame_path": str(path)}
        if self.saved == self.limit:
            event.update(record_limit_reached=True, record_limit=self.limit)
        return event


def annotate(frame, profile):
    """Render numbered calibration regions on a local screenshot, without IO."""
    from PIL import Image, ImageDraw
    canvas = Image.fromarray(frame)
    draw = ImageDraw.Draw(canvas)
    regions = [(name, rect) for name, rect in profile.hud.items()]
    regions += [(f"phase {item.phase.value}", item.region) for item in profile.phase_templates]
    for row_name in ("shop", "team"):
        for index, slot in enumerate(getattr(profile, row_name)):
            for field in ("portrait", "attack", "health", "level"):
                rect = getattr(slot, field)
                if rect is not None:
                    regions.append((f"{row_name}{index} {field}", rect))
    for index, slot in enumerate(profile.food):
        regions.append((f"food{index} portrait", slot.portrait))
        if slot.cost is not None:
            regions.append((f"food{index} cost", slot.cost))
    for label, rect in regions:
        x, y, w, h = rect.to_list()
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline="#ff40bb", width=2)
        draw.text((x, max(0, y - 13)), label, fill="#ff40bb", stroke_width=1, stroke_fill="black")
    for label, (x, y) in profile.buttons.items():
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="yellow")
        draw.text((x + 8, y), label, fill="yellow", stroke_width=1, stroke_fill="black")
    return np.asarray(canvas)


def add_session_arguments(command, *, play=False):
    command.add_argument("--profile", required=not play,
                         default=".local/desktop/calibration-native.json" if play else None)
    command.add_argument("--window", default="Super Auto Pets")
    if play:
        command.set_defaults(execute=True, start_arena=True)
        command.add_argument("--no-launch", action="store_true",
                             help="require an already open game instead of launching it through Steam")
        command.add_argument("--launch-timeout", type=positive_seconds, default=90,
                             help="seconds allowed for each Steam startup or normal shutdown; runtime and stop controls still apply")
        command.add_argument("--startup-polls", type=positive_int, default=300,
                             help="initial unknown observations allowed while the game loads; normal unknown limit resumes after recognition")
        command.add_argument("--arenas", type=positive_int, default=1,
                             help="maximum arenas to collect; shared budgets, at most N-1 normal restarts plus one separately bounded completed-client bootstrap if enabled")
        command.add_argument("--restart-completed", action=argparse.BooleanOptionalAction, default=False,
                             help="allow one normal restart of an initially completed zero-action arena")
    else:
        command.add_argument("--execute", action="store_true")
        command.add_argument("--start-arena", action="store_true",
                             help="enable calibrated main menu, Play, and arena start transitions")
    command.add_argument("--max-actions", type=positive_int, default=600 if play else 40,
                         help="maximum gameplay actions; shared across all arenas in a batch")
    command.add_argument("--max-polls", type=positive_int, default=12000 if play else 600)
    command.add_argument("--max-seconds", type=positive_seconds, default=3600 if play else 1800,
                         help="wall-clock budget checked before each observation and action")
    command.add_argument("--action-timeout", type=positive_seconds, default=30,
                         help="seconds allowed for observed action acknowledgment")
    command.add_argument("--max-incomplete-shop-polls", type=positive_int, default=30,
                         help="consecutive incomplete shop observations allowed before stopping; never retries input")
    command.add_argument("--sale-funded-rolls", action="store_true",
                         help="experimental: spend spare gold on a roll when supported sale income could still fund a replacement")
    command.add_argument("--log", default=".local/desktop/session.jsonl")
    command.add_argument("--stop-file", default=".local/desktop/STOP",
                         help="stop before further input when this local file exists")
    command.add_argument("--last-frame", default=".local/desktop/last-frame.png",
                         help="save the last captured frame locally for diagnosis")
    command.add_argument("--record-dir", help="optionally record changed boards in a private local directory")
    command.add_argument("--record-limit", type=positive_int, default=200,
                         help="maximum recorded frames per run; reaching it does not stop control")


def validate_profile_assets(perceptor):
    """Load all calibrated references before play can activate the window."""
    profile = perceptor.profile
    for item in profile.phase_templates:
        perceptor._template(item.template, item.region)
    for slot in (*profile.shop, *profile.team):
        if slot.empty_template is not None:
            perceptor._template(slot.empty_template, slot.portrait)
        for _, path in slot.species_references():
            perceptor._template(path, slot.portrait)
    for slot in profile.food:
        if slot.empty_template is not None:
            perceptor._template(slot.empty_template, slot.portrait)
        for _, path in slot.food_references():
            perceptor._template(path, slot.portrait)
    for item in profile.numeric_templates:
        region, _, _ = profile.numeric_region(item.field)
        perceptor._template(item.template, region)


def stopped_before_start(args):
    """Preserve a zero-input session receipt without constructing desktop IO."""
    result = SessionResult(0, 0, 0, "stopped")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        for event in ({"event": "started", "poll": 0, "elapsed": 0, "preview": False},
                      {"event": "finished", "poll": 0, "elapsed": 0, "result": result.to_dict()}):
            log.write(json.dumps(event) + "\n")
    print(json.dumps(result.to_dict(), indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="save the foreground game client to a private local image")
    capture.add_argument("--output", required=True)
    capture.add_argument("--window", default="Super Auto Pets")
    crop = commands.add_parser("template", help="extract a calibrated reference from a recorded frame")
    crop.add_argument("--image", required=True)
    crop.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    crop.add_argument("--output", required=True)
    inspect = commands.add_parser("inspect", help="recognize a recorded image without desktop access")
    inspect.add_argument("--profile", required=True)
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--overlay", help="save a calibration overlay")
    run = commands.add_parser("run", help="preview a stable live proposal; --execute enables inputs")
    add_session_arguments(run)
    play = commands.add_parser("play", help="take game control with execution and arena transitions enabled")
    add_session_arguments(play, play=True)
    args = p.parse_args(argv)
    if args.command == "play" and args.arenas > 1:
        if args.no_launch or args.window != GAME_TITLE:
            p.error("--arenas greater than 1 requires automatic Steam launch and the default game window")
    if args.command == "play" and args.restart_completed and (args.no_launch or args.window != GAME_TITLE):
        p.error("--restart-completed requires automatic Steam launch and the default game window")
    return args


def run_desktop_session(args, profile, perceptor, window, should_stop, session_info=None):
    """Run one arena with the existing receipt and recording contract."""
    runtime = DesktopRuntime(profile, perceptor, window, execute=args.execute, should_stop=should_stop)
    policy = DesktopPolicy(sale_funded_rolls=getattr(args, "sale_funded_rolls", False))
    recorder = FrameRecorder(args.record_dir, args.record_limit) if args.record_dir else None
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        def event(data):
            if data.get("event") == "started" and policy.sale_funded_rolls:
                data = {**data, "policy": {"name": "stats", "sale_funded_rolls": True}}
            if session_info is not None and data.get("event") == "observed":
                phase = data["board"]["phase"]
                if phase != Phase.UNKNOWN.value and session_info["first_recognized_phase"] is None:
                    session_info["first_recognized_phase"] = phase
            if recorder is not None:
                data = recorder.record(data, runtime.last_frame)
            log.write(json.dumps(data) + "\n")
            log.flush()
        session = DesktopSession(runtime.observe, runtime.act, policy,
                                 preview=not args.execute, max_actions=args.max_actions,
                                 max_polls=args.max_polls, action_timeout=args.action_timeout,
                                 shop_ready=lambda board: (board.turn is not None and board.turn > 0
                                                           and policy.observation_complete(board)),
                                 max_incomplete_shop_polls=getattr(args, "max_incomplete_shop_polls", 30),
                                 max_initial_unknown_polls=args.startup_polls if args.command == "play" else None,
                                 event_callback=event,
                                 phase_actions=runtime.phase_actions(start_arena=args.start_arena),
                                 should_stop=should_stop)
        try:
            if args.command == "play" and not should_stop():
                window.activate(profile.image_size)
            result = session.run()
        except KeyboardInterrupt:
            event({"event": "stopped", "reason": "keyboard_interrupt"})
            print("Stopped by keyboard interrupt.")
            return None
        finally:
            if runtime.last_frame is not None:
                save_image(runtime.last_frame, args.last_frame)
    if getattr(args, "arenas", 1) == 1:
        print(json.dumps(result.to_dict(), indent=2))
    return result


def session_failed(args, result):
    return bool(result.error or result.pending_action or result.reason in {
        "action_timeout", "unknown_timeout", "next_shop_timeout", "turn_unreadable",
        "illegal_action", "transition_timeout", "repeated_transition", "shop_observation_timeout"} or (
        args.command == "play" and result.reason == "transition_disabled"))


def run_with_completed_restart(args, profile, perceptor, window, should_stop, open_window, session_info=None):
    """Allow one zero-action terminal bootstrap, retaining its separate receipt."""
    result = run_desktop_session(args, profile, perceptor, window, should_stop, session_info)
    info = {"polls": 0, "attempts": 0, "log": None}
    def empty_terminal(value):
        return (value is not None and value.reason == "result" and value.actions == 0
                and value.acknowledgments == 0 and value.pending_action is None and value.error is None
                and value.last_board is not None and value.last_board.phase == Phase.RESULT)
    if not args.restart_completed or not empty_terminal(result):
        return result, window, info
    info["polls"] = result.polls
    path = Path(args.log).with_name(f"{Path(args.log).stem}-bootstrap-{uuid4().hex}.jsonl")
    info["log"] = str(path.resolve())
    with path.open("x", encoding="utf-8") as log:
        def event(kind, **fields):
            log.write(json.dumps({"event": kind, **fields}) + "\n")
            log.flush()
        event("bootstrap_started", receipt=result.to_dict(), max_attempts=1)
        actual_started = False
        actual_error = None
        try:
            if should_stop():
                result = SessionResult(0, 0, 0, "stopped")
            elif info["polls"] + 1 >= args.max_polls:
                result = SessionResult(0, 0, 0, "max_polls")
            else:
                fresh = perceptor.observe(window.capture())
                info["polls"] += 1
                event("bootstrap_observed", board=fresh.to_dict(), polls=info["polls"])
                if should_stop():
                    result = SessionResult(0, 0, 0, "stopped")
                elif fresh.phase != Phase.RESULT:
                    raise RuntimeError("completed screen changed before bootstrap; no close requested")
                else:
                    info["attempts"] = 1
                    event("bootstrap_close_attempted", attempt=1)
                    if not window.close_for_restart(profile.image_size, timeout=args.launch_timeout):
                        result = SessionResult(0, 0, 0, "stopped")
                    elif should_stop():
                        result = SessionResult(0, 0, 0, "stopped")
                    else:
                        window = open_window()
                        if window is None or should_stop():
                            result = SessionResult(0, 0, 0, "stopped")
                        else:
                            remaining = argparse.Namespace(**vars(args))
                            remaining.max_polls -= info["polls"]
                            if session_info is not None:
                                session_info["first_recognized_phase"] = None
                            actual_started = True
                            result = None
                            result = run_desktop_session(remaining, profile, perceptor, window, should_stop, session_info)
                            if empty_terminal(result):
                                result = replace(result, reason="empty_terminal",
                                                 error="restarted client still shows a completed arena; no second bootstrap allowed")
        except KeyboardInterrupt:
            if actual_started:
                actual_error = "KeyboardInterrupt during actual arena; counts unknown"
                raise
            result = SessionResult(0, 0, 0, "stopped")
        except Exception as exc:
            if actual_started:
                actual_error = f"{type(exc).__name__}: {exc}"
                raise
            result = SessionResult(0, 0, 0, "bootstrap_error", error=f"{type(exc).__name__}: {exc}")
        finally:
            event("bootstrap_finished", **info, result=result.to_dict() if result is not None else None,
                  actual_session_counts_known=not actual_started or result is not None,
                  actual_session_error=actual_error)
    return result, window, info


def run_arena_batch(args, profile, perceptor, open_window, should_stop):
    """Collect bounded arenas, restarting only after verified terminal results."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    batch_name = f"{Path(args.log).stem}-batch-{stamp}-{uuid4().hex}"
    batch_dir = Path(args.log).parent / batch_name
    batch_dir.mkdir(parents=True, exist_ok=False)
    summary = {"requested_arenas": args.arenas, "completed_arenas": 0, "sessions": [],
               "actions": 0, "acknowledgments": 0, "polls": 0, "restart_polls": 0,
               "restart_attempts": 0, "max_restart_attempts": args.arenas - 1,
               "reason": "stopped", "error": None, "directory": str(batch_dir.resolve())}
    if args.restart_completed:
        summary.update(bootstrap_attempts=0, max_bootstrap_attempts=1, bootstrap_polls=0)
    summary_path = batch_dir / "summary.json"
    log_path = batch_dir / "batch.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        def event(kind, **fields):
            log.write(json.dumps({"event": kind, "batch": batch_name, **fields}) + "\n")
            log.flush()
        event("batch_started", requested_arenas=args.arenas, directory=summary["directory"],
              max_actions=args.max_actions, max_polls=args.max_polls, max_seconds=args.max_seconds,
              max_restart_attempts=summary["max_restart_attempts"])
        stage = "startup"
        try:
            for arena in range(1, args.arenas + 1):
                if should_stop():
                    break
                if summary["actions"] >= args.max_actions:
                    summary["reason"] = "max_actions"
                    break
                if summary["polls"] >= args.max_polls:
                    summary["reason"] = "max_polls"
                    break
                stage = "startup"
                window = open_window()
                if window is None or should_stop():
                    break
                arena_dir = batch_dir / f"arena-{arena:04d}"
                arena_dir.mkdir()
                session_args = argparse.Namespace(**vars(args))
                session_args.log = str(arena_dir / "session.jsonl")
                session_args.last_frame = str(arena_dir / Path(args.last_frame).name)
                session_args.record_dir = (str(Path(args.record_dir) / batch_name / f"arena-{arena:04d}")
                                           if args.record_dir else None)
                session_args.max_actions = args.max_actions - summary["actions"]
                session_args.max_polls = args.max_polls - summary["polls"]
                stage = "session"
                session_info = {"first_recognized_phase": None}
                if args.restart_completed and arena == 1:
                    result, window, bootstrap = run_with_completed_restart(
                        session_args, profile, perceptor, window, should_stop, open_window, session_info)
                    summary["polls"] += bootstrap["polls"]
                    summary["bootstrap_polls"] += bootstrap["polls"]
                    summary["bootstrap_attempts"] += bootstrap["attempts"]
                    if bootstrap["log"]:
                        event("bootstrap_finished", **bootstrap)
                else:
                    result = run_desktop_session(session_args, profile, perceptor, window, should_stop, session_info)
                if result is None:
                    break
                summary["actions"] += result.actions
                summary["acknowledgments"] += result.acknowledgments
                summary["polls"] += result.polls
                receipt = {"arena": arena, "log": str(Path(session_args.log).resolve()),
                           **session_info, "result": result.to_dict()}
                summary["sessions"].append(receipt)
                event("arena_finished", **receipt)
                # Merely observing a terminal client does not collect an arena.
                # This also prevents recounting a restored terminal screen.
                has_gameplay = result.actions > 0
                complete = (result.reason == "result" and result.last_board is not None
                            and result.last_board.phase == Phase.RESULT
                            and result.pending_action is None and result.error is None
                            and result.actions == result.acknowledgments
                            and has_gameplay)
                if not complete:
                    empty_terminal = result.reason == "result" and not has_gameplay
                    summary["reason"] = ("empty_terminal" if empty_terminal else
                                         "incomplete_terminal" if result.reason == "result" else result.reason)
                    if session_failed(args, result) or result.reason == "result":
                        summary["error"] = result.error or (
                            "terminal result contained no acknowledged gameplay" if empty_terminal
                            else f"arena did not finish safely: {summary['reason']}")
                    break
                summary["completed_arenas"] += 1
                if arena == args.arenas:
                    summary["reason"] = "completed"
                    break
                if should_stop():
                    break
                if summary["actions"] >= args.max_actions:
                    summary["reason"] = "max_actions"
                    break
                # A fresh result check and at least one next-session poll must fit.
                if summary["polls"] + 1 >= args.max_polls:
                    summary["reason"] = "max_polls"
                    break
                stage = "restart"
                fresh = perceptor.observe(window.capture())
                summary["polls"] += 1
                summary["restart_polls"] += 1
                event("restart_observed", arena=arena, board=fresh.to_dict(), polls=summary["polls"])
                if should_stop():
                    break
                if fresh.phase != Phase.RESULT:
                    raise RuntimeError("terminal screen changed before restart; no close requested")
                # Shutdown attempts are separate control operations, capped at N-1.
                # An attempt is retained even if Win32 refuses the request.
                summary["restart_attempts"] += 1
                event("restart_attempted", arena=arena, attempt=summary["restart_attempts"])
                if not window.close_for_restart(profile.image_size, timeout=args.launch_timeout):
                    break
        except KeyboardInterrupt:
            summary["reason"] = "stopped"
        except Exception as exc:
            summary["reason"] = f"{stage}_error"
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            event("batch_finished", summary=str(summary_path.resolve()),
                  completed_arenas=summary["completed_arenas"], actions=summary["actions"],
                  polls=summary["polls"], restart_attempts=summary["restart_attempts"],
                  reason=summary["reason"], error=summary["error"])
    print(json.dumps({**{key: value for key, value in summary.items() if key != "sessions"},
                      "summary": str(summary_path.resolve())}, indent=2))
    if summary["error"]:
        raise RuntimeError(f"desktop batch stopped: {summary['reason']}; see {summary_path}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.command == "capture":
        save_image(WindowsGameWindow(args.window).capture(), args.output)
        print(f"Saved private game capture to {args.output}")
        return
    if args.command == "template":
        frame = read_image(args.image)
        rect = Rect.from_list(args.region)
        rect.validate((frame.shape[1], frame.shape[0]))
        save_image(rect.crop(frame), args.output)
        print(f"Saved local template to {args.output}")
        return
    if args.command == "play" and Path(args.stop_file).exists():
        stopped_before_start(args)
        return
    try:
        profile = VisionProfile.load(args.profile)
    except FileNotFoundError as exc:
        if args.command != "play":
            raise
        raise RuntimeError(f"play requires a calibrated desktop profile at {args.profile}; see docs/desktop.md") from exc
    if args.command != "inspect":
        # Validate before constructing a desktop dependency or taking any screenshot.
        profile.validate(require_calibrated=True)
    perceptor = Perceptor(profile, ocr=CachedOCR(tesseract_ocr))
    if args.command == "inspect":
        frame = read_image(args.image)
        board = perceptor.observe(frame)
        action = DesktopPolicy().choose_action(board)
        print(json.dumps({"board": board.to_dict(), "proposal": action.to_dict() if action else None}, indent=2))
        if args.overlay:
            save_image(annotate(frame, profile), args.overlay)
        return
    if args.execute:
        with desktop_control_lease():
            return run_controller(args, profile, perceptor)
    return run_controller(args, profile, perceptor)


def run_controller(args, profile, perceptor):
    """Run startup and all arenas; the caller holds a lease when input is enabled."""
    deadline = time.monotonic() + args.max_seconds
    escape_requested = lambda: False
    def should_stop():
        return (Path(args.stop_file).exists() or time.monotonic() >= deadline
                or escape_requested())

    if args.command == "play":
        validate_profile_assets(perceptor)
        validate_live_dependencies()
        if should_stop():
            stopped_before_start(args)
            return
    def open_window():
        nonlocal escape_requested
        window = connect_game_window(args.window, WindowsGameWindow, launch=not args.no_launch,
                                     timeout=args.launch_timeout, should_stop=should_stop,
                                     is_ready=lambda candidate: candidate.ready_for_profile(profile.image_size))
        if window is not None:
            escape_requested = window.stop_requested
            window.should_stop = should_stop
        return window

    if args.command == "play" and args.arenas > 1:
        run_arena_batch(args, profile, perceptor, open_window, should_stop)
        return
    if args.command == "play":
        window = open_window()
        if window is None:
            stopped_before_start(args)
            return
    else:
        window = WindowsGameWindow(args.window)
    if args.command == "play" and args.restart_completed:
        result, window, bootstrap = run_with_completed_restart(
            args, profile, perceptor, window, should_stop, open_window)
        if bootstrap["log"]:
            print(json.dumps({"bootstrap": bootstrap, "reason": result.reason if result is not None else "stopped"}, indent=2))
    else:
        result = run_desktop_session(args, profile, perceptor, window, should_stop)
    if result is not None and session_failed(args, result):
        raise RuntimeError(f"desktop session stopped: {result.reason}; see {args.log}")


def cli():
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    cli()
