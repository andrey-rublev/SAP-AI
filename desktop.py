"""Inspect recorded game frames or run the calibrated desktop controller."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from desktop_runtime import CachedOCR, DesktopRuntime, WindowsGameWindow, tesseract_ocr
from desktop_session import DesktopSession
from desktop_state import DesktopPolicy
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
    for label, rect in regions:
        x, y, w, h = rect.to_list()
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline="#ff40bb", width=2)
        draw.text((x, max(0, y - 13)), label, fill="#ff40bb", stroke_width=1, stroke_fill="black")
    for label, (x, y) in profile.buttons.items():
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="yellow")
        draw.text((x + 8, y), label, fill="yellow", stroke_width=1, stroke_fill="black")
    return np.asarray(canvas)


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
    run.add_argument("--profile", required=True)
    run.add_argument("--window", default="Super Auto Pets")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--max-actions", type=positive_int, default=40)
    run.add_argument("--max-polls", type=positive_int, default=600)
    run.add_argument("--max-seconds", type=positive_seconds, default=1800,
                     help="wall-clock budget checked before each observation and action")
    run.add_argument("--action-timeout", type=positive_seconds, default=30,
                     help="seconds allowed for observed action acknowledgment")
    run.add_argument("--log", default=".local/desktop/session.jsonl")
    run.add_argument("--stop-file", default=".local/desktop/STOP",
                     help="stop before further input when this local file exists")
    run.add_argument("--last-frame", default=".local/desktop/last-frame.png",
                     help="save the last captured frame locally for diagnosis")
    run.add_argument("--record-dir", help="optionally record changed boards in a private local directory")
    run.add_argument("--record-limit", type=positive_int, default=200,
                     help="maximum recorded frames per run; reaching it does not stop control")
    return p.parse_args(argv)


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
    profile = VisionProfile.load(args.profile)
    perceptor = Perceptor(profile, ocr=CachedOCR(tesseract_ocr))
    if args.command == "inspect":
        frame = read_image(args.image)
        board = perceptor.observe(frame)
        action = DesktopPolicy().choose_action(board)
        print(json.dumps({"board": board.to_dict(), "proposal": action.to_dict() if action else None}, indent=2))
        if args.overlay:
            save_image(annotate(frame, profile), args.overlay)
        return
    # Validate before constructing a desktop dependency or taking any screenshot.
    profile.validate(require_calibrated=True)
    deadline = time.monotonic() + args.max_seconds
    def should_stop():
        return Path(args.stop_file).exists() or time.monotonic() >= deadline

    window = WindowsGameWindow(args.window)
    runtime = DesktopRuntime(profile, perceptor, window, execute=args.execute, should_stop=should_stop)
    recorder = FrameRecorder(args.record_dir, args.record_limit) if args.record_dir else None
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        def event(data):
            if recorder is not None:
                data = recorder.record(data, runtime.last_frame)
            log.write(json.dumps(data) + "\n")
            log.flush()
        session = DesktopSession(runtime.observe, runtime.act, DesktopPolicy(),
                                 preview=not args.execute, max_actions=args.max_actions,
                                 max_polls=args.max_polls, action_timeout=args.action_timeout,
                                 event_callback=event, phase_actions=runtime.phase_actions(),
                                 should_stop=should_stop)
        try:
            result = session.run()
        except KeyboardInterrupt:
            event({"event": "stopped", "reason": "keyboard_interrupt"})
            print("Stopped by keyboard interrupt.")
            return
        finally:
            if runtime.last_frame is not None:
                save_image(runtime.last_frame, args.last_frame)
        print(json.dumps(result.to_dict(), indent=2))
        if result.error or result.pending_action or result.reason in {
                "action_timeout", "unknown_timeout", "next_shop_timeout", "turn_unreadable",
                "illegal_action", "transition_timeout", "repeated_transition"}:
            raise RuntimeError(f"desktop session stopped: {result.reason}; see {log_path}")


def cli():
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    cli()
