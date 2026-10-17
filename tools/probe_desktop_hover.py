"""Capture new development diagnostic hover frames from an observed shop.

This tool never launches, activates, clicks, or trains. Targets are zero-based
calibrated shop/team slots. Captures are explicitly development data before any
capture occurs; they must not enter held-out species evaluation. Existing
recordings are neither read nor reclassified. Tooltip text is not interpreted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / ".local"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop import validate_profile_assets
from desktop_runtime import (CachedOCR, DesktopUnavailable, OCR_TIMEOUT_SECONDS,
                             WindowsGameWindow, tesseract_ocr, validate_live_dependencies)
from desktop_state import Phase
from desktop_vision import Perceptor, VisionProfile


@dataclass(frozen=True)
class HoverTarget:
    row: str
    index: int


class ProbeStopped(RuntimeError):
    pass


def target_argument(value):
    match = re.fullmatch(r"(shop|team):([0-4])", value)
    if match is None:
        raise argparse.ArgumentTypeError("target must be shop:0..4 or team:0..4")
    return HoverTarget(match[1], int(match[2]))


def seconds_argument(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be a finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=".local/desktop/calibration-native.json")
    parser.add_argument("--window", default="Super Auto Pets")
    parser.add_argument("--target", action="append", type=target_argument, required=True,
                        help="zero-based calibrated occupied slot; repeat for at most five distinct targets")
    parser.add_argument("--output-dir", help="new directory strictly inside this repository's .local")
    parser.add_argument("--stop-file", default=".local/desktop/STOP")
    parser.add_argument("--max-seconds", type=seconds_argument, default=20.)
    parser.add_argument("--settle-seconds", type=seconds_argument, default=.75)
    args = parser.parse_args(argv)
    if len(args.target) > 5 or len(set(args.target)) != len(args.target):
        parser.error("choose at most five distinct targets")
    if args.max_seconds > 30:
        parser.error("--max-seconds cannot exceed 30")
    if args.settle_seconds > 2:
        parser.error("--settle-seconds cannot exceed 2")
    return args


def private_path(value):
    """Reject traversal, links, junctions and paths outside private storage."""
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError("private output paths cannot contain parent traversal")
    path = Path(os.path.abspath(raw if raw.is_absolute() else ROOT / raw))
    root = Path(os.path.abspath(PRIVATE_ROOT))
    if path == root or root not in path.parents:
        raise ValueError("output must be inside the repository's private .local directory")
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("linked or redirected private output paths are not allowed")
    return path


def write_json(path, value):
    with private_path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def write_image(path, frame):
    from PIL import Image
    with private_path(path).open("xb") as output:
        Image.fromarray(frame).save(output, format="PNG")


def run_probe(args, *, clock=None, sleep=None):
    """Inject clocks and module IO in tests; no retries after failed movement."""
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    started = clock()
    deadline = started + args.max_seconds
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = private_path(args.output_dir or f".local/desktop/hover-probes/{stamp}-{uuid4().hex}")
    if output.exists():
        raise ValueError("hover output directory must be new")
    stop_file = Path(args.stop_file)
    if not stop_file.is_absolute():
        stop_file = ROOT / stop_file
    escape_requested = lambda: False
    def stopped():
        return stop_file.exists() or clock() >= deadline or escape_requested()
    def require_running():
        if stopped():
            raise ProbeStopped("stop requested or probe deadline reached")
    def wait(seconds):
        end = clock() + seconds
        while clock() < end:
            require_running()
            sleep(min(.05, end - clock(), max(0., deadline - clock())))
        require_running()

    report = {"version": 1, "data_usage": "development_diagnostic", "directory": None,
              "reason": "stopped", "error": None, "targets": [], "pointer_moves": 0,
              "captures": 0, "pointer_restored": False, "elapsed": 0.}
    if stopped():
        return report
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    profile = VisionProfile.load(profile_path)
    profile.validate(require_calibrated=True)
    if profile.pointer_rest is None:
        raise ValueError("hover probe requires a calibrated pointer_rest to clear tooltips between targets")
    for target in args.target:
        if target.index >= len(getattr(profile, target.row)):
            raise ValueError(f"profile has no calibrated {target.row}:{target.index} slot")
    def guarded_ocr(image):
        require_running()
        # Reserve the existing engine's entire bounded call before starting it.
        if deadline - clock() <= OCR_TIMEOUT_SECONDS:
            raise ProbeStopped("remaining probe budget is too small for bounded OCR")
        result = tesseract_ocr(image)
        require_running()
        return result
    perceptor = Perceptor(profile, ocr=CachedOCR(guarded_ocr))
    validate_profile_assets(perceptor)
    validate_live_dependencies()
    if stopped():
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    private_path(output).mkdir(exist_ok=False)
    report["directory"] = str(output)
    # Written before constructing a window or capturing any new development frame.
    write_json(output / "manifest.json", {
        "version": 1, "data_usage": "development_diagnostic", "held_out_evaluation": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "profile": str(profile_path),
        "targets": [{"row": target.row, "index": target.index} for target in args.target],
        "max_seconds": args.max_seconds, "max_captures": len(args.target) * 3,
        "max_pointer_moves": len(args.target) * 2 + 1,
        "identity_interpretation": "none",
    })
    try:
        require_running()
        window = WindowsGameWindow(args.window)
        escape_requested = window.stop_requested
        window.should_stop = stopped
        def move(point):
            require_running()
            report["pointer_moves"] += 1
            window.hover(point, profile.image_size)
            require_running()
        def observe():
            require_running()
            frame = window.capture()
            report["captures"] += 1
            require_running()
            board = perceptor.observe(frame)
            require_running()
            return board, frame
        for number, target in enumerate(args.target, 1):
            move(profile.pointer_rest)
            report["pointer_restored"] = True
            wait(.15)
            before, _ = observe()
            wait(.1)
            stable, frame = observe()
            if before.phase != Phase.SHOP or stable.phase != Phase.SHOP:
                raise DesktopUnavailable("hover probe requires a freshly observed SHOP; no target hover attempted")
            if before.fingerprint() != stable.fingerprint():
                raise DesktopUnavailable("shop changed during probe preflight; no target hover attempted")
            slots = getattr(stable, target.row)
            if target.index >= len(slots) or slots[target.index].occupied is not True:
                raise DesktopUnavailable("hover target is not known occupied; no target hover attempted")
            point = getattr(profile, target.row)[target.index].portrait.center
            prefix = f"{number:02d}-{target.row}-{target.index}"
            before_path, after_path = output / f"{prefix}-before.png", output / f"{prefix}-after.png"
            write_image(before_path, frame)
            entry = {"row": target.row, "index": target.index, "point": list(point),
                     "before": stable.to_dict(), "before_frame": str(before_path),
                     "hover_attempted": False, "after": None, "after_frame": None}
            report["targets"].append(entry)
            require_running()
            entry["hover_attempted"] = True
            report["pointer_restored"] = False
            move(point)
            wait(args.settle_seconds)
            after, frame = observe()
            write_image(after_path, frame)
            entry.update(after=after.to_dict(), after_frame=str(after_path))
        move(profile.pointer_rest)
        report["pointer_restored"] = True
        report["reason"] = "completed"
    except (ProbeStopped, KeyboardInterrupt):
        report["reason"] = "stopped"
    except Exception as exc:
        report["reason"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed"] = max(0., clock() - started)
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    report = run_probe(parse_args(argv))
    print(json.dumps({key: value for key, value in report.items() if key != "targets"}, indent=2))
    if report["error"]:
        raise RuntimeError(f"hover probe stopped: {report['error']}; see {report['directory']}")
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
