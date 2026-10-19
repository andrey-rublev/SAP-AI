"""Capture new development diagnostic hover frames from an observed shop.

This tool never launches, activates, clicks, or trains. Targets are zero-based
calibrated shop/team slots. Captures are explicitly development data before any
capture occurs; they must not enter held-out species evaluation. Existing
recordings are neither read nor reclassified. A verified appearance sequence is
diagnostic evidence, not proof of target identity. Tooltip text is not interpreted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / ".local"

from sap_ai.desktop.cli import validate_profile_assets
from sap_ai.desktop.control import desktop_control_lease
from sap_ai.desktop.runtime import (CachedOCR, DesktopUnavailable, OCR_TIMEOUT_SECONDS,
                             WindowsGameWindow, tesseract_ocr, validate_live_dependencies)
from sap_ai.desktop.state import Board, Phase
from sap_ai.desktop.tooltip_layout import TooltipLayout, inspect_tooltip_panel
from sap_ai.desktop.vision import Perceptor, VisionProfile


ASSOCIATION_LIMIT = (
    "A stable new panel after hover and matching clear typed Boards are development diagnostic evidence, "
    "not proof of target identity or causation. Absence applies only within the calibrated search region. "
    "Unknown Board fields remain unknown."
)


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
    parser.add_argument("--tooltip-layout", required=True,
                        help="private calibrated tooltip layout JSON; no game coordinates are inferred")
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
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with private_path(path).open("xb") as output:
        output.write(payload)
    return hashlib.sha256(payload).hexdigest()


def write_image(path, frame):
    from PIL import Image
    with private_path(path).open("xb") as output:
        Image.fromarray(frame).save(output, format="PNG")


def validate_layout(layout, image_size):
    if not isinstance(layout, TooltipLayout):
        raise TypeError("tooltip layout must be an explicit TooltipLayout")
    x, y, width, height = layout.search_rect
    if x + width > image_size[0] or y + height > image_size[1]:
        raise ValueError("tooltip search rectangle is outside the calibrated profile")
    return layout


def load_tooltip_layout(path, image_size):
    path = private_path(path)
    payload = path.read_bytes()
    data = json.loads(payload)
    if (not isinstance(data, dict) or set(data) != {"version", "calibrated", "image_size", "layout"}
            or type(data["version"]) is not int or data["version"] != 1 or data["calibrated"] is not True
            or not isinstance(data["image_size"], list) or any(type(n) is not int for n in data["image_size"])
            or data["image_size"] != list(image_size) or not isinstance(data["layout"], dict)):
        raise ValueError("tooltip layout needs version 1, calibrated=true and the exact profile image_size")
    try:
        layout = TooltipLayout(**data["layout"])
    except TypeError as exc:
        raise ValueError("unsupported tooltip layout fields") from exc
    return validate_layout(layout, image_size), {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def pixel_sha256(image):
    array = np.asarray(image)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def freeze_profile_snapshot(profile):
    """Detach loaded profile metadata from both its source and the capture reader."""
    try:
        payload = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        snapshot = json.loads(payload)
        detached = VisionProfile.from_dict(json.loads(payload), base_dir=profile.base_dir)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"loaded calibration cannot be frozen: {exc}") from exc
    return detached, {"data": snapshot, "canonical_sha256": hashlib.sha256(payload).hexdigest(),
                      "hash_format": "sorted-key compact UTF-8 JSON, ensure_ascii=false, allow_nan=false"}


def freeze_loaded_reference_pixels(profile, perceptor):
    """Bind cached RGB pixels actually loaded by asset preflight, never reread files.

    Own read-only copies keep later changes to a source file or previously held
    array from altering the references used by this probe. Zero-reference
    profiles do not require a perceptor cache API.
    """
    uses = {}
    def add(name, region, field):
        uses.setdefault(name, []).append({"field": field, "region": region.to_list()})
    for index, item in enumerate(profile.phase_templates):
        add(item.template, item.region, f"phase.{index}")
    for row in ("shop", "team"):
        for index, slot in enumerate(getattr(profile, row)):
            if slot.empty_template is not None:
                add(slot.empty_template, slot.portrait, f"{row}.{index}.empty")
            for species, name in slot.species_references():
                add(name, slot.portrait, f"{row}.{index}.species.{species}")
    for index, slot in enumerate(profile.food):
        if slot.empty_template is not None:
            add(slot.empty_template, slot.portrait, f"food.{index}.empty")
        for food_id, name in slot.food_references():
            add(name, slot.portrait, f"food.{index}.food_id.{food_id}")
    for item in profile.numeric_templates:
        region, _, _ = profile.numeric_region(item.field)
        add(item.template, region, f"numeric.{item.field}.{item.value}")
    if not uses:
        return []
    cache = getattr(perceptor, "_templates", None)
    if not isinstance(cache, dict):
        raise ValueError("loaded calibration reference cache is missing")
    references = []
    for name, fields in sorted(uses.items()):
        array = cache.get(name)
        if array is None:
            raise ValueError(f"loaded calibration reference is missing: {name!r}")
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"loaded calibration reference must be uint8 RGB: {name!r}")
        for use in fields:
            _, _, width, height = use["region"]
            if array.shape != (height, width, 3):
                raise ValueError(f"loaded calibration reference dimensions are incoherent: {name!r}")
        frozen = array.copy(order="C")
        frozen.flags.writeable = False
        cache[name] = frozen
        references.append({"name": name, "shape": list(frozen.shape), "dtype": frozen.dtype.str,
                           "pixel_sha256": pixel_sha256(frozen), "uses": fields})
    return references


def collect_association(profile, target, layout, *, capture, hover, observe, wait,
                        require_running, settle_seconds=.75, evidence=None,
                        record_frame=None, record_title=None):
    """Six supplied frames establish only a bounded tooltip appearance sequence.

    No native or file IO is used unless supplied by the caller. Board perception
    runs only after proven tooltip absence. Hover frames never supply pet stats
    or identities; both clear board pairs must match across the whole sequence.
    """
    profile.validate(require_calibrated=True)
    validate_layout(layout, profile.image_size)
    if (not isinstance(target, HoverTarget) or target.row not in {"shop", "team"}
            or type(target.index) is not int or not 0 <= target.index < len(getattr(profile, target.row))
            or profile.pointer_rest is None):
        raise ValueError("association needs an explicit calibrated occupied slot and pointer_rest")
    if type(settle_seconds) not in (int, float) or not math.isfinite(settle_seconds) or not 0 < settle_seconds <= 2:
        raise ValueError("settle_seconds must be finite, positive and at most two")
    point = getattr(profile, target.row)[target.index].portrait.center
    entry = evidence if evidence is not None else {}
    entry.update(row=target.row, index=target.index, point=list(point), appearance_sequence_verified=False,
                 association_limit=ASSOCIATION_LIMIT, identity_interpretation="none",
                 hover_attempted=False, before=None, after=None, before_frame=None, after_frame=None,
                 samples=[], pointer_restored=False)

    def move(destination):
        require_running()
        entry["pointer_restored"] = False
        hover(destination)
        require_running()
        entry["pointer_restored"] = destination == profile.pointer_rest

    def sample(stage, number):
        require_running()
        frame = np.asarray(capture())
        require_running()
        if frame.dtype != np.uint8 or frame.shape != (profile.image_size[1], profile.image_size[0], 3):
            raise ValueError("probe frame must match the calibrated uint8 RGB image_size")
        frame = frame.copy()  # A supplied camera may reuse its capture buffer.
        row = {"stage": stage, "number": number, "pixel_sha256": pixel_sha256(frame)}
        entry["samples"].append(row)
        if record_frame is not None:
            row["frame"] = record_frame(stage, number, frame)
        found = inspect_tooltip_panel(frame, layout)
        require_running()
        row.update(tooltip_status=found.status, tooltip_reason=found.reason)
        if found.title is not None:
            title = found.title
            row.update(panel_rect=list(title.panel_rect), title_rect=list(title.title_rect),
                       title_pixel_sha256=pixel_sha256(title.title_crop))
            if record_title is not None:
                row["title_frame"] = record_title(stage, number, title.title_crop)
        if stage == "hover":
            if found.status != "unique" or found.title is None:
                raise DesktopUnavailable(f"settled hover tooltip is {found.status}; association rejected")
            return frame, found.title, None, row
        if found.status != "absent":
            raise DesktopUnavailable(f"tooltip absence not established after parking: {found.status}")
        board = observe(frame)
        require_running()
        if not isinstance(board, Board) or board.phase != Phase.SHOP:
            raise DesktopUnavailable("hover probe requires a freshly observed SHOP; no association established")
        slots = getattr(board, target.row)
        if target.index >= len(slots) or slots[target.index].occupied is not True:
            raise DesktopUnavailable("hover target is not known occupied; association rejected")
        row["board"] = board.to_dict()
        return frame, None, board, row

    move(profile.pointer_rest)
    wait(.15)
    _, _, before, _ = sample("before", 1)
    wait(.1)
    _, _, stable, before_row = sample("before", 2)
    if before.fingerprint() != stable.fingerprint():
        raise DesktopUnavailable("shop changed during probe preflight; no target hover attempted")
    entry.update(before=stable.to_dict(), before_frame=before_row.get("frame"))
    require_running()
    entry["hover_attempted"] = True
    move(point)
    wait(settle_seconds)
    first_frame, first, _, _ = sample("hover", 1)
    wait(.1)
    second_frame, second, _, _ = sample("hover", 2)
    x, y, width, height = first.panel_rect
    if (first.panel_rect != second.panel_rect or first.title_rect != second.title_rect
            or not np.array_equal(first.title_crop, second.title_crop)
            or not np.array_equal(first_frame[y:y + height, x:x + width], second_frame[y:y + height, x:x + width])):
        raise DesktopUnavailable("tooltip panel or title changed between settled frames; association rejected")
    move(profile.pointer_rest)
    wait(.15)
    _, _, cleared, _ = sample("clear", 1)
    wait(.1)
    _, _, final, after_row = sample("clear", 2)
    if any(board.fingerprint() != stable.fingerprint() for board in (cleared, final)):
        raise DesktopUnavailable("underlying shop changed or became unreadable after hover; association rejected")
    require_running()
    entry.update(after=final.to_dict(), after_frame=after_row.get("frame"), appearance_sequence_verified=True,
                 panel_rect=list(first.panel_rect), title_rect=list(first.title_rect),
                 title_pixel_sha256=pixel_sha256(first.title_crop))
    return entry


def run_probe(args, *, clock=None, sleep=None, tooltip_layout=None):
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

    report = {"version": 2, "data_usage": "development_diagnostic", "directory": None,
              "reason": "stopped", "error": None, "targets": [], "pointer_moves": 0,
              "captures": 0, "pointer_restored": False, "elapsed": 0.,
              "manifest_sha256": None,
              "association_limit": ASSOCIATION_LIMIT, "identity_interpretation": "none"}
    if stopped():
        return report
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    profile = VisionProfile.load(profile_path)
    profile.validate(require_calibrated=True)
    profile, profile_snapshot = freeze_profile_snapshot(profile)
    if profile.pointer_rest is None:
        raise ValueError("hover probe requires a calibrated pointer_rest to clear tooltips between targets")
    for target in args.target:
        if target.index >= len(getattr(profile, target.row)):
            raise ValueError(f"profile has no calibrated {target.row}:{target.index} slot")
    if tooltip_layout is None:
        tooltip_layout, layout_source = load_tooltip_layout(args.tooltip_layout, profile.image_size)
    else:
        validate_layout(tooltip_layout, profile.image_size)
        layout_source = {"source": "explicit_caller"}
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
    reference_pixels = freeze_loaded_reference_pixels(profile, perceptor)
    validate_live_dependencies()
    if stopped():
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    private_path(output).mkdir(exist_ok=False)
    report["directory"] = str(output)
    # Written before constructing a window or capturing any new development frame.
    report["manifest_sha256"] = write_json(output / "manifest.json", {
        "version": 2, "data_usage": "development_diagnostic", "held_out_evaluation": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "profile": str(profile_path),
        "profile_snapshot": profile_snapshot, "loaded_reference_pixels": reference_pixels,
        "reference_pixel_hash_format": "SHA256 of str((shape, dtype.str)).encode() followed by C-order RGB bytes",
        "targets": [{"row": target.row, "index": target.index} for target in args.target],
        "max_seconds": args.max_seconds, "max_captures": len(args.target) * 6,
        "max_pointer_moves": len(args.target) * 3,
        "tooltip_layout": {**layout_source, "layout": asdict(tooltip_layout)},
        "identity_interpretation": "none",
        "association_limit": ASSOCIATION_LIMIT,
    })
    try:
        with desktop_control_lease():
            require_running()
            window = WindowsGameWindow(args.window)
            escape_requested = window.stop_requested
            window.should_stop = stopped
            def move(point):
                require_running()
                report["pointer_moves"] += 1
                window.hover(point, profile.image_size)
                require_running()
            def capture():
                require_running()
                frame = window.capture()
                report["captures"] += 1
                require_running()
                return frame
            for number, target in enumerate(args.target, 1):
                prefix = f"{number:02d}-{target.row}-{target.index}"
                entry = {}
                report["targets"].append(entry)
                def record_frame(stage, index, frame):
                    path = output / f"{prefix}-{stage}-{index}.png"
                    write_image(path, frame)
                    return str(path)
                def record_title(stage, index, title):
                    path = output / f"{prefix}-{stage}-{index}-title.png"
                    write_image(path, title)
                    return str(path)
                collect_association(profile, target, tooltip_layout, capture=capture, hover=move,
                                    observe=perceptor.observe, wait=wait, require_running=require_running,
                                    settle_seconds=args.settle_seconds, evidence=entry,
                                    record_frame=record_frame, record_title=record_title)
                report["pointer_restored"] = entry["pointer_restored"]
            report["reason"] = "completed"
    except (ProbeStopped, KeyboardInterrupt):
        report["reason"] = "stopped"
    except Exception as exc:
        report["reason"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if report["targets"]:
            report["pointer_restored"] = report["targets"][-1].get("pointer_restored", False)
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
