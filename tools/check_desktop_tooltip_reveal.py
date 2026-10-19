"""Check an explicitly declared private six-frame development sequence offline.

Usage: python tools/check_desktop_tooltip_reveal.py --manifest PATH --output DIR
DIR must be new and under this checkout's .local. No window, capture, OCR,
profile discovery or Board perception is performed. Supplied Boards are caller
annotations, not established observations. Success supplies appearance evidence
and a title crop, never global clearance or a pet identity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import stat
import sys

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

from sap_ai.desktop.state import Board
from sap_ai.desktop.tooltip_layout import TooltipLayout
from sap_ai.desktop.tooltip_reveal import TooltipChangeLimits, verify_revealed_tooltip
from tools import probe_desktop_hover as private_io


MAX_MANIFEST_BYTES = 256 * 1024
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_FRAME_PIXELS = 4_000_000
STAGES = ("clear", "clear", "hover", "hover", "clear", "clear")
CLEAR_INDICES = (0, 1, 4, 5)
CODE_SHA256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
    "tools/check_desktop_tooltip_reveal.py", "tools/probe_desktop_hover.py",
    "sap_ai/desktop/tooltip_reveal.py", "sap_ai/desktop/tooltip_layout.py", "sap_ai/desktop/state.py")}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read_bounded(path, maximum):
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("diagnostic inputs must be regular files")
    with path.open("rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"diagnostic input exceeds its byte budget: {path.name}")
    return data


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(expected)}")


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _declarations(payload, manifest_path):
    data = json.loads(payload, object_pairs_hook=_object)
    _fields(data, ("version", "data_usage", "held_out_evaluation", "image_size", "layout", "change_limits",
                   "target", "frames", "clear_boards"), "development manifest")
    if (type(data["version"]) is not int or data["version"] != 1
            or data["data_usage"] != "development_diagnostic" or data["held_out_evaluation"] is not False):
        raise ValueError("only version 1 development_diagnostic data with held_out_evaluation=false is allowed")
    size = data["image_size"]
    if (not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size)
            or size[0] * size[1] > MAX_FRAME_PIXELS):
        raise ValueError("image_size must contain two positive integers within the four-million-pixel budget")
    if not isinstance(data["layout"], dict) or not isinstance(data["change_limits"], dict):
        raise ValueError("layout and change_limits must be explicit mappings")
    try:
        layout = TooltipLayout(**data["layout"])
        limits = TooltipChangeLimits(**data["change_limits"])
    except TypeError as exc:
        raise ValueError("unsupported or missing layout/change_limits fields") from exc
    x, y, width, height = layout.search_rect
    if x + width > size[0] or y + height > size[1]:
        raise ValueError("layout search must lie inside declared image_size")
    target = data["target"]
    _fields(target, ("row", "index"), "target")
    if (not isinstance(target["row"], str) or target["row"] not in {"shop", "team"}
            or type(target["index"]) is not int or not 0 <= target["index"] < 5):
        raise ValueError("target must identify shop/team and an index from zero through four")
    entries = data["frames"]
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError("exactly six ordered frame declarations are required")
    paths = []
    for expected_stage, entry in zip(STAGES, entries):
        _fields(entry, ("stage", "image", "sha256"), "frame")
        name = entry["image"]
        if entry["stage"] != expected_stage or not _digest(entry["sha256"]):
            raise ValueError("frames need fixed clear/clear/hover/hover/clear/clear stages and SHA256 hashes")
        if (not isinstance(name, str) or not name.strip() or Path(name).drive or Path(name).is_absolute()
                or ":" in name
                or Path(name).suffix.lower() != ".png"):
            raise ValueError("frame images must be relative PNG paths inside the manifest directory")
        path = private_io.private_path(manifest_path.parent / name)
        if not path.is_relative_to(manifest_path.parent) or path in paths:
            raise ValueError("six distinct canonical frame paths inside the manifest directory are required")
        paths.append(path)
    records = data["clear_boards"]
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("four caller-supplied clear Board records are required")
    boards = []
    for record, index in zip(records, CLEAR_INDICES):
        _fields(record, ("frame_sha256", "board", "label_source"), "clear Board record")
        if record["frame_sha256"] != entries[index]["sha256"]:
            raise ValueError("each caller Board must bind its corresponding clear frame hash")
        if not isinstance(record["label_source"], str) or not record["label_source"].strip():
            raise ValueError("caller Boards need a nonempty label_source description")
        if not isinstance(record["board"], dict):
            raise ValueError("caller Boards must be explicit typed Board mappings")
        try:
            boards.append(Board.from_dict(record["board"]))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid caller Board declaration: {exc}") from exc
    return data, paths, boards, layout, limits


def run(manifest_path, output_dir):
    """Write one new private report, using only explicit bounded PNG inputs.

    Reserve the output directory before reading inputs, preventing a late
    overwrite rejection after verification. Invalid input may leave it empty.
    All declarations validate before reading any frame bytes; every byte hash
    validates before decoding any image. Identical bytes can be separate static
    captures, so supplied capture history remains a caller assertion.
    """
    output = private_io.private_path(output_dir)
    if output.exists():
        raise ValueError("offline reveal output directory must be new")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = private_io.private_path(manifest_path)
    payload = _read_bounded(manifest_path, MAX_MANIFEST_BYTES)
    data, paths, boards, layout, limits = _declarations(payload, manifest_path)
    images = []
    for entry, path in zip(data["frames"], paths):
        raw = _read_bounded(path, MAX_FRAME_BYTES)
        if _sha(raw) != entry["sha256"]:
            raise ValueError(f"declared frame hash does not match: {entry['image']}")
        images.append(raw)
    frames = []
    for raw in images:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("frames must be single-image RGB PNG files")
            if list(image.size) != data["image_size"]:
                raise ValueError("frame dimensions must exactly match declared image_size")
            frames.append(np.asarray(image).copy())
    result = verify_revealed_tooltip(frames=frames, clear_boards=boards, layout=layout,
                                     target_row=data["target"]["row"], target_index=data["target"]["index"],
                                     change_limits=limits)
    report = {"version": 1, "data_usage": "development_diagnostic", "held_out_evaluation": False,
              "status": result.status, "reason": result.reason, "limitation": result.limitation,
              "global_clearance": False, "identity_interpretation": "none",
              "clear_board_interpretation": "caller_supplied_unverified_annotations",
              "manifest_sha256": _sha(payload), "code_sha256": dict(CODE_SHA256),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__},
              "image_size": data["image_size"], "target": data["target"], "frames": data["frames"],
              "layout": asdict(layout), "change_limits": asdict(limits),
              "clear_boards": [{**record, "board": board.to_dict()} for record, board in zip(data["clear_boards"], boards)],
              "changed_fraction": result.changed_fraction, "new_white_fraction": result.new_white_fraction,
              "unchanged_boundary_components": result.unchanged_boundary_components, "title": None}
    if result.status == "revealed" and result.title is not None:
        title_path = output / "title.png"
        private_io.write_image(title_path, result.title.title_crop)
        report["title"] = {"image": title_path.name, "sha256": _sha(title_path.read_bytes()),
                           "panel_rect": list(result.title.panel_rect), "title_rect": list(result.title.title_rect),
                           "white_fraction": result.title.white_fraction, "white_edge_fraction": result.title.white_edge_fraction,
                           "dark_perimeter_fraction": result.title.dark_perimeter_fraction}
    report_path = output / "report.json"
    private_io.write_json(report_path, report)
    return report_path, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="new private output directory")
    args = parser.parse_args(argv)
    try:
        path, report = run(args.manifest, args.output)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(path), "status": report["status"], "reason": report["reason"],
                      "global_clearance": False, "identity_interpretation": "none"}))
    return int(report["status"] != "revealed")


if __name__ == "__main__":
    raise SystemExit(main())
