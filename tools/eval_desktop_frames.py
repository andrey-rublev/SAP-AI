"""Bounded offline perception study on privately labeled screenshots.

    python tools/eval_desktop_frames.py --manifest .local/desktop/labels.json \
        --profile .local/desktop/calibration-native.json --sweep

Manifest format: {"version":1,"frames":[{"image":"frame.png", "labels":
{"phase":"shop","gold":10,"team.0.occupied":true,"team.0.attack":2},
"label_source":"independent visual observation"}]}. Omit unknown labels.
Photometric perturbations preserve geometry; their results do not establish
accuracy on new live scenes. No desktop window, capture, or input is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import io
import itertools
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_runtime import CachedOCR, tesseract_ocr
from desktop_state import Phase
from desktop_vision import Perceptor, VisionProfile
from evaluation import positive_int


def digest(data):
    return hashlib.sha256(data).hexdigest()


def variants(sweep=False):
    """Identity first, followed by modest deterministic changes, without shifts."""
    settings = [(1.0, 1.0, None)]
    if sweep:
        settings += list(itertools.product(
            (.92, .94, .96, .98, 1.0, 1.02, 1.04, 1.06, 1.08),
            (.94, .97, 1.0, 1.03, 1.06), (None, 95, 90)))
    else:
        settings += [(.96, 1., None), (1.04, 1., None),
                     (1., .96, None), (1., 1.04, None), (1., 1., 95), (1., 1., 90)]
    return [{"name": f"b{b:.2f}-c{c:.2f}-j{j or 'none'}",
             "brightness": b, "contrast": c, "jpeg_quality": j}
            for b, c, j in dict.fromkeys(settings)]


def perturb(frame, variant):
    image = Image.fromarray(frame)
    if variant["brightness"] != 1:
        image = ImageEnhance.Brightness(image).enhance(variant["brightness"])
    if variant["contrast"] != 1:
        image = ImageEnhance.Contrast(image).enhance(variant["contrast"])
    if variant["jpeg_quality"] is not None:
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=variant["jpeg_quality"], subsampling=0)
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            return np.asarray(decoded.convert("RGB")).copy()
    return np.asarray(image).copy()


def validate_labels(labels):
    if not isinstance(labels, dict) or "phase" not in labels:
        raise ValueError("each frame needs explicit phase and known labels")
    for field, value in labels.items():
        if field == "phase":
            Phase(value)
            continue
        match = re.fullmatch(r"(shop|team)\.([0-4])\.(occupied|attack|health|level)", field)
        if field not in {"gold", "turn", "wins", "lives"} and not match:
            raise ValueError(f"unsupported label field: {field}")
        stat = match[3] if match else field
        if stat == "occupied":
            if type(value) is not bool:
                raise ValueError("occupancy labels must be known booleans")
        elif type(value) is not int or not (1 if stat in {"turn", "health", "level"} else 0) <= value <= (3 if stat == "level" else 99):
            raise ValueError(f"label must be a known integer in range: {field}")
        if match and stat != "occupied" and labels.get(f"{match[1]}.{match[2]}.occupied") is not True:
            raise ValueError("slot stat labels require occupied=true")
    if labels["phase"] != "shop" and len(labels) != 1:
        raise ValueError("non-shop frames may label only phase; production skips their OCR")


def load_manifest(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("version") != 1 or not isinstance(data.get("frames"), list) or not data["frames"]:
        raise ValueError("manifest needs version 1 and nonempty frames")
    frames, seen = [], set()
    for entry in data["frames"]:
        validate_labels(entry["labels"])
        if not isinstance(entry.get("label_source"), str) or not entry["label_source"].strip():
            raise ValueError("labels need an independent label_source description")
        image_path = (path.parent / entry["image"]).resolve()
        if Path(entry["image"]).is_absolute() or not image_path.is_relative_to(path.parent):
            raise ValueError("manifest images must be relative and inside its directory")
        if image_path in seen:
            raise ValueError("duplicate manifest image")
        seen.add(image_path)
        image_bytes = image_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            size = list(image.size)
        frames.append({**entry, "sha256": digest(image_bytes), "size": size,
                       "bytes": image_bytes})
    return raw, frames


def snapshot_profile(path, directory):
    """Freeze profile and every referenced template before live calibration changes."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    profile = VisionProfile.from_dict(json.loads(raw), base_dir=path.parent)
    names = {item.template for item in (*profile.phase_templates, *profile.numeric_templates)}
    for slot in (*profile.shop, *profile.team):
        names.update(slot.species_templates.values())
        if slot.empty_template:
            names.add(slot.empty_template)
    directory.mkdir()
    hashes = {}
    for name in sorted(names):
        data = profile.template_path(name).read_bytes()
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = digest(data)
    target = directory / "profile-snapshot.json"
    if target.exists():
        raise ValueError("template conflicts with profile snapshot filename")
    target.write_bytes(raw)
    return VisionProfile.load(target), {"path": str(path), "sha256": digest(raw), "templates": hashes}


def field_value(board, field):
    value = board
    try:
        for part in field.split("."):
            value = value[int(part)] if part.isdigit() else value[part]
        return value
    except (KeyError, IndexError, TypeError):
        return None


def score(labels, board, *, observation_failed=False):
    counts = {"correct": 0, "unknown": 0, "incorrect": 0, "unsafe_false_empty": 0}
    fields, issues = {}, []
    for field, expected in labels.items():
        actual = field_value(board, field)
        if observation_failed:
            actual = None
        same = not observation_failed and type(actual) is type(expected) and actual == expected
        outcome = "correct" if same else "unknown" if actual is None or (field == "phase" and actual == "unknown") else "incorrect"
        counts[outcome] += 1
        fields[field] = outcome
        false_empty = field.endswith(".occupied") and expected is True and actual is False
        counts["unsafe_false_empty"] += int(false_empty)
        if outcome != "correct":
            issues.append({"field": field, "expected": expected, "actual": actual,
                           "outcome": outcome, "unsafe_false_empty": false_empty})
    return counts, fields, issues


def checkpoint(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class EvaluationStopped(TimeoutError):
    pass


def run(manifest_path, profile_path, output_dir, *, max_seconds=3600, max_cases=1000,
        max_examples=20, stop_file=None, sweep=False, ocr=tesseract_ocr, clock=time.monotonic):
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if type(max_cases) is not int or max_cases <= 0 or type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_cases must be positive and max_examples nonnegative integers")
    started = clock()
    raw_manifest, frames = load_manifest(manifest_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = Path(output_dir).resolve() / f"{stamp}-{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "manifest-snapshot.json").write_bytes(raw_manifest)
    profile, provenance = snapshot_profile(profile_path, directory / "profile")
    if any(tuple(frame["size"]) != profile.image_size for frame in frames):
        raise ValueError("source frame dimensions must exactly match the profile")
    settings = variants(sweep)
    packages = {}
    for name in ("numpy", "Pillow", "pytesseract"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {"status": "running", "limitation": "Recorded-frame perturbations do not establish live accuracy; reference-source frames are not held-out validation.",
              "profile": provenance, "manifest_sha256": digest(raw_manifest),
              "sources": [{k: v for k, v in frame.items() if k != "bytes"} for frame in frames],
              "code_sha256": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in
                              (Path(__file__).resolve(), ROOT / "desktop_vision.py", ROOT / "desktop_runtime.py", ROOT / "desktop_state.py")},
              "python": platform.python_version(), "packages": packages, "variants": settings,
              "limits": {"seconds": max_seconds, "cases": max_cases, "examples": max_examples},
              "counts": {"correct": 0, "unknown": 0, "incorrect": 0, "unsafe_false_empty": 0},
              "per_field": {}, "per_variant": {}, "phase_confusion": {}, "cases": [],
              "ocr_calls": 0, "ocr_seconds": 0, "examples_saved": 0, "observation_errors": 0}
    report_path = directory / "report.json"

    def check_stop():
        if stop_file is not None and Path(stop_file).exists():
            raise EvaluationStopped("stop_file")
        if clock() - started >= max_seconds:
            raise EvaluationStopped("time_limit")

    def measured_ocr(crop):
        check_stop()
        began = clock()
        report["ocr_calls"] += 1
        try:
            return ocr(crop)
        finally:
            report["ocr_seconds"] += clock() - began

    cached = CachedOCR(measured_ocr)

    def bounded_ocr(crop):
        check_stop()  # Also responsive when an exact crop hits the cache.
        return cached(crop)

    perceptor = Perceptor(profile, ocr=bounded_ocr)
    checkpoint(report_path, report)
    try:
        # Variant-first ordering covers all source scenes before increasing stress.
        for variant, source in itertools.product(settings, frames):
            check_stop()
            if len(report["cases"]) >= max_cases:
                raise EvaluationStopped("case_limit")
            began = clock()
            with Image.open(io.BytesIO(source["bytes"])) as image:
                frame = perturb(np.asarray(image.convert("RGB")), variant)
            error = None
            try:
                observed = perceptor.observe(frame).to_dict()
            except EvaluationStopped:
                raise
            except Exception as exc:
                observed = {"phase": "unknown"}
                error = f"{type(exc).__name__}: {exc}"
            counts, fields, issues = score(source["labels"], observed, observation_failed=error is not None)
            report["observation_errors"] += int(error is not None)
            case = {"image": source["image"], "variant": variant["name"], "observed": observed,
                    "counts": counts, "issues": issues, "seconds": clock() - began, "error": error}
            if issues and report["examples_saved"] < max_examples:
                name = f"failure-{len(report['cases']) + 1:06d}.png"
                Image.fromarray(frame).save(directory / name)
                case["example"] = name
                report["examples_saved"] += 1
            report["cases"].append(case)
            for key, value in counts.items():
                report["counts"][key] += value
                bucket = report["per_variant"].setdefault(variant["name"], {})
                bucket[key] = bucket.get(key, 0) + value
            for field, outcome in fields.items():
                bucket = report["per_field"].setdefault(field, {"correct": 0, "unknown": 0, "incorrect": 0})
                bucket[outcome] += 1
            expected = source["labels"]["phase"]
            confusion = report["phase_confusion"].setdefault(expected, {})
            actual = "observation_error" if error is not None else observed["phase"]
            confusion[actual] = confusion.get(actual, 0) + 1
            report["elapsed_seconds"] = clock() - started
            checkpoint(report_path, report)
        report["status"] = "complete"
    except EvaluationStopped as exc:
        report["status"] = str(exc)
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        report["elapsed_seconds"] = clock() - started
        checkpoint(report_path, report)
    return report_path, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", default=".local/desktop/frame-evaluations")
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--max-cases", type=positive_int, default=1000)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--stop-file", default=".local/desktop/STOP_FRAME_EVAL")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)
    path, report = run(args.manifest, args.profile, args.output_dir, max_seconds=args.max_seconds,
                       max_cases=args.max_cases, max_examples=args.max_examples,
                       stop_file=args.stop_file, sweep=args.sweep)
    print(json.dumps({"report": str(path), "status": report["status"], "cases": len(report["cases"]),
                      "counts": report["counts"], "observation_errors": report["observation_errors"]}))
    return int(report["observation_errors"] > 0 or report["counts"]["incorrect"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
