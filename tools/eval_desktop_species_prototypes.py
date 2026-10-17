"""Offline real-head-crop nearest-neighbor baseline; never opens final test.

Use the frozen train/calibration manifest accepted by train_desktop_species.
Compare direct and mirror-invariant RGB distances, with the latter designated
primary before evaluation. Calibration can only tighten distance <=.04 and
cross-class margin >=.015. Models, references and reports stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_desktop_species as crops

MAX_DISTANCE = .04
MIN_MARGIN = .015
IMPLEMENTATION_SOURCE = Path(__file__).read_bytes()
IMPLEMENTATION_SHA256 = hashlib.sha256(IMPLEMENTATION_SOURCE).hexdigest()
UNSAFE = ("known_wrong_accepts", "unknown_false_accepts", "empty_false_accepts", "occupied_false_empty")


def validate_policy(max_distance, margin):
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (max_distance, margin)) or not 0 <= max_distance <= MAX_DISTANCE
            or not MIN_MARGIN <= margin <= 1):
        raise ValueError("prototype policy must keep distance <=.04 and margin >=.015")


class Prototypes:
    """Class minima compete; extra same-class references are not rival classes."""
    def __init__(self, data):
        self.labels = list(data["labels"])
        self.rows = [row for row, _ in data["unique"] if row["split"] == "train"]
        self.arrays = np.stack([array for row, array in data["unique"] if row["split"] == "train"])
        self.indices = {label: np.array([i for i, row in enumerate(self.rows) if row["expected"] == label])
                        for label in self.labels}
        if any(not len(indices) for indices in self.indices.values()):
            raise ValueError("prototype training must cover every class including unknown and empty")

    def distances(self, array, *, mirrored):
        scores = np.abs(self.arrays - array).mean(axis=(1, 2, 3))
        if mirrored:
            scores = np.minimum(scores, np.abs(self.arrays[:, :, :, ::-1] - array).mean(axis=(1, 2, 3)))
        classes = []
        for label, indices in self.indices.items():
            index = int(indices[int(scores[indices].argmin())])
            classes.append((float(scores[index]), label, self.rows[index]["input_sha256"]))
        classes.sort()
        first, second = classes[:2]
        return {"top_label": first[1], "distance": first[0], "runner_up_label": second[1],
                "runner_up_distance": second[0], "cross_class_margin": second[0] - first[0],
                "prototype_input_sha256": first[2], "class_distances": {label: value for value, label, _ in classes}}


def score(records, labels, max_distance=MAX_DISTANCE, margin=MIN_MARGIN, *, keep_records=True):
    validate_policy(max_distance, margin)
    known_labels = set(labels) - {crops.UNKNOWN, crops.EMPTY}
    result = {"samples": len(records), "known_samples": 0, "known_correct": 0, "known_wrong_accepts": 0,
              "unknown_samples": 0, "unknown_false_accepts": 0, "empty_samples": 0,
              "empty_false_accepts": 0, "empty_correct": 0, "occupied_false_empty": 0, "accepted_species": 0,
              "per_species": {label: {"samples": 0, "correct": 0} for label in sorted(known_labels)}}
    output = []
    for record in records:
        prediction = record["top_label"] if (record["distance"] <= max_distance
                       and record["cross_class_margin"] > margin) else crops.UNKNOWN
        expected = record["expected"]
        accepted = prediction in known_labels
        result["accepted_species"] += int(accepted)
        result["occupied_false_empty"] += int(expected != crops.EMPTY and prediction == crops.EMPTY)
        if expected == crops.EMPTY:
            result["empty_samples"] += 1
            result["empty_false_accepts"] += int(accepted)
            result["empty_correct"] += int(prediction == crops.EMPTY)
        elif expected == crops.UNKNOWN:
            result["unknown_samples"] += 1
            result["unknown_false_accepts"] += int(accepted)
        else:
            result["known_samples"] += 1
            result["known_correct"] += int(prediction == expected)
            result["known_wrong_accepts"] += int(accepted and prediction != expected)
            result["per_species"][expected]["samples"] += 1
            result["per_species"][expected]["correct"] += int(prediction == expected)
        if keep_records:
            output.append({**record, "prediction": prediction})
    result["known_recall"] = result["known_correct"] / result["known_samples"] if result["known_samples"] else None
    result["accepted_species_precision"] = result["known_correct"] / result["accepted_species"] if result["accepted_species"] else None
    result["missing_known_species"] = [label for label, counts in result["per_species"].items() if not counts["samples"]]
    result["policy"] = {"max_distance": max_distance, "margin": margin}
    if keep_records:
        result["records"] = output
    return result


def calibrate(records, labels):
    """Maximize retained known identities under zero observed unsafe outcomes.

    Candidate boundaries are supplied by calibration distances/margins only.
    No test samples or model variants choose the primary representation.
    """
    distances = sorted({MAX_DISTANCE, 0., *(float(np.nextafter(row["distance"], -math.inf))
                                           for row in records if 0 < row["distance"] <= MAX_DISTANCE)})
    margins = sorted({MIN_MARGIN, 1., *(row["cross_class_margin"] for row in records
                                       if MIN_MARGIN <= row["cross_class_margin"] <= 1)})
    best = None
    for maximum in distances:
        for minimum in margins:
            result = score(records, labels, maximum, minimum, keep_records=False)
            if any(result[key] for key in UNSAFE):
                continue
            rank = (result["known_correct"], maximum, -minimum)
            if best is None or rank > best[0]:
                best = (rank, maximum, minimum)
    # Margin one always rejects normalized RGB distances, including exact ties.
    if best is None:
        raise ValueError("no safe prototype calibration policy")
    report = score(records, labels, best[1], best[2])
    return {"status": "calibrated", "policy": report["policy"], "report": report,
            "candidate_policies": len(distances) * len(margins)}


def run(manifest_path, output, seconds=600):
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
            or not 0 < seconds <= 600):
        raise ValueError("prototype experiment must have a positive limit of at most 600 seconds")
    start = time.monotonic()
    manifest_path = crops.private_path(manifest_path)
    output = crops.private_path(output)
    if output.exists():
        raise ValueError("prototype experiment requires a new output directory")
    manifest_bytes = manifest_path.read_bytes()
    declarations = json.loads(manifest_bytes)
    if (declarations.get("test_withheld") is not True
            or any(row.get("split") not in ("train", "calibration") for row in declarations.get("samples", []))):
        raise ValueError("prototype comparison requires withheld final test; no test images will be opened")
    data = crops.load_real_crops(manifest_path, _manifest_bytes=manifest_bytes)
    classifier = Prototypes(data)
    output.mkdir(parents=True)
    crops.write_output(output / "source.py", lambda handle: handle.write(IMPLEMENTATION_SOURCE))
    crops.write_output(output / "training-manifest.json", lambda handle: handle.write(manifest_bytes))
    def check_deadline():
        if time.monotonic() - start > seconds:
            raise TimeoutError("prototype experiment exceeded its time budget")
    variants = {}
    for name, mirrored in (("direct_rgb", False), ("mirror_rgb", True)):
        records = []
        for row, array in data["unique"]:
            if row["split"] != "calibration":
                continue
            check_deadline()
            records.append({**row, **classifier.distances(array, mirrored=mirrored)})
        check_deadline()
        variants[name] = {"mirror_invariant": mirrored, "fixed": score(records, data["labels"]),
                          "calibration": calibrate(records, data["labels"])}
        check_deadline()
    payload = io.BytesIO()
    np.savez_compressed(payload, arrays=classifier.arrays,
                        labels=np.array([row["expected"] for row in classifier.rows]),
                        input_sha256=np.array([row["input_sha256"] for row in classifier.rows]))
    artifact_bytes = payload.getvalue()
    crops.write_output(output / "prototypes.npz", lambda handle: handle.write(artifact_bytes))
    plan = {"kind": "real_crop_rgb_prototypes", "primary_variant": "mirror_rgb", "primary_selected_before_calibration": True,
            "manifest_sha256": data["provenance"]["manifest_sha256"], "implementation_sha256": IMPLEMENTATION_SHA256,
            "loader_implementation_sha256": crops.IMPLEMENTATION_SHA256,
            "prototype_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "labels": data["labels"],
            "prototype_source_inputs": [row["input_sha256"] for row in classifier.rows],
            "data_provenance": data["provenance"],
            "policies": {name: value["calibration"]["policy"] for name, value in variants.items()},
            "constraints": {"max_distance_cap": MAX_DISTANCE, "min_margin": MIN_MARGIN}, "test_state": "withheld"}
    crops.dump(output / "plan.json", plan)
    report = {"kind": "real_crop_prototype_calibration", "seconds": time.monotonic() - start, "threads": 1,
              "plan_sha256": crops.digest(output / "plan.json"), "primary_variant": "mirror_rgb",
              "implementation_sha256": IMPLEMENTATION_SHA256, **data["provenance"], "variants": variants,
              "test_state": "withheld", "eligible_for_promotion": False,
              "limitations": ["Calibration metrics select rejection limits and are not final-test accuracy.",
                              "Fixed game icon artwork recurs; distinct inputs do not prove novel art or pose coverage.",
                              "Within-session appearances remain correlated despite duplicate and near-alias gates.",
                              "Only two annotated arenas; sparse classes and unknown-species coverage.",
                              "Offline RGB template baseline only; no production profile or recognizer changes."]}
    crops.dump(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600)
    args = parser.parse_args(argv)
    report = run(args.manifest, args.output, args.seconds)
    print(json.dumps({"seconds": report["seconds"], "test_state": report["test_state"],
                      "variants": {name: {"fixed": {key: value for key, value in variant["fixed"].items() if key != "records"},
                                          "calibrated": {key: value for key, value in variant["calibration"]["report"].items() if key != "records"}}
                                   for name, variant in report["variants"].items()}}))


if __name__ == "__main__":
    main()
