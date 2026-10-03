"""Compare completed private perception studies on exactly the same cases.

    python tools/compare_desktop_frames.py baseline/report.json candidate/report.json

This is a paired diagnostic, never authorization to change live calibration.
No screenshots, desktop IO, or OCR are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_desktop_frames import score, validate_labels


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def indexed_cases(report):
    """Reject partial, duplicate, or inconsistent studies before pairing."""
    if report.get("status") != "complete":
        raise ValueError("comparison requires completed studies")
    if not _hash(report.get("manifest_sha256")):
        raise ValueError("report needs a valid manifest_sha256")
    if not isinstance(report.get("python"), str) or not report["python"].strip():
        raise ValueError("report needs a nonempty python version")
    packages = report.get("packages")
    if (not isinstance(packages, dict) or not packages
            or any(not isinstance(name, str) or not name.strip()
                   or (value is not None and (not isinstance(value, str) or not value.strip()))
                   for name, value in packages.items())):
        raise ValueError("report needs package version provenance")
    if not isinstance(report.get("profile"), dict) or not _hash(report["profile"].get("sha256")):
        raise ValueError("report needs a valid profile hash")
    code = report.get("code_sha256")
    if (not isinstance(code, dict) or not code
            or any(not isinstance(name, str) or not name.strip() or not _hash(value)
                   for name, value in code.items())):
        raise ValueError("report needs valid implementation hashes")
    sources, variants = report.get("sources"), report.get("variants")
    if not isinstance(sources, list) or not sources or not isinstance(variants, list) or not variants:
        raise ValueError("report needs nonempty sources and variants")
    labels, names = {}, set()
    for source in sources:
        name = source["image"]
        if not isinstance(name, str) or not name or name in labels:
            raise ValueError("source images must be unique nonempty names")
        size = source.get("size")
        if (not _hash(source.get("sha256")) or not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
                or not isinstance(source.get("label_source"), str) or not source["label_source"].strip()):
            raise ValueError("source needs a hash, dimensions and independent label provenance")
        validate_labels(source["labels"])
        labels[name] = source["labels"]
    for variant in variants:
        name = variant["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("variant names must be unique nonempty names")
        names.add(name)
    cases, counts, errors = {}, Counter(), 0
    for case in report.get("cases", []):
        key = (case["image"], case["variant"])
        if key in cases or key[0] not in labels or key[1] not in names:
            raise ValueError("duplicate or unexpected evaluation case")
        failed = case.get("error") is not None
        scored, fields, issues = score(labels[key[0]], case["observed"], observation_failed=failed)
        if case.get("counts") != scored:
            raise ValueError("case counts disagree with labeled observations")
        cases[key] = {"fields": fields, "failed": failed,
                      "false_empty": {issue["field"] for issue in issues if issue["unsafe_false_empty"]}}
        counts.update(scored)
        errors += int(failed)
    expected = {(image, variant) for image in labels for variant in names}
    if set(cases) != expected:
        raise ValueError("completed report is missing evaluation cases")
    if report.get("counts") != dict(counts) or report.get("observation_errors") != errors:
        raise ValueError("report totals disagree with evaluation cases")
    return cases


def compare(baseline, candidate, *, max_examples=20):
    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a nonnegative integer")
    before, after = indexed_cases(baseline), indexed_cases(candidate)
    # Profile and implementation hashes may differ intentionally. Data, labels,
    # perturbations and runtime packages must agree to isolate the experiment.
    for name in ("manifest_sha256", "sources", "variants", "python", "packages"):
        if name not in baseline or name not in candidate or baseline[name] != candidate[name]:
            raise ValueError(f"studies differ in {name}; cases are not comparable")
    if set(before) != set(after):
        raise ValueError("studies do not cover the same cases")
    transitions, per_field, examples = Counter(), {}, []
    improved = regressed = new_incorrect = new_false_empty = new_errors = 0
    for key in sorted(before):
        old, new = before[key], after[key]
        new_errors += int(new["failed"] and not old["failed"])
        new_false_empty += len(new["false_empty"] - old["false_empty"])
        changes = {}
        for field, old_outcome in old["fields"].items():
            new_outcome = new["fields"][field]
            transition = f"{old_outcome}->{new_outcome}"
            transitions[transition] += 1
            per_field.setdefault(field, Counter())[transition] += 1
            improved += int(old_outcome != "correct" and new_outcome == "correct")
            regressed += int(old_outcome == "correct" and new_outcome != "correct")
            new_incorrect += int(old_outcome != "incorrect" and new_outcome == "incorrect")
            if old_outcome != new_outcome:
                changes[field] = transition
        if (changes or old["failed"] != new["failed"]) and len(examples) < max_examples:
            examples.append({"image": key[0], "variant": key[1], "changes": changes,
                             "baseline_error": old["failed"], "candidate_error": new["failed"]})
    return {"cases": len(before), "transitions": dict(transitions),
            "per_field": {field: dict(counts) for field, counts in per_field.items()},
            "improved": improved, "regressed": regressed, "new_incorrect": new_incorrect,
            "new_false_empty": new_false_empty, "new_observation_errors": new_errors,
            "examples": examples,
            "baseline_profile": baseline.get("profile"), "candidate_profile": candidate.get("profile"),
            "baseline_code_sha256": baseline.get("code_sha256"),
            "candidate_code_sha256": candidate.get("code_sha256"),
            "limitation": "Paired recorded-frame results do not validate unseen scenes or authorize live profile changes."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        result = compare(json.loads(Path(args.baseline).read_text(encoding="utf-8")),
                         json.loads(Path(args.candidate).read_text(encoding="utf-8")),
                         max_examples=args.max_examples)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(any(result[name] for name in ("regressed", "new_incorrect", "new_false_empty", "new_observation_errors")))


if __name__ == "__main__":
    raise SystemExit(main())
