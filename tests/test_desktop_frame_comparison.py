"""Paired perception experiments must expose regressions hidden by totals."""
from copy import deepcopy
import json

import pytest

from tools.compare_desktop_frames import compare, main
from tools.eval_desktop_frames import score


def report(observed, *, labels=None, error=None):
    labels = labels or {"phase": "shop", "gold": 10, "turn": 3}
    counts, _, _ = score(labels, observed, observation_failed=error is not None)
    return {"status": "complete", "manifest_sha256": "a" * 64,
            "sources": [{"image": "frame.png", "sha256": "b" * 64, "size": [16, 12],
                         "labels": labels, "label_source": "independent fixture"}],
            "variants": [{"name": "identity", "brightness": 1}], "python": "3.12",
            "packages": {"numpy": "2.1"}, "profile": {"sha256": "c" * 64},
            "code_sha256": {"desktop_vision.py": "d" * 64},
            "counts": counts, "observation_errors": int(error is not None),
            "cases": [{"image": "frame.png", "variant": "identity", "observed": observed,
                       "counts": counts, "error": error}]}


def test_equal_aggregate_totals_do_not_hide_lost_correct_reading():
    baseline = report({"phase": "shop", "gold": 10, "turn": None})
    candidate = report({"phase": "shop", "gold": None, "turn": 3})
    assert baseline["counts"] == candidate["counts"]
    candidate["profile"]["sha256"] = "e" * 64
    candidate["code_sha256"]["desktop_vision.py"] = "f" * 64
    result = compare(baseline, candidate)
    assert result["improved"] == result["regressed"] == 1
    assert result["per_field"]["gold"] == {"correct->unknown": 1}
    assert result["per_field"]["turn"] == {"unknown->correct": 1}
    assert result["candidate_profile"] != result["baseline_profile"]
    assert result["candidate_code_sha256"] != result["baseline_code_sha256"]


def test_new_wrong_reading_is_regression_even_when_another_field_improves():
    result = compare(report({"phase": "shop", "gold": 10, "turn": None}),
                     report({"phase": "shop", "gold": 11, "turn": 3}))
    assert result["improved"] == result["regressed"] == result["new_incorrect"] == 1
    assert result["transitions"]["correct->incorrect"] == 1


def test_false_empty_moving_between_slots_is_new_failure_despite_equal_totals():
    labels = {"phase": "shop", "team.0.occupied": True, "team.1.occupied": True}
    baseline = report({"phase": "shop", "team": [{"occupied": False}, {"occupied": True}]}, labels=labels)
    candidate = report({"phase": "shop", "team": [{"occupied": True}, {"occupied": False}]}, labels=labels)
    assert baseline["counts"] == candidate["counts"]
    result = compare(baseline, candidate)
    assert result["new_false_empty"] == 1
    assert result["regressed"] == 1


def test_observation_errors_are_separate_from_unknown_readings():
    result = compare(report({"phase": "unknown", "gold": None, "turn": None}),
                     report({"phase": "unknown"}, error="TimeoutError"))
    assert result["new_observation_errors"] == 1
    assert result["regressed"] == result["improved"] == result["new_incorrect"] == 0


@pytest.mark.parametrize("field", ["manifest_sha256", "sources", "variants", "python", "packages"])
def test_different_data_labels_perturbations_or_environment_cannot_be_paired(field):
    baseline = report({"phase": "shop", "gold": 10, "turn": 3})
    candidate = deepcopy(baseline)
    if field == "sources":
        candidate[field][0]["sha256"] = "e" * 64
    elif field == "variants":
        candidate[field][0]["brightness"] = .9
    elif field == "packages":
        candidate[field] = {"numpy": "2.2"}
    else:
        candidate[field] = "e" * 64 if field == "manifest_sha256" else "3.13"
    with pytest.raises(ValueError, match=field):
        compare(baseline, candidate)


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("manifest_sha256"),
    lambda r: r.update(manifest_sha256=None),
    lambda r: r.update(manifest_sha256="not-a-hash"),
    lambda r: r.update(python=None),
    lambda r: r.update(python=" "),
    lambda r: r.update(packages=None),
    lambda r: r.update(packages={}),
    lambda r: r.update(packages={"numpy": True}),
    lambda r: r.update(profile={}),
    lambda r: r.update(code_sha256={}),
    lambda r: r["sources"][0].pop("sha256"),
    lambda r: r["sources"][0].update(sha256=None),
    lambda r: r["sources"][0].pop("size"),
    lambda r: r["sources"][0].update(size=[True, 12]),
    lambda r: r["sources"][0].update(size=[16, 0]),
    lambda r: r["sources"][0].pop("label_source"),
    lambda r: r["sources"][0].update(label_source=" "),
])
def test_equal_missing_or_invalid_provenance_is_not_evidence_of_comparable_studies(mutation):
    baseline = report({"phase": "shop", "gold": 10, "turn": 3})
    mutation(baseline)
    with pytest.raises(ValueError):
        compare(baseline, deepcopy(baseline))


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(status="case_limit"), "completed"),
    (lambda r: r["cases"].clear(), "missing"),
    (lambda r: r["cases"].append(deepcopy(r["cases"][0])), "duplicate"),
    (lambda r: r["cases"][0].update(image="unlisted.png"), "unexpected"),
    (lambda r: r["sources"].append(deepcopy(r["sources"][0])), "unique"),
    (lambda r: r["variants"].append(deepcopy(r["variants"][0])), "unique"),
    (lambda r: r["counts"].update(correct=99), "totals"),
    (lambda r: r["cases"][0]["counts"].update(correct=99), "case counts"),
    (lambda r: r.update(observation_errors=99), "totals"),
])
def test_partial_duplicate_and_inconsistent_reports_are_rejected(mutation, match):
    baseline = report({"phase": "shop", "gold": 10, "turn": 3})
    candidate = deepcopy(baseline)
    # Decouple totals from the fixture's initially shared counts object.
    candidate["counts"] = dict(candidate["counts"])
    mutation(candidate)
    with pytest.raises(ValueError, match=match):
        compare(baseline, candidate)


def test_examples_are_bounded_without_losing_transition_counts():
    baseline = report({"phase": "shop", "gold": 10, "turn": 3})
    candidate = report({"phase": "shop", "gold": None, "turn": None})
    result = compare(baseline, candidate, max_examples=0)
    assert result["examples"] == [] and result["regressed"] == 2
    assert compare(baseline, candidate, max_examples=1)["examples"][0]["changes"] == {
        "gold": "correct->unknown", "turn": "correct->unknown"}


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_example_limit_is_rejected(value):
    with pytest.raises(ValueError, match="max_examples"):
        compare({}, {}, max_examples=value)


def test_cli_signals_regression_and_success_without_touching_input_reports(tmp_path, capsys):
    before = report({"phase": "shop", "gold": None, "turn": 3})
    after = report({"phase": "shop", "gold": 10, "turn": 3})
    paths = [tmp_path / "before.json", tmp_path / "after.json"]
    for path, value in zip(paths, (before, after)):
        path.write_text(json.dumps(value), encoding="utf-8")
    original = [path.read_bytes() for path in paths]
    assert main([str(paths[0]), str(paths[1])]) == 0
    assert json.loads(capsys.readouterr().out)["improved"] == 1
    assert main([str(paths[1]), str(paths[0])]) == 1
    assert json.loads(capsys.readouterr().out)["regressed"] == 1
    assert [path.read_bytes() for path in paths] == original
