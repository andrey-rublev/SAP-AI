"""Offline evaluation contracts; no desktop capture or Tesseract process."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools import eval_desktop_frames as study


@pytest.fixture
def source(tmp_path):
    frame = np.full((12, 16, 3), 220, np.uint8)
    frame[:3, :3] = 10
    Image.fromarray(frame).save(tmp_path / "frame.png")
    Image.fromarray(frame[:3, :3]).save(tmp_path / "phase.png")
    profile = {"version": 1, "image_size": [16, 12], "calibrated": False,
               "phase_templates": [{"phase": "shop", "region": [0, 0, 3, 3],
                                    "template": "phase.png", "max_distance": .1}],
               "hud": {"gold": [4, 4, 3, 3]}}
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    manifest = {"version": 1, "frames": [{"image": "frame.png", "labels": {"phase": "shop", "gold": 10},
                                         "label_source": "independent fixture definition"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def run(source, **kwargs):
    kwargs.setdefault("ocr", lambda crop: "10")
    return study.run(source / "manifest.json", source / "profile.json", source / "reports", **kwargs)


def test_transforms_are_deterministic_preserve_shape_and_identity():
    frame = np.arange(15 * 17 * 3, dtype=np.uint8).reshape(15, 17, 3)
    variants = study.variants()
    assert np.array_equal(study.perturb(frame, variants[0]), frame)
    for item in study.variants(True):
        result = study.perturb(frame, item)
        assert result.shape == frame.shape and result.dtype == np.uint8
        assert np.array_equal(result, study.perturb(frame, item))
    assert len({v["name"] for v in study.variants(True)}) == 135


@pytest.mark.parametrize("labels", [
    {}, {"phase": "nonsense"}, {"phase": "shop", "gold": None},
    {"phase": "shop", "gold": True}, {"phase": "shop", "turn": 0},
    {"phase": "shop", "team.0.occupied": None},
    {"phase": "shop", "team.0.attack": 3},
    {"phase": "shop", "team.0.occupied": False, "team.0.health": 3},
    {"phase": "shop", "team.0.species": "guess"},
    {"phase": "battle", "gold": 10},
])
def test_ambiguous_or_inconsistent_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        study.validate_labels(labels)


def test_occupancy_errors_and_unknowns_are_distinct_and_unlabeled_fields_excluded():
    counts, fields, issues = study.score(
        {"phase": "shop", "gold": 10, "team.0.occupied": True, "team.0.attack": 2, "team.1.occupied": False},
        {"phase": "shop", "gold": None, "wins": 99, "team": [{"occupied": False}, {"occupied": None}]})
    assert counts == {"correct": 1, "unknown": 3, "incorrect": 1, "unsafe_false_empty": 1}
    assert fields["team.0.occupied"] == "incorrect"
    assert any(i["unsafe_false_empty"] for i in issues)
    assert "wins" not in fields


def test_boolean_does_not_equal_numeric_reading():
    counts, _, _ = study.score({"gold": 1}, {"gold": True})
    assert counts["incorrect"] == 1


@pytest.mark.parametrize("row,index,species", [("team", 0, "pig"), ("shop", 4, "green ant")])
def test_species_labels_accept_canonical_known_text_only_on_occupied_slots(row, index, species):
    labels = {"phase": "shop", f"{row}.{index}.occupied": True,
              f"{row}.{index}.species": species}
    study.validate_labels(labels)


@pytest.mark.parametrize("species", [None, 1, True, [], "", " ", "Pig", " pig", "green  ant"])
def test_species_labels_reject_unknown_or_noncanonical_text(species):
    with pytest.raises(ValueError):
        study.validate_labels({"phase": "shop", "team.0.occupied": True,
                               "team.0.species": species})


@pytest.mark.parametrize("occupied", [False, None])
def test_species_labels_require_known_occupied_slot(occupied):
    with pytest.raises(ValueError):
        study.validate_labels({"phase": "shop", "team.0.occupied": occupied,
                               "team.0.species": "pig"})


@pytest.mark.parametrize("expected,portrait,observed,outcome", [
    ("pig", 100, "pig", "correct"),
    ("ant", 160, None, "unknown"),
    ("ant", 100, "pig", "incorrect"),
])
def test_real_perceptor_species_labels_distinguish_match_unknown_and_false_pig(
        source, expected, portrait, observed, outcome):
    # The non-Pig label is independent of the profile's deliberately limited
    # Pig reference; a false visual match must count as wrong, not unknown.
    frame_path = source / "frame.png"
    frame = np.asarray(Image.open(frame_path).convert("RGB")).copy()
    frame[4:8, 8:12] = portrait
    Image.fromarray(frame).save(frame_path)
    Image.new("RGB", (4, 4), (100, 100, 100)).save(source / "pig.png")
    Image.new("RGB", (4, 4), (0, 0, 0)).save(source / "empty.png")
    profile_path = source / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile["team"] = [{"portrait": [8, 4, 4, 4], "attack": [4, 8, 1, 1],
                        "health": [5, 8, 1, 1], "empty_template": "empty.png",
                        "species_templates": {"pig": "pig.png"}}]
    profile_path.write_text(json.dumps(profile))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["labels"] = {"phase": "shop", "team.0.occupied": True,
                                        "team.0.species": expected}
    manifest_path.write_text(json.dumps(manifest))
    path, report = run(source, ocr=lambda crop: "2", max_cases=1, max_examples=0)
    assert report["cases"][0]["observed"]["team"][0]["occupied"] is True
    assert report["cases"][0]["observed"]["team"][0]["species"] == observed
    assert report["counts"] == {"correct": 2 + (outcome == "correct"),
                                "unknown": int(outcome == "unknown"),
                                "incorrect": int(outcome == "incorrect"), "unsafe_false_empty": 0}
    assert report["per_field"]["team.0.species"] == {
        name: int(name == outcome) for name in ("correct", "unknown", "incorrect")
    }
    assert report["species_confusion"] == {
        "team.0.species": {expected: {observed or "<unknown>": 1}}
    }
    assert report["observation_errors"] == 0
    assert json.loads(path.read_text())["species_confusion"] == report["species_confusion"]
    assert (path.parent / "profile/pig.png").is_file()


def test_real_perceptor_uses_injected_cached_ocr_and_checkpoints_provenance(source):
    calls = []
    path, report = run(source, ocr=lambda crop: calls.append(crop.copy()) or "10", max_cases=2)
    assert report["status"] == "case_limit" and len(report["cases"]) == 2
    assert report["counts"]["correct"] == 4
    assert report["ocr_calls"] == len(calls) == 2
    saved = json.loads(path.read_text())
    assert saved == report
    assert report["profile"]["sha256"] == study.digest((source / "profile.json").read_bytes())
    assert report["sources"][0]["sha256"] == study.digest((source / "frame.png").read_bytes())
    assert "desktop_runtime.py" in report["code_sha256"]
    assert (path.parent / "profile/phase.png").read_bytes() == (source / "phase.png").read_bytes()


def test_snapshot_is_immutable_when_original_calibration_changes(source):
    def ocr(crop):
        (source / "profile.json").write_text("changed while running")
        (source / "phase.png").write_bytes(b"changed while running")
        (source / "frame.png").write_bytes(b"changed while running")
        return "10"
    path, report = run(source, ocr=ocr, max_cases=2)
    assert report["counts"]["correct"] == 4
    assert json.loads((path.parent / "profile/profile-snapshot.json").read_text())["version"] == 1
    with Image.open(path.parent / "profile/phase.png") as image:
        assert image.size == (3, 3)


def test_snapshot_freezes_every_alternate_species_reference_before_ocr(source):
    frame_path = source / "frame.png"
    with Image.open(frame_path) as image:
        frame = np.asarray(image.convert("RGB")).copy()
    frame[4:8, 8:12] = 140
    Image.fromarray(frame).save(frame_path)
    references = {"pig.png": 90, "pig-alternate.png": 140}
    original = {}
    for name, color in references.items():
        Image.new("RGB", (4, 4), (color,) * 3).save(source / name)
        original[name] = (source / name).read_bytes()
    profile_path = source / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile["team"] = [{"portrait": [8, 4, 4, 4], "attack": [4, 8, 1, 1],
                        "health": [5, 8, 1, 1], "species_templates": {"pig": list(references)}}]
    profile_path.write_text(json.dumps(profile))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["labels"] = {"phase": "shop", "team.0.occupied": True,
                                        "team.0.species": "pig"}
    manifest_path.write_text(json.dumps(manifest))
    def changing_ocr(crop):
        for name in references:
            (source / name).write_bytes(b"changed after freezing")
        return "2"
    path, report = run(source, ocr=changing_ocr, max_cases=2)
    assert report["counts"] == {"correct": 6, "unknown": 0, "incorrect": 0, "unsafe_false_empty": 0}
    for name, data in original.items():
        assert (path.parent / "profile" / name).read_bytes() == data
        assert report["profile"]["templates"][name] == study.digest(data)


def test_non_shop_phase_never_invokes_ocr(source):
    profile_path = source / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile["phase_templates"][0]["phase"] = "battle"
    profile_path.write_text(json.dumps(profile))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["labels"] = {"phase": "battle"}
    manifest_path.write_text(json.dumps(manifest))
    def forbidden(crop):
        pytest.fail("non-shop frames must not invoke OCR")
    _, report = run(source, ocr=forbidden)
    assert report["status"] == "complete" and report["ocr_calls"] == 0
    assert report["phase_confusion"] == {"battle": {"battle": 7}}


def test_failures_save_bounded_examples_but_all_cases_are_scored(source):
    path, report = run(source, ocr=lambda crop: "9", max_examples=2)
    assert len(report["cases"]) == 7 and report["examples_saved"] == 2
    assert len(list(path.parent.glob("failure-*.png"))) == 2
    assert report["counts"]["incorrect"] == 7
    assert report["per_field"]["gold"]["incorrect"] == 7
    assert len(report["per_variant"]) == 7


def test_ocr_timeout_is_an_error_not_a_correct_read(source):
    def failed(crop):
        raise TimeoutError("bounded OCR timed out")
    _, report = run(source, ocr=failed, max_cases=1, max_examples=0)
    assert report["counts"]["unknown"] == 2
    assert report["cases"][0]["error"] == "TimeoutError: bounded OCR timed out"
    assert report["examples_saved"] == 0
    assert report["observation_errors"] == 1
    assert report["phase_confusion"] == {"shop": {"observation_error": 1}}


def test_unknown_label_does_not_turn_observation_exception_into_success(source, monkeypatch):
    path = source / "manifest.json"
    data = json.loads(path.read_text())
    data["frames"][0]["labels"] = {"phase": "unknown"}
    path.write_text(json.dumps(data))
    def failed(self, frame):
        raise RuntimeError("broken observation")
    monkeypatch.setattr(study.Perceptor, "observe", failed)
    _, report = run(source, max_cases=1)
    assert report["observation_errors"] == 1
    assert report["counts"] == {"correct": 0, "unknown": 1, "incorrect": 0, "unsafe_false_empty": 0}
    assert report["cases"][0]["error"] == "RuntimeError: broken observation"
    assert report["phase_confusion"] == {"unknown": {"observation_error": 1}}


@pytest.mark.parametrize("errors,incorrect,expected", [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
def test_cli_exit_status_and_summary_report_failed_observations(monkeypatch, capsys, errors, incorrect, expected):
    report = {"status": "complete", "cases": [], "observation_errors": errors,
              "counts": {"correct": 0, "unknown": 1, "incorrect": incorrect, "unsafe_false_empty": 0}}
    monkeypatch.setattr(study, "run", lambda *args, **kwargs: (Path("report.json"), report))
    assert study.main(["--manifest", "labels.json", "--profile", "profile.json"]) == expected
    assert json.loads(capsys.readouterr().out)["observation_errors"] == errors


def test_stop_file_checkpoint_without_any_ocr(source):
    stop = source / "STOP"
    stop.touch()
    path, report = run(source, stop_file=stop)
    assert report["status"] == "stop_file" and report["ocr_calls"] == 0
    assert report["cases"] == [] and path.is_file()


def test_deadline_checked_during_observation_not_only_between_frames(source):
    ticks = iter([0, 0, 0, 0, 2, 2])
    path, report = run(source, max_seconds=1, clock=lambda: next(ticks, 2))
    assert report["status"] == "time_limit"
    assert report["cases"] == [] and report["ocr_calls"] == 0
    assert path.is_file()


def test_stop_requested_inside_ocr_does_not_process_more_cases(source):
    stop = source / "STOP"
    def ocr(crop):
        stop.touch()
        return "10"
    _, report = run(source, ocr=ocr, stop_file=stop)
    assert report["status"] == "stop_file" and len(report["cases"]) == 1


@pytest.mark.parametrize("kwargs", [{"max_seconds": 0}, {"max_seconds": float("nan")},
                                        {"max_cases": True}, {"max_examples": -1}])
def test_invalid_budgets_rejected(source, kwargs):
    with pytest.raises(ValueError):
        run(source, **kwargs)


def test_dimension_mismatch_rejected_without_resizing(source):
    Image.new("RGB", (17, 12)).save(source / "frame.png")
    with pytest.raises(ValueError, match="dimensions"):
        run(source)


def test_manifest_rejects_path_escape(source):
    path = source / "manifest.json"
    data = json.loads(path.read_text())
    data["frames"][0]["image"] = "../outside.png"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside"):
        study.load_manifest(path)


def test_repeated_runs_get_unique_output_paths(source):
    first, _ = run(source, max_cases=1)
    second, _ = run(source, max_cases=1)
    assert first != second and first.exists() and second.exists()
