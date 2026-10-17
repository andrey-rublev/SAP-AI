import hashlib
import io
import json
import os

import numpy as np
from PIL import Image
import pytest

from tools.train_desktop_species import UNKNOWN, SyntheticCrops, fit_sprite, model_for, real_results, render, train
from tools import train_desktop_species as species


def test_fit_recovers_partial_mirrored_sprite_geometry():
    pytest.importorskip("cv2")
    sprite = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    array = np.asarray(sprite).copy()
    rng = np.random.default_rng(8)
    array[3:20, 2:17, :3] = rng.integers(0, 255, (17, 15, 3))
    array[3:20, 2:17, 3] = 255
    sprite = Image.fromarray(array)
    background = Image.new("RGB", (13, 11), "#aabbaa")
    expected = dict(size=24, x=5, y=8, width=13, height=11, mirror=True)
    crop = render(sprite, background, expected)
    actual = fit_sprite(crop, sprite, background, sizes=(20, 24, 28))
    assert {key: actual[key] for key in expected} == expected
    assert actual["mse"] < 1e-6


@pytest.fixture
def plan(tmp_path):
    entries, backgrounds = [], []
    for i, split in enumerate(("train", "validation")):
        background = tmp_path / f"background-{split}.png"
        Image.new("RGB", (24, 12), (50 + i * 80, 100, 50)).save(background)
        backgrounds.append({"path": str(background), "split": split})
        for label, color in ((UNKNOWN, "red"), ("fish", "blue")):
            path = tmp_path / f"sprite-{split}-{label}.png"
            Image.new("RGBA", (24, 24), color).save(path)
            entries.append({"path": str(path), "split": split, "label": label,
                            "fit": dict(size=24, x=0, y=0, width=24, height=12, mirror=False)})
    return {"labels": [UNKNOWN, "fish"], "entries": entries, "backgrounds": backgrounds}


def test_synthetic_samples_replay_and_split_source_assets(plan):
    training, validation = SyntheticCrops(plan, "train"), SyntheticCrops(plan, "validation")
    assert {x["path"] for x in training.entries}.isdisjoint({x["path"] for x in validation.entries})
    first, label = training.sample(17)
    repeated, repeated_label = training.sample(17)
    assert np.array_equal(first, repeated)
    assert label == repeated_label == 1
    assert first.shape == (3, 32, 64)
    assert 0 <= first.min() <= first.max() <= 1
    # Semantic split is in the seed, not a shared mutable RNG advanced by callers.
    training.sample(900)
    assert np.array_equal(first, training.sample(17)[0])
    assert not np.array_equal(first, validation.sample(17)[0])


def test_incomplete_or_invalid_split_rejected(plan):
    with pytest.raises(ValueError, match="invalid"):
        SyntheticCrops(plan, "heldout")
    plan["entries"] = [x for x in plan["entries"] if x["label"] != "fish"]
    with pytest.raises(ValueError, match="incomplete"):
        SyntheticCrops(plan, "train")


def test_real_report_counts_wrong_unknown_acceptance_and_excludes_nonshop(tmp_path):
    torch = pytest.importorskip("torch")
    image_path = tmp_path / "frame.png"
    pixels = np.random.default_rng(42).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(np.tile(pixels, (1, 2, 1))).save(image_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"image_size": [32, 16], "shop": [{"portrait": [0, 0, 16, 16]}],
                                        "team": [{"portrait": [16, 0, 16, 16]}]}))
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"label_source": "synthetic test", "frames": [
        {"image": "frame.png", "phase": "shop", "shop": [UNKNOWN], "team": ["fish"]},
        {"image": "not-read.png", "phase": "battle"}]}))
    observed_inputs = []
    class AlwaysFish(torch.nn.Module):
        def forward(self, x):
            observed_inputs.append(x.numpy()[0].copy())
            return torch.tensor([[0., 10.]])
    plan = {"profile": str(profile_path), "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "labels": [UNKNOWN, "fish"]}
    report = real_results(AlwaysFish(), plan, labels_path)
    assert report["samples"] == 2
    assert report["unique_pixel_crops"] == 1
    assert report["correct_accepted"] == report["known_correct"] == 1
    assert report["unknown_false_accepts"] == 1
    expected_input = np.asarray(Image.fromarray(pixels).resize((64, 32), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255
    assert np.array_equal(observed_inputs[0], expected_input)
    assert report["excluded_non_shop_frames"] == ["not-read.png"]
    rejected = real_results(AlwaysFish(), plan, labels_path, threshold=1.0)
    assert rejected["accepted"] == rejected["known_correct"] == 0
    assert rejected["unknown_false_accepts"] == 0
    profile_path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        real_results(AlwaysFish(), plan, labels_path)


def test_tiny_model_output_matches_class_count():
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        assert tuple(model_for(10)(torch.zeros(2, 3, 32, 64)).shape) == (2, 10)


@pytest.mark.parametrize("kwargs", [{"seconds": 0}, {"max_steps": 0}, {"batch_size": 0}, {"threads": 0},
                                    {"seconds": float("nan")}, {"seconds": float("inf")},
                                    {"seconds": -float("inf")}, {"max_steps": 1.5}, {"threads": True}])
def test_training_bounds_rejected_before_reading_files(kwargs):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="positive"):
        train("missing-plan", "missing-labels", **kwargs)


def test_preprocessing_is_explicit_bilinear_and_shared(plan, monkeypatch):
    pixels = np.random.default_rng(17).integers(0, 255, (69, 125, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    expected = np.asarray(image.resize((64, 32), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255
    assert np.array_equal(species.preprocess(image), expected)
    assert not np.array_equal(species.preprocess(image), np.asarray(image.resize((64, 32)), dtype=np.float32).transpose(2, 0, 1) / 255)
    sentinel = np.zeros((3, 32, 64), dtype=np.float32)
    monkeypatch.setattr(species, "preprocess", lambda image: sentinel)
    assert SyntheticCrops(plan, "train").sample(1)[0] is sentinel


@pytest.fixture
def private(tmp_path, monkeypatch):
    root = tmp_path / ".local"
    root.mkdir()
    monkeypatch.setattr(species, "PRIVATE_ROOT", root)
    return root


def test_output_rejects_outside_private_root_and_traversal(private):
    for path in (private.parent / "model.pt", private / ".." / "model.pt"):
        with pytest.raises(ValueError, match="private"):
            species.dump(path, {})
    species.dump(private / "report.json", {"safe": True})
    assert json.loads((private / "report.json").read_text()) == {"safe": True}


@pytest.mark.parametrize("filename", ["model.pt", "progress.json", "report.json"])
def test_linked_output_rejected_before_training(private, filename):
    pytest.importorskip("torch")
    target = private.parent / "external.txt"
    target.write_text("untouched")
    os.link(target, private / filename)
    with pytest.raises(ValueError, match="unlinked"):
        species.train(private / "missing-plan.json", "unused", max_steps=1)
    assert target.read_text() == "untouched"


def test_symlinked_parent_rejected(private):
    external = private.parent / "external"
    external.mkdir()
    try:
        (private / "redirect").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable for this test account")
    with pytest.raises(ValueError, match="linked"):
        species.dump(private / "redirect" / "report.json", {})
    assert not list(external.iterdir())


def test_windows_reparse_parent_rejected_without_symlink_privilege(private, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    redirect = private / "junction"
    redirect.mkdir()
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == redirect:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400, st_nlink=result.st_nlink)
        return result
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="linked"):
        species.dump(redirect / "model.pt", {})
    assert not (redirect / "model.pt").exists()


def test_failed_atomic_writer_preserves_existing_output(private):
    target = private / "report.json"
    target.write_text("original")
    def fail(handle):
        handle.write(b"partial")
        raise RuntimeError("write failed")
    with pytest.raises(RuntimeError):
        species.write_output(target, fail)
    assert target.read_text() == "original"
    assert not list(private.glob("*.tmp"))


def test_train_records_loaded_plan_and_source_hashes(private, monkeypatch):
    torch = pytest.importorskip("torch")
    plan_path = private / "plan.json"
    original = json.dumps({"labels": [UNKNOWN, "fish"], "limitations": []}).encode()
    plan_path.write_bytes(original)
    loaded_source_hash = species.IMPLEMENTATION_SHA256
    class Crops:
        def __init__(self, plan, split):
            # Simulate a file edit after loading the plan; its attribution must stay fixed.
            plan_path.write_text("{}")
        def sample(self, index):
            return np.zeros((3, 32, 64), dtype=np.float32), 0
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(2))
        def forward(self, x):
            return self.logits.unsqueeze(0).expand(x.shape[0], -1)
    monkeypatch.setattr(species, "SyntheticCrops", Crops)
    monkeypatch.setattr(species, "model_for", lambda count: Tiny())
    monkeypatch.setattr(species, "real_results", lambda *args: {"test": True})
    monkeypatch.setattr(species, "digest", lambda path: pytest.fail("must not hash files again after loading"))
    report = species.train(plan_path, "unused", max_steps=1, batch_size=1, threads=1)
    saved = torch.load(private / "model.pt", weights_only=True)
    for record in (report, saved):
        assert record["plan_sha256"] == hashlib.sha256(original).hexdigest()
        assert record["implementation_sha256"] == loaded_source_hash
        assert record["preprocessing"] == species.PREPROCESSING


def test_evaluate_refuses_overwrite_without_loading_model(private):
    pytest.importorskip("torch")
    target = private / "original-report.json"
    target.write_text("preserved")
    with pytest.raises(ValueError, match="new report"):
        species.evaluate_checkpoint("missing", "missing", "missing", target)
    assert target.read_text() == "preserved"


def test_prepare_accepts_alternate_portraits_and_keeps_first_geometry_reference(private, monkeypatch):
    for name, color in (("empty0.png", 20), ("empty1.png", 40),
                        ("fish.png", 80), ("fish-alternate.png", 120)):
        Image.new("RGB", (12, 8), (color,) * 3).save(private / name)
    profile = {"version": 1, "image_size": [24, 8],
               "shop": [{"portrait": [0, 0, 12, 8], "empty_template": "empty0.png",
                          "species_templates": {"fish": ["fish.png", "fish-alternate.png"]}}],
               "team": [{"portrait": [12, 0, 12, 8], "empty_template": "empty1.png",
                          "species_templates": {}}]}
    profile_path = private / "profile.json"
    profile_path.write_text(json.dumps(profile))
    for directory, label in (("assets", "fish"), ("negatives", "ant")):
        location = private / directory
        location.mkdir()
        textures = []
        for index, color in enumerate(("red", "blue")):
            name = f"{index}.png"
            sprite = Image.new("RGBA", (16, 16), color)
            sprite.save(location / name)
            textures.append({"candidate_species": label, "file": name, "path_id": index,
                             "rgba_sha256": hashlib.sha256(sprite.tobytes()).hexdigest()})
        (location / "manifest.json").write_text(json.dumps({"textures": textures}))
    fits = []
    def fit(crop, sprite, background):
        fits.append(np.asarray(crop).copy())
        return dict(size=16, x=0, y=0, width=12, height=8, mirror=False, mse=.01)
    monkeypatch.setattr(species, "fit_sprite", fit)
    result = species.prepare(profile_path, private / "assets", private / "negatives", private / "prepared")
    assert all(np.all(crop == 80) for crop in fits) and len(fits) == 2
    assert result["references"]["fish"] == {"path": str(private / "fish.png"),
                                               "sha256": species.digest(private / "fish.png")}
    assert len(result["entries"]) == 4 and result["labels"] == [UNKNOWN, "fish"]
    assert (private / "prepared/profile-snapshot.json").read_bytes() == profile_path.read_bytes()


@pytest.fixture
def real_manifest(private):
    samples = []
    rng = np.random.default_rng(497)
    for split_index, split in enumerate(species.REAL_SPLITS):
        for index, expected in enumerate(("fish", UNKNOWN, species.EMPTY)):
            name = f"{split}-{index}.png"
            Image.fromarray(rng.integers(0, 255, (32, 64, 3), dtype=np.uint8)).save(private / name)
            samples.append({"image": name, "rect": [0, 0, 64, 32], "expected": expected,
                            "occupied": expected != species.EMPTY, "split": split,
                            "session": split, "pose_group": f"{split}-{index}",
                            "label_source": "Independently labeled synthetic test image",
                            **({"source_species": ("ant", "beaver", "horse")[split_index]} if expected == UNKNOWN else {})})
    value = {"version": 1, "known_species": ["fish"], "samples": samples, "reference_images": []}
    path = private / "real.json"
    path.write_text(json.dumps(value))
    return path, value


def _save_manifest(fixture):
    path, value = fixture
    path.write_text(json.dumps(value))
    return path


def test_real_crops_deduplicate_model_inputs_and_keep_occupancy_separate(real_manifest):
    path, value = real_manifest
    value["samples"].append(dict(value["samples"][0]))
    data = species.load_real_crops(_save_manifest(real_manifest))
    assert data["provenance"]["sample_count"] == 10
    assert data["provenance"]["unique_input_count"] == 9
    assert data["provenance"]["counts"]["test"] == {UNKNOWN: 1, species.EMPTY: 1, "fish": 1}
    assert all(row["occupied"] == (row["expected"] != species.EMPTY) for row in data["samples"])
    assert data["provenance"]["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["session", "pose_group"])
def test_real_split_rejects_shared_declared_groups(real_manifest, field):
    _, value = real_manifest
    value["samples"][3][field] = value["samples"][0][field]
    with pytest.raises(ValueError, match=f"{field} crosses"):
        species.load_real_crops(_save_manifest(real_manifest))


def test_real_split_rejects_different_filenames_with_equal_frame_pixels(real_manifest):
    path, _ = real_manifest
    (path.parent / "calibration-0.png").write_bytes((path.parent / "train-0.png").read_bytes())
    with pytest.raises(ValueError, match="frame crosses"):
        species.load_real_crops(path)


def test_real_split_rejects_equal_crops_inside_different_full_frames(real_manifest):
    path, value = real_manifest
    image = Image.open(path.parent / "train-0.png")
    bigger = Image.new("RGB", (65, 32), "red")
    bigger.paste(image)
    bigger.save(path.parent / "calibration-0.png")
    with pytest.raises(ValueError, match="crop crosses"):
        species.load_real_crops(path)


def test_real_split_rejects_effectively_identical_resized_inputs(real_manifest):
    path, value = real_manifest
    Image.new("RGB", (64, 32), "red").save(path.parent / "train-0.png")
    Image.new("RGB", (128, 64), "red").save(path.parent / "calibration-0.png")
    value["samples"][3]["rect"] = [0, 0, 128, 64]
    with pytest.raises(ValueError, match="input crosses"):
        species.load_real_crops(_save_manifest(real_manifest))


def test_real_split_rejects_near_duplicate_pose_without_exact_hash_overlap(real_manifest):
    path, _ = real_manifest
    image = np.asarray(Image.open(path.parent / "train-0.png")).copy()
    Image.fromarray(np.minimum(image.astype(int) + 1, 255).astype(np.uint8)).save(path.parent / "calibration-0.png")
    with pytest.raises(ValueError, match="near-duplicate"):
        species.load_real_crops(path)


def test_real_split_rejects_mirrored_training_augmentation_in_holdout(real_manifest):
    path, _ = real_manifest
    with Image.open(path.parent / "train-0.png") as original:
        original.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(path.parent / "calibration-0.png")
    with pytest.raises(ValueError, match="mirror crosses"):
        species.load_real_crops(path)


def test_real_split_rejects_unknown_species_shared_with_calibration(real_manifest):
    _, value = real_manifest
    value["samples"][4]["source_species"] = value["samples"][1]["source_species"]
    with pytest.raises(ValueError, match="unknown source species crosses"):
        species.load_real_crops(_save_manifest(real_manifest))


@pytest.mark.parametrize("change,match", [({"occupied": False}, "occupied"),
                                          ({"expected": "Fish"}, "label"),
                                          ({"pose_group": ""}, "pose_group"),
                                          ({"label_source": ""}, "label_source"),
                                          ({"rect": [-1, 0, 64, 32]}, "rectangle"),
                                          ({"rect": [0, 0, 65, 32]}, "rectangle")])
def test_real_manifest_fails_closed_on_invalid_labels_and_crop_geometry(real_manifest, change, match):
    real_manifest[1]["samples"][0].update(change)
    with pytest.raises(ValueError, match=match):
        species.load_real_crops(_save_manifest(real_manifest))


def test_real_manifest_rejects_conflicting_labels_for_identical_input(real_manifest):
    _, value = real_manifest
    duplicate = dict(value["samples"][0], expected=UNKNOWN, source_species="cricket")
    value["samples"].insert(1, duplicate)
    with pytest.raises(ValueError, match="conflicting labels"):
        species.load_real_crops(_save_manifest(real_manifest))


def test_real_manifest_needs_explicit_empty_and_occupied_unknown_per_split(real_manifest):
    _, value = real_manifest
    value["samples"] = [row for row in value["samples"] if not (row["split"] == "test" and row["expected"] == species.EMPTY)]
    with pytest.raises(ValueError, match="incomplete real test"):
        species.load_real_crops(_save_manifest(real_manifest))


def test_real_manifest_rejects_template_reference_in_holdout(real_manifest):
    _, value = real_manifest
    value["reference_images"] = ["test-0.png"]
    with pytest.raises(ValueError, match="calibration/template reference"):
        species.load_real_crops(_save_manifest(real_manifest))


def test_real_metrics_do_not_count_empty_or_unknown_as_correct_known_species():
    rows = [{"expected": expected, "top_label": prediction, "confidence": .99}
            for expected, prediction in (("fish", "fish"), ("fish", "ant"),
                                         (UNKNOWN, "fish"), (species.EMPTY, "fish"),
                                         (UNKNOWN, species.EMPTY), (species.EMPTY, species.EMPTY))]
    report = species.score_real_predictions(rows, .95)
    assert report["known_samples"] == 2 and report["known_correct"] == 1
    assert report["known_wrong_accepts"] == report["unknown_false_accepts"] == report["empty_false_accepts"] == 1
    assert report["occupied_false_empty"] == report["empty_correct"] == 1
    assert report["unknown_samples"] == report["empty_samples"] == 2


def test_real_calibration_raises_threshold_on_unknown_false_accept_without_test_feedback():
    rows = [{"expected": "fish", "top_label": "fish", "confidence": .999},
            {"expected": UNKNOWN, "top_label": "fish", "confidence": .99},
            {"expected": species.EMPTY, "top_label": species.EMPTY, "confidence": .999}]
    calibrated = species.calibrate_real_threshold(rows)
    assert calibrated["status"] == "calibrated"
    assert .99 < calibrated["threshold"] < .999
    assert calibrated["report"]["known_correct"] == 1
    assert calibrated["report"]["unknown_false_accepts"] == 0
    rows[1]["confidence"] = 1.
    assert species.calibrate_real_threshold(rows)["status"] == "unsafe_at_all_thresholds"
    with pytest.raises(ValueError, match=".95"):
        species.score_real_predictions(rows, .94)


@pytest.mark.parametrize("kwargs", [{"seconds": 2701}, {"threads": 4}, {"seconds": float("nan")},
                                    {"batch_size": 0}, {"max_steps": True}])
def test_real_training_limits_fail_before_reading_files(kwargs):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="bounds"):
        species.train_real("missing", "missing", **kwargs)


def test_real_training_one_step_records_disjoint_unique_data_without_promoting(real_manifest, private):
    pytest.importorskip("torch")
    report = species.train_real(real_manifest[0], private / "run", seconds=2, max_steps=1, threads=1, batch_size=3)
    assert report["steps"] == 1 and report["eligible_for_promotion"] is False
    assert report["test"]["samples"] == report["calibration"]["report"]["samples"] == 3
    assert report["near_duplicate_mae"] == .01
    assert report["preprocessing"] == species.PREPROCESSING
    assert (private / "run/model.pt").is_file()
    assert (private / "run/data-audit.json").is_file()


def test_real_final_test_stays_unread_until_frozen_checkpoint_and_threshold(real_manifest, private):
    pytest.importorskip("torch")
    path, value = real_manifest
    complete = json.loads(json.dumps(value))
    value["samples"] = [row for row in value["samples"] if row["split"] != "test"]
    value["test_withheld"] = True
    _save_manifest(real_manifest)
    report = species.train_real(path, private / "frozen", max_steps=1, threads=1, batch_size=3)
    assert report["test_state"] == "withheld"
    assert report["test"] is report["test_at_original_threshold"] is None
    # Newly labeled test is attached after training; it cannot alter the frozen
    # calibration/training rows, class order, preprocessing or threshold.
    final_manifest = private / "final.json"
    final_manifest.write_text(json.dumps(complete))
    final = species.evaluate_real_checkpoint(path, final_manifest, private / "frozen/model.pt", private / "final-report.json", threads=1)
    assert final["test"]["samples"] == 3
    assert final["test"]["threshold"] == report["calibration"]["threshold"]
    assert final["training_manifest_sha256"] == report["manifest_sha256"]
    complete["samples"][0]["label_source"] = "Changed after test was opened"
    final_manifest.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="changed before final test"):
        species.evaluate_real_checkpoint(path, final_manifest, private / "frozen/model.pt", private / "changed.json", threads=1)


def test_withheld_real_manifest_rejects_accidentally_included_final_test(real_manifest):
    real_manifest[1]["test_withheld"] = True
    with pytest.raises(ValueError, match="cannot contain test"):
        species.load_real_crops(_save_manifest(real_manifest))


def test_real_frozen_source_hashes_detect_changed_images_before_opening_test(real_manifest, private):
    torch = pytest.importorskip("torch")
    path, value = real_manifest
    value["samples"] = [row for row in value["samples"] if row["split"] != "test"]
    value["test_withheld"] = True
    _save_manifest(real_manifest)
    species.train_real(path, private / "fixed", max_steps=1, threads=1, batch_size=3)
    source = private / "train-0.png"
    with Image.open(source) as image:
        image = image.copy()
    image.putpixel((0, 0), (0, 0, 0))
    image.save(source)
    # Test path deliberately does not exist: changed sources must fail first.
    with pytest.raises(ValueError, match="changed since the frozen checkpoint"):
        species.evaluate_real_checkpoint(path, private / "unopened-test.json", private / "fixed/model.pt", private / "unwritten.json")
    assert not (private / "unwritten.json").exists()


def test_real_checkpoint_without_embedded_source_provenance_cannot_use_companion_report(real_manifest, private):
    torch = pytest.importorskip("torch")
    path, value = real_manifest
    value["samples"] = [row for row in value["samples"] if row["split"] != "test"]
    value["test_withheld"] = True
    _save_manifest(real_manifest)
    species.train_real(path, private / "fixed", max_steps=1, threads=1, batch_size=3)
    model_path = private / "fixed/model.pt"
    checkpoint = torch.load(model_path, weights_only=True)
    checkpoint.pop("data_provenance")
    torch.save(checkpoint, model_path)
    assert (private / "fixed/report.json").is_file()
    with pytest.raises(ValueError, match="lacks frozen source/reference provenance; retrain"):
        species.evaluate_real_checkpoint(path, private / "unopened-test.json", model_path, private / "unwritten.json")


@pytest.fixture
def frozen_real_case(real_manifest, private):
    pytest.importorskip("torch")
    path, value = real_manifest
    complete = json.loads(json.dumps(value))
    value["samples"] = [row for row in value["samples"] if row["split"] != "test"]
    value["test_withheld"] = True
    _save_manifest(real_manifest)
    species.train_real(path, private / "fixed", max_steps=1, threads=1, batch_size=3)
    final_manifest = private / "final.json"
    final_manifest.write_text(json.dumps(complete))
    return path, complete, final_manifest, private / "fixed/model.pt"


@pytest.mark.parametrize("change", ["label_source", "pose_group", "rect", "references", "state",
                                    "threshold_low", "threshold_nan", "threshold_bool", "threshold_string",
                                    "status_invalid", "status_unsafe"])
def test_real_preflight_opens_no_final_test_image_on_invalid_declarations_or_checkpoint(frozen_real_case, private, monkeypatch, change):
    import torch
    training_path, complete, final_path, model_path = frozen_real_case
    if change in ("label_source", "pose_group", "rect"):
        complete["samples"][0][change] = [0, 0, 63, 32] if change == "rect" else "changed after training"
    elif change == "references":
        complete["reference_images"] = ["test-0.png"]
    else:
        checkpoint = torch.load(model_path, weights_only=True)
        if change == "state":
            checkpoint["state_dict"]["0.weight"] = torch.zeros(1)
        elif change.startswith("threshold"):
            checkpoint["calibration_threshold"] = {"threshold_low": .94, "threshold_nan": float("nan"),
                                                   "threshold_bool": True, "threshold_string": ".99"}[change]
        else:
            checkpoint["calibration_status"] = "invalid" if change == "status_invalid" else "unsafe_at_all_thresholds"
        torch.save(checkpoint, model_path)
    final_path.write_text(json.dumps(complete))
    test_pixels = {(private / row["image"]).read_bytes() for row in complete["samples"] if row["split"] == "test"}
    opened = []
    original_open = Image.open
    def watch_image_open(value, *args, **kwargs):
        if isinstance(value, io.BytesIO) and value.getvalue() in test_pixels:
            opened.append("final test")
        return original_open(value, *args, **kwargs)
    monkeypatch.setattr(Image, "open", watch_image_open)
    with pytest.raises((ValueError, RuntimeError)):
        species.evaluate_real_checkpoint(training_path, final_path, model_path, private / "refused.json", threads=1)
    assert opened == []
    assert not (private / "refused.json").exists()


def test_real_final_loader_uses_preflighted_manifest_bytes_and_reports_missing_classes(frozen_real_case, private, monkeypatch):
    training_path, complete, final_path, model_path = frozen_real_case
    original_bytes = final_path.read_bytes()
    original_loader = species.load_real_crops
    def mutate_after_preflight(path, **kwargs):
        if path == final_path:
            complete["samples"][0]["label_source"] = "Mutated during evaluation"
            final_path.write_text(json.dumps(complete))
            assert kwargs["_manifest_bytes"] == original_bytes
        return original_loader(path, **kwargs)
    monkeypatch.setattr(species, "load_real_crops", mutate_after_preflight)
    report = species.evaluate_real_checkpoint(training_path, final_path, model_path, private / "final-report.json", threads=1)
    assert report["manifest_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert report["test"]["samples"] == 3
    assert report["missing_known_species"] == {"calibration": [], "test": []}


def test_real_preflight_rejects_relocated_relative_sources_without_opening_final_images(frozen_real_case, private, monkeypatch):
    training_path, complete, _, model_path = frozen_real_case
    relocated = private / "relocated"
    relocated.mkdir()
    for row in complete["samples"]:
        (relocated / row["image"]).write_bytes((private / row["image"]).read_bytes())
    final_path = relocated / "final.json"
    final_path.write_text(json.dumps(complete))
    test_pixels = {(relocated / row["image"]).read_bytes() for row in complete["samples"] if row["split"] == "test"}
    original_open = Image.open
    def watch_image_open(value, *args, **kwargs):
        if isinstance(value, io.BytesIO) and value.getvalue() in test_pixels:
            pytest.fail("relocated final-test image opened before source paths were validated")
        return original_open(value, *args, **kwargs)
    monkeypatch.setattr(Image, "open", watch_image_open)
    with pytest.raises(ValueError, match="source or reference paths changed before final test"):
        species.evaluate_real_checkpoint(training_path, final_path, model_path, private / "refused-relocated.json", threads=1)
    assert not (private / "refused-relocated.json").exists()
