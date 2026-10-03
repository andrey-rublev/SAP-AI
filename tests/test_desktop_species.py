import hashlib
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
