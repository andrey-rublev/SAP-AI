"""Capture provenance binds loaded metadata and reference pixels without native IO."""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from desktop import validate_profile_assets
from desktop_state import Phase
from desktop_tooltip_layout import TooltipLayout
from desktop_vision import (FoodSlotConfig, NumberTemplate, NumericColor, Perceptor,
                            PhaseTemplate, Rect, SlotConfig, VisionProfile)
from tools import probe_desktop_hover as probe


@pytest.fixture
def calibration(tmp_path):
    directory = tmp_path / ".local" / "calibration"
    directory.mkdir(parents=True)
    profile = VisionProfile(
        image_size=(80, 60), base_dir=directory, calibrated=True, pointer_rest=(75, 55),
        phase_templates=(PhaseTemplate(Phase.SHOP, Rect(0, 0, 4, 3), "phase.png"),),
        hud={"gold": Rect(0, 5, 4, 3)}, buttons={"roll": (72, 42)},
        shop=(SlotConfig(Rect(8, 25, 5, 4), empty_template="pet-empty.png",
                         species_templates={"tiger": ["tiger-a.png", "tiger-b.png"]}),),
        team=(SlotConfig(Rect(20, 25, 5, 4), empty_template="pet-empty.png",
                         species_templates={"rhino": "rhino.png"}),),
        food=(FoodSlotConfig(Rect(40, 25, 5, 4), cost=Rect(40, 32, 4, 3),
                             empty_template="food-empty.png",
                             food_templates={"apple": ["apple-a.png", "apple-b.png"]}),),
        numeric_templates=(NumberTemplate("gold", 10, "gold.png"),
                           NumberTemplate("food.0.cost", 3, "food-cost.png")),
        numeric_colors=(NumericColor("gold", (200, 0, 0), (255, 80, 80)),),
    ).validate(require_calibrated=True)
    references = {
        "phase.png": (3, 4, 3), "pet-empty.png": (4, 5, 3),
        "tiger-a.png": (4, 5, 3), "tiger-b.png": (4, 5, 3), "rhino.png": (4, 5, 3),
        "food-empty.png": (4, 5, 3), "apple-a.png": (4, 5, 3), "apple-b.png": (4, 5, 3),
        "gold.png": (3, 4, 3), "food-cost.png": (3, 4, 3),
    }
    arrays = {name: np.full(shape, index*13, dtype=np.uint8)
              for index, (name, shape) in enumerate(references.items(), 1)}
    for name, array in arrays.items():
        Image.fromarray(array).save(directory / name)
    path = directory / "profile.json"
    profile.save(path)
    return SimpleNamespace(profile=profile, path=path, arrays=arrays, directory=directory)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def test_profile_snapshot_roundtrips_numeric_colors_and_detaches_both_nested_mappings(calibration):
    original = calibration.profile
    detached, record = probe.freeze_profile_snapshot(original)
    expected = json.loads(json.dumps(original.to_dict()))
    assert record["data"] == detached.to_dict() == expected
    assert record["canonical_sha256"] == canonical_hash(expected)
    assert detached.numeric_colors == original.numeric_colors
    assert detached.base_dir == original.base_dir

    original.shop[0].species_templates["tiger"].append("later.png")
    original.buttons["roll"] = (0, 0)
    assert detached.shop[0].species_templates["tiger"] == ["tiger-a.png", "tiger-b.png"]
    assert detached.buttons["roll"] == (72, 42)
    detached.food[0].food_templates["apple"].append("also-later.png")
    assert record["data"] == expected and canonical_hash(record["data"]) == record["canonical_sha256"]


def test_all_reference_families_variants_and_shared_uses_bind_loaded_rgb_not_changed_files(calibration):
    reader = Perceptor(calibration.profile)
    validate_profile_assets(reader)
    reader._templates = {name: array.copy() for name, array in reader._templates.items()}
    loaded = dict(reader._templates)
    for name in calibration.arrays:
        (calibration.directory / name).unlink()
    records = probe.freeze_loaded_reference_pixels(calibration.profile, reader)
    by_name = {record["name"]: record for record in records}
    assert set(by_name) == set(calibration.arrays)
    assert {use["field"] for record in records for use in record["uses"]} == {
        "phase.0", "shop.0.empty", "shop.0.species.tiger", "team.0.empty", "team.0.species.rhino",
        "food.0.empty", "food.0.food_id.apple", "numeric.gold.10", "numeric.food.0.cost.3",
    }
    assert len(by_name["pet-empty.png"]["uses"]) == 2
    for name, original in loaded.items():
        record = by_name[name]
        assert record["shape"] == list(original.shape) and record["dtype"] == "|u1"
        assert np.array_equal(reader._templates[name], calibration.arrays[name])
        assert reader._templates[name] is not original and not reader._templates[name].flags.writeable
        digest = record["pixel_sha256"]
        assert len(digest) == 64
        # Mutating a previously held array cannot change the bound reader pixels.
        original[:] = 255
        assert np.array_equal(reader._templates[name], calibration.arrays[name])
    assert records == probe.freeze_loaded_reference_pixels(calibration.profile, reader)


def test_changed_cached_pixel_changes_reference_digest_without_changing_profile_digest(calibration):
    first = SimpleNamespace(_templates={name: a.copy() for name, a in calibration.arrays.items()})
    second = SimpleNamespace(_templates={name: a.copy() for name, a in calibration.arrays.items()})
    second._templates["apple-b.png"][0, 0, 2] += 1
    _, snapshot_a = probe.freeze_profile_snapshot(calibration.profile)
    first_records = probe.freeze_loaded_reference_pixels(calibration.profile, first)
    _, snapshot_b = probe.freeze_profile_snapshot(calibration.profile)
    second_records = probe.freeze_loaded_reference_pixels(calibration.profile, second)
    assert snapshot_a == snapshot_b
    first_hashes = {r["name"]: r["pixel_sha256"] for r in first_records}
    second_hashes = {r["name"]: r["pixel_sha256"] for r in second_records}
    assert {name for name in first_hashes if first_hashes[name] != second_hashes[name]} == {"apple-b.png"}


@pytest.mark.parametrize("failure", ["no_cache", "missing", "grayscale", "float", "wrong_size"])
def test_missing_or_incoherent_loaded_assets_have_clear_errors(calibration, failure):
    cache = {name: a.copy() for name, a in calibration.arrays.items()}
    if failure == "missing":
        del cache["apple-b.png"]
    elif failure == "grayscale":
        cache["apple-b.png"] = np.zeros((4, 5), dtype=np.uint8)
    elif failure == "float":
        cache["apple-b.png"] = np.zeros((4, 5, 3), dtype=float)
    elif failure == "wrong_size":
        cache["apple-b.png"] = np.zeros((5, 4, 3), dtype=np.uint8)
    reader = SimpleNamespace() if failure == "no_cache" else SimpleNamespace(_templates=cache)
    with pytest.raises(ValueError, match="loaded calibration reference.*(missing|uint8 RGB|incoherent)"):
        probe.freeze_loaded_reference_pixels(calibration.profile, reader)


def test_zero_reference_profile_does_not_access_a_cache_api():
    class NoCache:
        @property
        def _templates(self):
            raise AssertionError("zero references must not need a cache")
    assert probe.freeze_loaded_reference_pixels(VisionProfile((80, 60)), NoCache()) == []


def test_reference_reused_at_incompatible_geometry_is_rejected(calibration):
    profile = replace(calibration.profile, phase_templates=(
        PhaseTemplate(Phase.SHOP, Rect(0, 0, 5, 4), "pet-empty.png"),
        PhaseTemplate(Phase.BATTLE, Rect(0, 0, 4, 3), "pet-empty.png"),
    ))
    reader = SimpleNamespace(_templates={name: a.copy() for name, a in calibration.arrays.items()})
    with pytest.raises(ValueError, match="dimensions are incoherent.*pet-empty"):
        probe.freeze_loaded_reference_pixels(profile, reader)


def test_run_freezes_before_native_io_and_binds_report_to_exact_manifest_bytes(calibration, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "PRIVATE_ROOT", tmp_path / ".local")
    monkeypatch.setattr(probe.VisionProfile, "load", Mock(return_value=calibration.profile))
    monkeypatch.setattr(probe, "desktop_control_lease", Mock(side_effect=nullcontext))
    monkeypatch.setattr(probe, "validate_live_dependencies", Mock())
    window = Mock(stop_requested=Mock(return_value=False))
    output = tmp_path / ".local" / "probe"
    layout = TooltipLayout((0, 0, 80, 60), (28, 32), (38, 42), (3, 3, 20, 10))
    original_data = json.loads(json.dumps(calibration.profile.to_dict()))
    readers = []
    def reader_factory(profile, **kwargs):
        reader = Perceptor(profile, **kwargs)
        readers.append(reader)
        return reader
    def assets(reader):
        validate_profile_assets(reader)
        # Neither the source JSON nor reference files are capture-time inputs
        # after the corresponding objects/pixels have been loaded.
        calibration.path.write_text("{}")
        for name in calibration.arrays:
            (calibration.directory / name).write_bytes(b"changed after loading")
        calibration.profile.shop[0].species_templates["tiger"].append("later.png")
        calibration.profile.buttons["roll"] = (0, 0)
    def construct(title):
        manifest = json.loads((output / "manifest.json").read_bytes())
        assert manifest["profile"] == str(calibration.path)
        assert manifest["profile_snapshot"]["data"] == original_data
        assert manifest["profile_snapshot"]["canonical_sha256"] == canonical_hash(original_data)
        assert readers[0].profile.to_dict() == original_data
        for record in manifest["loaded_reference_pixels"]:
            assert np.array_equal(readers[0]._templates[record["name"]], calibration.arrays[record["name"]])
        return window
    def collect(profile, target, layout, **kwargs):
        assert profile.to_dict() == original_data
        kwargs["evidence"]["pointer_restored"] = True
    monkeypatch.setattr(probe, "Perceptor", reader_factory)
    monkeypatch.setattr(probe, "validate_profile_assets", assets)
    monkeypatch.setattr(probe, "WindowsGameWindow", Mock(side_effect=construct))
    monkeypatch.setattr(probe, "collect_association", Mock(side_effect=collect))
    args = probe.parse_args(["--profile", str(calibration.path), "--target", "shop:0",
                            "--tooltip-layout", "unused-explicit-layout", "--output-dir", str(output)])
    report = probe.run_probe(args, clock=lambda: 0., sleep=lambda duration: None, tooltip_layout=layout)
    assert report["reason"] == "completed" and report["error"] is None
    assert report["manifest_sha256"] == hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    assert json.loads((output / "report.json").read_bytes()) == report
    assert report["captures"] == report["pointer_moves"] == 0
    window.capture.assert_not_called()
    window.hover.assert_not_called()


def test_provenance_failure_prevents_dependencies_lease_and_window(calibration, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "PRIVATE_ROOT", tmp_path / ".local")
    monkeypatch.setattr(probe.VisionProfile, "load", Mock(return_value=calibration.profile))
    monkeypatch.setattr(probe, "Perceptor", Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(probe, "validate_profile_assets", Mock())
    native = [Mock(), Mock(), Mock()]
    for name, mock in zip(("validate_live_dependencies", "desktop_control_lease", "WindowsGameWindow"), native):
        monkeypatch.setattr(probe, name, mock)
    output = tmp_path / ".local" / "probe"
    args = probe.parse_args(["--profile", str(calibration.path), "--target", "shop:0",
                            "--tooltip-layout", "unused", "--output-dir", str(output)])
    layout = TooltipLayout((0, 0, 80, 60), (28, 32), (38, 42), (3, 3, 20, 10))
    with pytest.raises(ValueError, match="reference cache is missing"):
        probe.run_probe(args, tooltip_layout=layout)
    assert not output.exists()
    for mock in native:
        mock.assert_not_called()
