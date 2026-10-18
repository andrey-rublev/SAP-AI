"""Profile decoding and byte provenance using temporary synthetic files only."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from tools import eval_desktop_frames as frames
from tools import train_desktop_species as species


DUPLICATES = (
    '"available_from_turn":1,"available_from_turn":1',
    r'"level":null,"le\u0076el":null',
    '"max_distance":0.001,"max_distance":0.06',
)


def profile_bytes(extra, *, full=False):
    prefix = '"version":1,' if full else ''
    text = ('{' + prefix + '"image_size":[24,8],"shop":[{"portrait":[0,0,12,8],'
            '"empty_template":"empty0.png",' + extra + '}],'
            '"team":[{"portrait":[12,0,12,8],"empty_template":"empty1.png"}]}')
    return text.encode("utf-8-sig")


def assert_source_read_once(monkeypatch, source):
    original, reads = Path.read_bytes, []

    def read(path):
        if path == source:
            reads.append(path)
            assert len(reads) == 1, "source profile bytes were reread"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    return reads


@pytest.mark.parametrize("extra", DUPLICATES)
def test_frame_snapshot_rejects_ambiguous_json_before_templates_or_destination(tmp_path, monkeypatch, extra):
    source, target = tmp_path / "profile.json", tmp_path / "snapshot"
    source.write_bytes(profile_bytes(extra, full=True))
    conversion = Mock(side_effect=AssertionError("ambiguous source reached asset validation"))
    monkeypatch.setattr(frames.VisionProfile, "from_dict", conversion)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        frames.snapshot_profile(source, target)
    conversion.assert_not_called()
    assert not target.exists()


@pytest.mark.parametrize("extra", DUPLICATES)
def test_species_prepare_rejects_ambiguous_partial_profile_before_images_or_artifacts(tmp_path, monkeypatch, extra):
    private = tmp_path / ".local"
    private.mkdir()
    source, output = private / "profile.json", private / "prepared"
    source.write_bytes(profile_bytes(extra))
    image = Mock(side_effect=AssertionError("image opened before profile rejection"))
    artifact = Mock(side_effect=AssertionError("artifact written before profile rejection"))
    monkeypatch.setattr(species, "PRIVATE_ROOT", private)
    monkeypatch.setattr(species.Image, "open", image)
    monkeypatch.setattr(species, "write_output", artifact)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        species.prepare(source, private / "assets", private / "negatives", output)
    image.assert_not_called()
    artifact.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("extra", DUPLICATES)
def test_species_report_rejects_ambiguous_snapshot_before_images_or_model(tmp_path, monkeypatch, extra):
    source, labels = tmp_path / "profile.json", tmp_path / "labels.json"
    raw = profile_bytes(extra)
    source.write_bytes(raw)
    labels.write_text('{"label_source":"synthetic","frames":[]}', encoding="utf-8")
    plan = {"profile": str(source), "profile_sha256": hashlib.sha256(raw).hexdigest(), "labels": []}
    model = Mock(side_effect=AssertionError("model called before profile rejection"))
    image = Mock(side_effect=AssertionError("image opened before profile rejection"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(species.Image, "open", image)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        species.real_results(model, plan, labels)
    image.assert_not_called()
    model.assert_not_called()


def test_frame_snapshot_keeps_original_bom_unicode_bytes_and_hash_without_source_reread(tmp_path, monkeypatch):
    data = {"version": 1, "image_size": [24, 8],
            "shop": [{"portrait": [0, 0, 12, 8], "species_templates": {"猫": "é.png"}}]}
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8-sig")
    source, target = tmp_path / "profile.json", tmp_path / "snapshot"
    source.write_bytes(raw)
    Image.new("RGB", (12, 8), "blue").save(tmp_path / "é.png")
    reads = assert_source_read_once(monkeypatch, source)
    profile, provenance = frames.snapshot_profile(source, target)
    assert reads == [source]
    assert (target / "profile-snapshot.json").read_bytes() == raw
    assert provenance["sha256"] == hashlib.sha256(raw).hexdigest()
    assert profile.base_dir == target
    assert profile.shop[0].species_templates == {"猫": "é.png"}


def test_species_report_accepts_partial_bom_unicode_profile_and_hashes_original_bytes(tmp_path, monkeypatch):
    # No version/full vision schema is required by this older crop-only reader.
    data = {"image_size": [24, 8], "shop": [{"portrait": [0, 0, 12, 8],
             "species_templates": {"猫": "é.png"}}], "team": []}
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8-sig")
    source, labels = tmp_path / "profile.json", tmp_path / "labels.json"
    source.write_bytes(raw)
    labels.write_text('{"label_source":"synthetic","frames":[]}', encoding="utf-8")
    plan = {"profile": str(source), "profile_sha256": hashlib.sha256(raw).hexdigest(), "labels": []}
    model = Mock(side_effect=AssertionError("empty labels must not call a model"))
    image = Mock(side_effect=AssertionError("empty labels must not open an image"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(species.Image, "open", image)
    reads = assert_source_read_once(monkeypatch, source)
    report = species.real_results(model, plan, labels)
    assert reads == [source] and report["samples"] == 0 and report["records"] == []
    image.assert_not_called()
    model.assert_not_called()
