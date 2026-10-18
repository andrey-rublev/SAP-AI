"""Offline wrapper tests with generated development PNGs and caller Boards."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from PIL import Image
import pytest

from desktop_state import Board, PetSlot, Phase
from desktop_tooltip_layout import TooltipLayout
from desktop_tooltip_reveal import TooltipChangeLimits
from tools import check_desktop_tooltip_reveal as tool


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / ".local"
    root.mkdir()
    directory = root / "development"
    directory.mkdir()
    monkeypatch.setattr(tool.private_io, "ROOT", tmp_path)
    monkeypatch.setattr(tool.private_io, "PRIVATE_ROOT", root)
    clear = np.full((140, 200, 3), (90, 110, 130), dtype=np.uint8)
    clear[:3, 8:14] = 247
    hovered = clear.copy()
    hovered[39:91, 59:121] = 0
    hovered[40:90, 60:120] = 247
    hovered[48:54, 70:90] = (190, 100, 15)
    frames = []
    for index, (stage, image) in enumerate(zip(tool.STAGES, (clear, clear, hovered, hovered, clear, clear))):
        path = directory / f"frame-{index}.png"
        Image.fromarray(image).save(path)
        frames.append({"stage": stage, "image": path.name, "sha256": sha(path.read_bytes())})
    board = Board(Phase.SHOP, gold=7, turn=2, shop=(PetSlot(True, attack=2, health=3),)).to_dict()
    data = {"version": 1, "data_usage": "development_diagnostic", "held_out_evaluation": False,
            "image_size": [200, 140], "target": {"row": "shop", "index": 0},
            "layout": asdict(TooltipLayout((0, 0, 200, 140), (60, 60), (50, 50), (8, 6, 40, 15),
                                            dark_perimeter_max_channel=60)),
            "change_limits": asdict(TooltipChangeLimits(30, .6, .6)), "frames": frames,
            "clear_boards": [{"frame_sha256": frames[index]["sha256"], "board": board,
                              "label_source": "independently specified synthetic Board; no perception"}
                             for index in tool.CLEAR_INDICES]}
    manifest = directory / "sequence.json"
    manifest.write_text(json.dumps(data))
    return directory, manifest, data, root / "result"


def save(study):
    _, manifest, data, _ = study
    manifest.write_text(json.dumps(data))


def bind_changed_frame(study, index):
    directory, _, data, _ = study
    digest = sha((directory / data["frames"][index]["image"]).read_bytes())
    data["frames"][index]["sha256"] = digest
    if index in tool.CLEAR_INDICES:
        data["clear_boards"][tool.CLEAR_INDICES.index(index)]["frame_sha256"] = digest
    save(study)


def forbid_frame_reads(monkeypatch):
    original = tool._read_bounded
    def guarded(path, maximum):
        if path.suffix.lower() == ".png":
            pytest.fail("frame bytes read before declarations were validated")
        return original(path, maximum)
    monkeypatch.setattr(tool, "_read_bounded", guarded)


def test_success_is_conditional_caller_board_evidence_with_success_only_detached_title(study):
    directory, manifest, data, output = study
    originals = {entry["image"]: (directory / entry["image"]).read_bytes() for entry in data["frames"]}
    path, report = tool.run(manifest, output)
    assert path == output / "report.json" and report["status"] == "revealed"
    assert report["global_clearance"] is False and report["identity_interpretation"] == "none"
    assert report["clear_board_interpretation"] == "caller_supplied_unverified_annotations"
    assert report["clear_boards"][0]["board"]["shop"][0]["species"] is None
    assert report["manifest_sha256"] == sha(manifest.read_bytes())
    assert report["change_limits"] == data["change_limits"] and report["frames"] == data["frames"]
    assert all(len(value) == 64 for value in report["code_sha256"].values())
    title_path = output / report["title"]["image"]
    assert report["title"]["sha256"] == sha(title_path.read_bytes())
    with Image.open(title_path) as image:
        assert image.size == (40, 15)
    assert json.loads(path.read_text())["status"] == "revealed"
    assert all((directory / name).read_bytes() == raw for name, raw in originals.items())


def test_observational_refusal_writes_report_without_title(study):
    directory, manifest, data, output = study
    (directory / "frame-0.png").write_bytes((directory / "frame-2.png").read_bytes())
    bind_changed_frame(study, 0)
    _, report = tool.run(manifest, output)
    assert report["status"] == "refused" and report["reason"] == "clear_panel_present"
    assert report["title"] is None and not (output / "title.png").exists()


@pytest.mark.parametrize("change", ["test_usage", "held_out", "numeric_false", "unsupported", "version_bool",
                                    "bad_stage", "extra_frame_field", "bad_hash", "duplicate_path", "path_alias",
                                    "parent_path", "absolute_path", "stream_path", "board_binding", "board_source", "board_schema",
                                    "extra_board_field", "few_boards", "extra_target", "invalid_target",
                                    "unknown_limit", "missing_limit", "unknown_layout", "search_outside", "pixel_budget"])
def test_invalid_or_test_metadata_never_reads_frame_bytes(study, monkeypatch, change):
    directory, manifest, data, output = study
    if change == "test_usage":
        data["data_usage"] = "final_test"
    elif change == "held_out":
        data["held_out_evaluation"] = True
    elif change == "numeric_false":
        data["held_out_evaluation"] = 0
    elif change == "unsupported":
        data["expected_species"] = "fish"
    elif change == "version_bool":
        data["version"] = True
    elif change == "bad_stage":
        data["frames"][2]["stage"] = "clear"
    elif change == "extra_frame_field":
        data["frames"][2]["test"] = False
    elif change == "bad_hash":
        data["frames"][2]["sha256"] = "not-a-hash"
    elif change in {"duplicate_path", "path_alias"}:
        data["frames"][1]["image"] = ("./" if change == "path_alias" else "") + data["frames"][0]["image"]
    elif change == "parent_path":
        data["frames"][2]["image"] = "../outside.png"
    elif change == "absolute_path":
        data["frames"][2]["image"] = str(directory / "frame-2.png")
    elif change == "stream_path":
        data["frames"][2]["image"] = "frame-2.png:other.png"
    elif change == "board_binding":
        data["clear_boards"][0]["frame_sha256"] = "0" * 64
    elif change == "board_source":
        data["clear_boards"][0]["label_source"] = " "
    elif change == "board_schema":
        data["clear_boards"][0]["board"]["inferred_species"] = "fish"
    elif change == "extra_board_field":
        data["clear_boards"][0]["trusted"] = True
    elif change == "few_boards":
        data["clear_boards"].pop()
    elif change == "extra_target":
        data["target"]["point"] = [1, 2]
    elif change == "invalid_target":
        data["target"]["index"] = True
    elif change == "unknown_limit":
        data["change_limits"]["relax"] = True
    elif change == "missing_limit":
        data["change_limits"].pop("min_channel_change")
    elif change == "unknown_layout":
        data["layout"]["ignore_boundary"] = True
    elif change == "search_outside":
        data["layout"]["search_rect"] = [0, 0, 201, 140]
    else:
        data["image_size"] = [4000, 4000]
    save(study)
    forbid_frame_reads(monkeypatch)
    with pytest.raises(ValueError):
        tool.run(manifest, output)
    assert not (output / "report.json").exists()


def test_duplicate_json_metadata_is_rejected_before_pixels(study, monkeypatch):
    _, manifest, _, output = study
    payload = manifest.read_text().replace('"held_out_evaluation": false', '"held_out_evaluation": true, "held_out_evaluation": false')
    manifest.write_text(payload)
    forbid_frame_reads(monkeypatch)
    with pytest.raises(ValueError, match="duplicate JSON"):
        tool.run(manifest, output)


@pytest.mark.parametrize("kind", ["existing", "outside"])
def test_output_is_checked_before_reading_even_the_manifest(study, monkeypatch, kind):
    directory, manifest, _, output = study
    if kind == "existing":
        output.mkdir()
    else:
        output = directory.parents[1] / "public-output"
    monkeypatch.setattr(tool, "_read_bounded", lambda *_: pytest.fail("input read before output validation"))
    with pytest.raises(ValueError):
        tool.run(manifest, output)


def test_all_hashes_validate_before_any_image_is_opened(study, monkeypatch):
    directory, manifest, _, output = study
    (directory / "frame-5.png").write_bytes(b"changed after annotation")
    monkeypatch.setattr(tool.Image, "open", lambda *_: pytest.fail("decoded before all hashes passed"))
    with pytest.raises(ValueError, match="frame hash"):
        tool.run(manifest, output)


@pytest.mark.parametrize("budget", ["manifest", "frame"])
def test_reads_obey_byte_budgets_before_decoding(study, monkeypatch, budget):
    _, manifest, _, output = study
    monkeypatch.setattr(tool, "MAX_MANIFEST_BYTES" if budget == "manifest" else "MAX_FRAME_BYTES", 8)
    monkeypatch.setattr(tool.Image, "open", lambda *_: pytest.fail("over-budget image decoded"))
    with pytest.raises(ValueError, match="byte budget"):
        tool.run(manifest, output)


@pytest.mark.parametrize("kind", ["wrong_dimensions", "rgba", "jpeg", "animated"])
def test_png_headers_must_match_single_rgb_declared_geometry(study, kind):
    directory, manifest, _, output = study
    path = directory / "frame-0.png"
    if kind == "wrong_dimensions":
        Image.new("RGB", (201, 140)).save(path)
    elif kind == "rgba":
        Image.new("RGBA", (200, 140)).save(path)
    elif kind == "jpeg":
        Image.new("RGB", (200, 140)).save(path, format="JPEG")
    else:
        Image.new("RGB", (200, 140)).save(path, save_all=True, append_images=[Image.new("RGB", (200, 140), "red")])
    bind_changed_frame(study, 0)
    with pytest.raises(ValueError, match="dimensions|RGB PNG"):
        tool.run(manifest, output)


def test_cli_reports_diagnostic_success_without_identity_claims(study, capsys):
    _, manifest, _, output = study
    assert tool.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "revealed" and summary["global_clearance"] is False
    assert summary["identity_interpretation"] == "none"
