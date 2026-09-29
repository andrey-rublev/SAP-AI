"""Synthetic Unity serialization fixtures; never require proprietary game files."""
import json
import struct

import pytest

from tools.extract_desktop_assets import (AssetFormatError, AssetObject, Texture, decode_texture,
                                         extract, object_name, read_objects, read_texture)


def string(value):
    data = struct.pack("<i", len(value.encode())) + value.encode()
    return data + b"\0" * (-len(data) % 4)


def texture(name="Ant", *, path="test.assets.resS", stream_offset=0, fmt=12, width=4, height=4):
    size = ((width + 3) // 4) * ((height + 3) // 4) * (8 if fmt == 10 else 16)
    fields = struct.pack("<7i", 0, width, height, size, 0, fmt, 1)
    fields += b"\0" * (88 - len(fields))
    return string(name) + fields + struct.pack("<iQI", 0, stream_offset, size) + string(path)


def serialized(items, *, unity="6000.3.11f1", endian=0, tree=0, version=22, duplicate_ids=False):
    meta = unity.encode() + b"\0" + struct.pack("<iBi", 19, tree, 1)
    meta += struct.pack("<iBh", 28, 0, -1) + b"\0" * 16 + struct.pack("<i", len(items))
    body = b""
    for index, item in enumerate(items):
        meta += b"\0" * (-len(meta) % 4)
        meta += struct.pack("<qQIi", 1 if duplicate_ids else index + 1, len(body), len(item), 0)
        body += item
    offset = 48 + len(meta)
    header = struct.pack(">IIII", 0, 0, version, 0) + bytes([endian, 0, 0, 0])
    header += struct.pack(">IQQQ", len(meta), offset + len(body), offset, 0)
    return header + meta + body


def parsed_texture(payload=None):
    data = serialized([texture() if payload is None else payload])
    _, objects = read_objects(data)
    return read_texture(data, objects[0])


def test_reads_named_objects_without_collapsing_duplicate_names():
    data = serialized([texture(), texture()])
    unity, objects = read_objects(data)
    assert unity == "6000.3.11f1"
    assert [obj.path_id for obj in objects] == [1, 2]
    assert [object_name(data, obj) for obj in objects] == ["Ant", "Ant"]
    assert parsed_texture().stream_size == 16


@pytest.mark.parametrize("kwargs", [{"version": 21}, {"endian": 1}, {"tree": 1},
                                    {"unity": "2022.3.1f1"}, {"duplicate_ids": True}])
def test_unsupported_serialization_rejected(kwargs):
    with pytest.raises(AssetFormatError):
        read_objects(serialized([texture(), texture()], **kwargs))


@pytest.mark.parametrize("cut", [0, 16, 47, 80, -1])
def test_truncated_serialization_rejected(cut):
    data = serialized([texture()])
    with pytest.raises(AssetFormatError):
        read_objects(data[:cut])


def test_invalid_object_type_and_extent_rejected():
    data = bytearray(serialized([texture()]))
    object_end = struct.unpack_from(">Q", data, 32)[0]
    struct.pack_into("<i", data, object_end - 4, 99)
    with pytest.raises(AssetFormatError, match="extent or type"):
        read_objects(bytes(data))
    data = bytearray(serialized([texture()]))
    struct.pack_into("<I", data, object_end - 8, 10_000)
    with pytest.raises(AssetFormatError, match="extent or type"):
        read_objects(bytes(data))


def test_overlapping_objects_rejected():
    data = bytearray(serialized([texture(), texture()]))
    object_end = struct.unpack_from(">Q", data, 32)[0]
    struct.pack_into("<Q", data, object_end - 16, 0)
    with pytest.raises(AssetFormatError, match="overlapping"):
        read_objects(bytes(data))


@pytest.mark.parametrize("path", ["../test.resS", "..\\test.resS", "C:\\test.resS", "/test.resS", "https://example/resS"])
def test_resource_path_traversal_rejected(path):
    with pytest.raises(AssetFormatError, match="sibling"):
        parsed_texture(texture(path=path))


@pytest.mark.parametrize("kwargs", [{"fmt": 4}, {"width": 0}, {"height": 9000}])
def test_unsupported_texture_dimensions_and_format_rejected(kwargs):
    with pytest.raises(AssetFormatError):
        parsed_texture(texture(**kwargs))


def test_texture_layout_and_mip_size_checked():
    payload = bytearray(texture())
    struct.pack_into("<i", payload, len(string("Ant")) + 12, 32)
    with pytest.raises(AssetFormatError, match="single complete"):
        parsed_texture(bytes(payload))
    with pytest.raises(AssetFormatError, match="layout or size"):
        parsed_texture(texture() + b"\0\0\0\0")


def test_texture_object_class_checked():
    with pytest.raises(AssetFormatError, match="not a Texture2D"):
        read_texture(b"", AssetObject(1, 213, 0, 0))


def test_dxt5_decodes_alpha_and_vertical_orientation():
    pytest.importorskip("PIL")
    # First block is opaque red, next block is opaque green. Unity stores bottom first.
    red = bytes([255, 0]) + b"\0" * 6 + struct.pack("<HHI", 0xF800, 0, 0)
    green = bytes([255, 0]) + b"\0" * 6 + struct.pack("<HHI", 0x07E0, 0, 0)
    item = Texture(1, "Ant", 4, 8, 12, 0, 32, "test.assets.resS")
    image = decode_texture(item, red + green)
    assert image.getpixel((0, 0)) == (0, 255, 0, 255)
    assert image.getpixel((0, 7)) == (255, 0, 0, 255)
    with pytest.raises(AssetFormatError, match="truncated"):
        decode_texture(item, red)


def test_extract_preserves_duplicate_ids_and_source_hashes(tmp_path):
    pytest.importorskip("PIL")
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "test.assets"
    asset.write_bytes(serialized([texture(), texture("Ant_2x"), texture()]))
    (source / "test.assets.resS").write_bytes(b"\0" * 16)
    output = tmp_path / "output"
    manifest = extract(asset, output, ["Ant"])
    assert len(manifest["textures"]) == 3
    assert len(list(output.glob("*.png"))) == 3
    assert [entry["path_id"] for entry in manifest["textures"]] == [1, 2, 3]
    assert all(len(entry["rgba_sha256"]) == 64 for entry in manifest["textures"])
    assert len(manifest["source_sha256"]) == 64
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_invalid_stream_or_missing_name_writes_nothing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "test.assets"
    asset.write_bytes(serialized([texture(stream_offset=8)]))
    (source / "test.assets.resS").write_bytes(b"\0" * 16)
    output = tmp_path / "output"
    with pytest.raises(AssetFormatError, match="exceeds"):
        extract(asset, output, ["Ant"])
    assert not output.exists()
    with pytest.raises(ValueError, match="not found"):
        extract(asset, output, ["Missing"])
    assert not output.exists()
    with pytest.raises(ValueError, match="outside"):
        extract(asset, source / "output", ["Ant"])


def test_output_cannot_be_created_in_game_installation_root(tmp_path):
    installation = tmp_path / "Super Auto Pets"
    source = installation / "Super Auto Pets_Data"
    source.mkdir(parents=True)
    asset = source / "test.assets"
    with pytest.raises(ValueError, match="outside the game installation"):
        extract(asset, installation / "extracted", ["Ant"])
    assert not (installation / "extracted").exists()


def test_existing_output_is_rejected_without_overwriting_anything(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text("preserve this output")
    with pytest.raises(ValueError, match="fresh output path"):
        extract(tmp_path / "missing.assets", output, ["Ant"])
    assert manifest.read_text() == "preserve this output"


def test_dangling_output_symlink_is_rejected_before_resolution(tmp_path, monkeypatch):
    # Exercise the guard without requiring symlink privileges on Windows CI.
    output = tmp_path / "link"
    monkeypatch.setattr(type(output), "is_symlink", lambda path: path == output)
    with pytest.raises(ValueError, match="fresh output path"):
        extract(tmp_path / "missing.assets", output, ["Ant"])
    assert not output.exists()


@pytest.mark.parametrize("items,error", [
    ([texture()] * 513, "texture count"),
    ([texture(width=8192, height=8192)], "decoded pixels"),
])
def test_aggregate_limits_reject_before_stream_reads_or_decode(tmp_path, monkeypatch, items, error):
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "test.assets"
    asset.write_bytes(serialized(items))
    # No resource file exists: validation must reject before opening it.
    def forbidden(*args):
        pytest.fail("oversized extraction reached the decoder")
    monkeypatch.setattr("tools.extract_desktop_assets.decode_texture", forbidden)
    with pytest.raises(AssetFormatError, match=error):
        extract(asset, tmp_path / "output", ["Ant"])
    assert not (tmp_path / "output").exists()


def test_resource_streams_are_not_loaded_whole_with_read_bytes(tmp_path, monkeypatch):
    from pathlib import Path
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "test.assets"
    asset.write_bytes(serialized([texture(stream_offset=32)]))
    stream = source / "test.assets.resS"
    payload = b"\0" * 64
    stream.write_bytes(payload)
    original = Path.read_bytes

    def guarded(path):
        if path == stream:
            pytest.fail("resource stream was read into a single byte buffer")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    manifest = extract(asset, tmp_path / "output", ["Ant"])
    import hashlib
    assert manifest["stream_sha256"][stream.name] == hashlib.sha256(payload).hexdigest()
    assert manifest["textures"][0]["stream_offset"] == 32


def test_changed_stream_is_rejected_before_writing_output(tmp_path, monkeypatch):
    from tools import extract_desktop_assets as module
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "test.assets"
    asset.write_bytes(serialized([texture()]))
    stream = source / "test.assets.resS"
    stream.write_bytes(b"\0" * 16)
    original = module.decode_texture

    def changed(texture, payload):
        image = original(texture, payload)
        stream.write_bytes(payload + b"extra")
        return image

    monkeypatch.setattr(module, "decode_texture", changed)
    with pytest.raises(AssetFormatError, match="changed during"):
        extract(asset, tmp_path / "output", ["Ant"])
    assert not (tmp_path / "output").exists()
