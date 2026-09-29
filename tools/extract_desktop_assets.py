"""Extract named local Unity pet textures for private desktop-vision experiments.

This deliberately supports only the observed Unity 6000.3 serialized-file layout:
version 22, little-endian metadata, stripped type trees, external DXT1/DXT5 textures.
It never writes into the game installation. Extracted artwork and manifests belong
under .local, not in Git. Texture names are labels, not verified gameplay metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import struct


DEFAULT_ASSET = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Super Auto Pets\Super Auto Pets_Data\sharedassets1.assets")
DEFAULT_NAMES = ("Ant", "Beaver", "Cricket", "Duck", "Fish", "Mosquito", "Otter", "Pig", "Pigeon")
SUPPORTED_UNITY_PREFIX = "6000.3."
MAX_TEXTURES = 512
MAX_DECODED_PIXELS = 64_000_000


class AssetFormatError(ValueError):
    """The source does not match the small, explicitly supported asset format."""


@dataclass(frozen=True)
class AssetObject:
    path_id: int
    class_id: int
    offset: int
    size: int


@dataclass(frozen=True)
class Texture:
    path_id: int
    name: str
    width: int
    height: int
    format: int
    stream_offset: int
    stream_size: int
    stream_path: str


class Reader:
    def __init__(self, data: bytes, start: int = 0, end: int | None = None):
        self.data, self.pos = data, start
        self.end = len(data) if end is None else end
        if not 0 <= start <= self.end <= len(data):
            raise AssetFormatError("invalid reader bounds")

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > self.end:
            raise AssetFormatError("truncated serialized data")
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def unpack(self, fmt: str):
        result = struct.unpack(fmt, self.take(struct.calcsize(fmt)))
        return result[0] if len(result) == 1 else result

    def align(self):
        self.take((-self.pos) % 4)

    def string(self, *, maximum: int = 4096) -> str:
        size = self.unpack("<i")
        if not 0 <= size <= maximum:
            raise AssetFormatError("invalid serialized string length")
        try:
            value = self.take(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AssetFormatError("invalid UTF-8 string") from exc
        self.align()
        return value


def read_objects(data: bytes) -> tuple[str, tuple[AssetObject, ...]]:
    header = Reader(data)
    _, _, version, _ = header.unpack(">IIII")
    endian = header.unpack("B")
    header.take(3)
    if version != 22 or endian != 0:
        raise AssetFormatError("requires version-22 little-endian serialized file")
    metadata_size, file_size, data_offset, _ = header.unpack(">IQQQ")
    if file_size != len(data) or not 48 <= 48 + metadata_size <= data_offset <= len(data):
        raise AssetFormatError("invalid serialized file boundaries")
    reader = Reader(data, 48, 48 + metadata_size)
    version_bytes = bytearray()
    for _ in range(40):
        char = reader.take(1)
        if char == b"\0":
            break
        version_bytes.extend(char)
    else:
        raise AssetFormatError("unterminated Unity version")
    try:
        unity_version = version_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise AssetFormatError("invalid Unity version") from exc
    if not unity_version.startswith(SUPPORTED_UNITY_PREFIX):
        raise AssetFormatError("unsupported Unity texture layout")
    platform, type_tree, count = reader.unpack("<iBi")
    if platform != 19 or type_tree != 0 or not 1 <= count <= 10000:
        raise AssetFormatError("requires Windows metadata with stripped type trees")
    types = []
    for _ in range(count):
        class_id, _, _ = reader.unpack("<iBh")
        if class_id == 114:
            reader.take(16)  # MonoBehaviour script GUID.
        reader.take(16)  # Old type hash; no dependency list without type trees.
        types.append(class_id)
    object_count = reader.unpack("<i")
    if not 0 <= object_count <= 1_000_000:
        raise AssetFormatError("invalid object count")
    objects, ids, spans = [], set(), []
    for _ in range(object_count):
        reader.align()
        path_id, relative_offset, size, type_id = reader.unpack("<qQIi")
        offset = data_offset + relative_offset
        if not 0 <= type_id < len(types) or size == 0 or offset + size > len(data):
            raise AssetFormatError("invalid object extent or type")
        if path_id in ids:
            raise AssetFormatError("duplicate object path ID")
        ids.add(path_id)
        spans.append((offset, offset + size))
        objects.append(AssetObject(path_id, types[type_id], offset, size))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise AssetFormatError("overlapping object data")
    return unity_version, tuple(objects)


def object_name(data: bytes, obj: AssetObject) -> str:
    if obj.class_id not in (28, 213):
        raise AssetFormatError("object has no supported name layout")
    return Reader(data, obj.offset, obj.offset + obj.size).string()


def read_texture(data: bytes, obj: AssetObject) -> Texture:
    if obj.class_id != 28:
        raise AssetFormatError("object is not a Texture2D")
    reader = Reader(data, obj.offset, obj.offset + obj.size)
    name = reader.string()
    fields_start = reader.pos
    reader.take(4)  # Fallback/downscale flags in the observed Unity 6000.3 layout.
    width, height, complete_size, stripped_mips, fmt, mip_count = reader.unpack("<iiiiii")
    if not 1 <= width <= 8192 or not 1 <= height <= 8192 or fmt not in (10, 12):
        raise AssetFormatError("unsupported texture dimensions or compression")
    expected_size = ((width + 3) // 4) * ((height + 3) // 4) * (8 if fmt == 10 else 16)
    if stripped_mips != 0 or mip_count != 1 or complete_size != expected_size:
        raise AssetFormatError("requires a single complete DXT mip")
    reader.take(fields_start + 88 - reader.pos)
    if reader.unpack("<i") != 0:
        raise AssetFormatError("inline texture bytes are unsupported")
    offset, size = reader.unpack("<QI")
    stream_path = reader.string()
    if reader.pos != reader.end or size != expected_size:
        raise AssetFormatError("unexpected texture stream layout or size")
    if not re.fullmatch(r"[A-Za-z0-9_. -]+\.resS", stream_path) or stream_path in (".", ".."):
        raise AssetFormatError("texture stream must be a sibling .resS file")
    return Texture(obj.path_id, name, width, height, fmt, offset, size, stream_path)


def decode_texture(texture: Texture, payload: bytes):
    """Return upright RGBA using Pillow's existing BCn decoder."""
    from PIL import Image

    if len(payload) != texture.stream_size:
        raise AssetFormatError("truncated texture stream")
    fourcc = int.from_bytes(b"DXT1" if texture.format == 10 else b"DXT5", "little")
    header = [124, 0x81007, texture.height, texture.width, len(payload), 0, 1]
    header += [0] * 11
    header += [32, 4, fourcc, 0, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0]
    with Image.open(io.BytesIO(b"DDS " + struct.pack("<31I", *header) + payload)) as image:
        return image.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(asset_path: Path, output: Path, names=DEFAULT_NAMES) -> dict:
    requested_output = Path(output)
    if requested_output.exists() or requested_output.is_symlink():
        raise ValueError("output already exists; use a fresh output path")
    asset_path, output = Path(asset_path).resolve(), requested_output.resolve()
    installation = (asset_path.parent.parent if asset_path.parent.name.casefold().endswith("_data")
                    else asset_path.parent)
    if output == installation or installation in output.parents:
        raise ValueError("output must be outside the game installation")
    selected = set(names)
    if not selected or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]*", name) for name in selected):
        raise ValueError("supply at least one safe exact texture name")
    desired = selected | {name + "_2x" for name in selected}
    data = asset_path.read_bytes()
    unity_version, objects = read_objects(data)
    textures, decoded_pixels = [], 0
    for obj in objects:
        if obj.class_id == 28 and object_name(data, obj) in desired:
            if len(textures) >= MAX_TEXTURES:
                raise AssetFormatError(f"selected texture count exceeds {MAX_TEXTURES}")
            texture = read_texture(data, obj)
            decoded_pixels += texture.width * texture.height
            if decoded_pixels > MAX_DECODED_PIXELS:
                raise AssetFormatError(f"selected decoded pixels exceed {MAX_DECODED_PIXELS}")
            textures.append(texture)
    missing = sorted(name for name in selected if not any(t.name in (name, name + "_2x") for t in textures))
    if missing:
        raise ValueError("texture names not found: " + ", ".join(missing))
    # Resolve and validate every stream before writing any output.
    streams, stream_stats = {}, {}
    for texture in textures:
        path = (asset_path.parent / texture.stream_path).resolve()
        if path.parent != asset_path.parent:
            raise AssetFormatError("stream resolves outside the asset directory")
        if texture.stream_path not in streams:
            streams[texture.stream_path] = path
            stat = path.stat()
            stream_stats[texture.stream_path] = (stat.st_size, stat.st_mtime_ns)
        if texture.stream_offset + texture.stream_size > stream_stats[texture.stream_path][0]:
            raise AssetFormatError("texture extent exceeds its resource stream")
    stream_hashes = {name: sha256_file(path) for name, path in streams.items()}
    source_hash = hashlib.sha256(data).hexdigest()
    entries, images = [], []
    for texture in textures:
        with streams[texture.stream_path].open("rb") as handle:
            handle.seek(texture.stream_offset)
            payload = handle.read(texture.stream_size)
        image = decode_texture(texture, payload)
        filename = f"{texture.name}__{texture.path_id}.png"
        images.append((filename, image))
        entries.append({"path_id": texture.path_id, "exact_name": texture.name,
                        "candidate_species": texture.name.removesuffix("_2x").lower(),
                        "width": texture.width, "height": texture.height, "texture_format": texture.format,
                        "stream_path": texture.stream_path, "stream_offset": texture.stream_offset,
                        "stream_size": texture.stream_size, "compressed_sha256": hashlib.sha256(payload).hexdigest(),
                        "rgba_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "file": filename})
    for name, path in streams.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != stream_stats[name]:
            raise AssetFormatError("resource stream changed during extraction")
    output.mkdir(parents=True, exist_ok=False)
    for filename, image in images:
        image.save(output / filename)
    manifest = {"version": 1, "unity_version": unity_version, "source_path": str(asset_path),
                "source_sha256": source_hash, "stream_sha256": stream_hashes,
                "orientation": "Unity texture decoded as DDS, then vertically flipped",
                "label_note": "Exact texture names only. Candidate species and active art require visual verification.",
                "textures": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=Path(".local/desktop/assets"))
    parser.add_argument("--name", action="append", help="exact name; _2x variants are also included")
    args = parser.parse_args(argv)
    try:
        manifest = extract(args.asset, args.output, args.name or DEFAULT_NAMES)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"textures": len(manifest["textures"]), "output": str(args.output.resolve()),
                      "source_sha256": manifest["source_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
