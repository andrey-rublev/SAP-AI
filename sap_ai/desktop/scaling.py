"""Adapt a calibrated profile to another client size with the same aspect ratio.

Super Auto Pets lays out its UI and board relative to the client size, so at
the calibrated aspect ratio every region, point and reference image scales by
one factor. Resized reference images are written once to a cache directory
and the scaled profile is loaded through the normal, validated loader.
Different aspect ratios are rejected: there, UI anchors and the world camera
scale differently and a uniform map would be wrong.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from sap_ai.desktop.vision import VisionProfile, decode_profile_json

SCALING_CACHE_VERSION = 2
ASPECT_TOLERANCE = .005
# Phase references shrunk from native frames differ from frames the game renders
# natively at the smaller size (text and edges resample differently). Measured
# on live 1904x1071 frames: true matches rose to at most 0.073 (RGB) while the
# nearest wrong screen stayed at 0.106 or more. The allowance grows with the
# amount of shrinking and is zero at the calibrated size.
RESAMPLE_ALLOWANCE = {"rgb": .06, "white_text": .12}
ALLOWANCE_REFERENCE_SHRINK = .25


def _size(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(item) is not int or item <= 0 for item in value)):
        raise ValueError(f"{name} must contain two positive integer dimensions")
    return tuple(value)


def scale_factor(source_size, target_size) -> float:
    source_size, target_size = _size(source_size, "source size"), _size(target_size, "target size")
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    if abs(sx - sy) > ASPECT_TOLERANCE * max(sx, sy):
        raise ValueError(f"cannot scale a {source_size[0]}x{source_size[1]} profile to "
                         f"{target_size[0]}x{target_size[1]}: aspect ratios differ")
    return (sx + sy) / 2


def fit_size(source_size, bounds) -> tuple[int, int]:
    """Largest client size with the profile's aspect ratio inside ``bounds``."""
    source_size, bounds = _size(source_size, "source size"), _size(bounds, "bounds")
    factor = min(bounds[0] / source_size[0], bounds[1] / source_size[1], 1.0)
    width = int(source_size[0] * factor) // 2 * 2
    height = round(width * source_size[1] / source_size[0])
    if width < 1 or height < 1:
        raise ValueError("bounds are too small for a client with this aspect ratio")
    return width, height


def scale_profile(path, target_size, cache_dir) -> VisionProfile:
    """Return the profile at ``path`` scaled to ``target_size`` (same aspect)."""
    path = Path(path)
    payload = path.read_bytes()
    data = decode_profile_json(payload)
    # Validate the source before rewriting paths or dropping any fields. Scaling
    # must not turn a rejected profile into an accepted one.
    profile = VisionProfile.from_dict(data, base_dir=path.parent)
    source = profile.image_size
    target = _size(target_size, "target size")
    if source == target:
        return profile
    factor = scale_factor(source, target)
    names = {item.template for item in (*profile.phase_templates, *profile.numeric_templates)}
    for slot in (*profile.shop, *profile.team):
        if slot.empty_template is not None:
            names.add(slot.empty_template)
        names.update(name for _, name in slot.species_references())
    for slot in profile.food:
        if slot.empty_template is not None:
            names.add(slot.empty_template)
        names.update(name for _, name in slot.food_references())
    # An edited reference image is a new calibration even when its path and the
    # JSON are unchanged. Include every referenced file and the scaling version.
    fingerprint = hashlib.sha256(payload + json.dumps([target, SCALING_CACHE_VERSION]).encode())
    for name in sorted(names):
        fingerprint.update(json.dumps(name).encode())
        fingerprint.update(hashlib.sha256(profile.template_path(name).read_bytes()).digest())
    digest = fingerprint.hexdigest()[:16]
    out_dir = Path(cache_dir) / f"{target[0]}x{target[1]}-{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(data)

    def rect(value):
        if value is None:
            return None
        x, y, w, h = value
        left, top = round(x * factor), round(y * factor)
        width = max(1, min(round((x + w) * factor) - left, target[0] - left))
        height = max(1, min(round((y + h) * factor) - top, target[1] - top))
        return [left, top, width, height]

    def point(value):
        return None if value is None else [min(round(value[0] * factor), target[0] - 1),
                                            min(round(value[1] * factor), target[1] - 1)]

    def image(name, region):
        """Resize a reference to exactly its scaled region; returns the cached name."""
        from PIL import Image
        width, height = region[2], region[3]
        # Distinct paths can collapse to the same flattened filename (a/b and
        # a__b). A path digest keeps those references separate.
        label = hashlib.sha256(name.encode()).hexdigest()[:16]
        cached = Path("scaled") / f"{label}-{width}x{height}.png"
        destination = out_dir / cached
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(profile.template_path(name)) as source_image:
                source_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(destination)
        return cached.as_posix()

    def references(value, region):
        if isinstance(value, str):
            return image(value, region)
        return [image(item, region) for item in value]

    data["image_size"] = list(target)
    shrink = min(abs(1 - factor) / ALLOWANCE_REFERENCE_SHRINK, 1.)
    for item in data.get("phase_templates", []):
        mode = item.get("match_mode", "rgb")
        item["max_distance"] = round(item.get("max_distance", .06) + RESAMPLE_ALLOWANCE[mode] * shrink, 4)
        item["region"] = rect(item["region"])
        item["template"] = image(item["template"], item["region"])
        options = item.get("white_text")
        if options and "min_ink_pixels" in options:
            options["min_ink_pixels"] = max(1, round(options["min_ink_pixels"] * factor * factor))
    data["hud"] = {name: rect(value) for name, value in data.get("hud", {}).items()}
    for row in ("shop", "team"):
        for slot in data.get(row, []):
            for name in ("portrait", "attack", "health", "level"):
                slot[name] = rect(slot.get(name))
            if slot.get("empty_template"):
                slot["empty_template"] = image(slot["empty_template"], slot["portrait"])
            slot["species_templates"] = {species: references(value, slot["portrait"])
                                         for species, value in slot.get("species_templates", {}).items()}
    for slot in data.get("food", []):
        slot["portrait"] = rect(slot["portrait"])
        slot["cost"] = rect(slot.get("cost"))
        if slot.get("empty_template"):
            slot["empty_template"] = image(slot["empty_template"], slot["portrait"])
        slot["food_templates"] = {food: references(value, slot["portrait"])
                                  for food, value in slot.get("food_templates", {}).items()}
    data["buttons"] = {name: point(value) for name, value in data.get("buttons", {}).items()}
    if data.get("pointer_rest") is not None:
        data["pointer_rest"] = point(data["pointer_rest"])
    # Numeric references are crops of their field's region; resolve via the scaled profile.
    numeric = data.pop("numeric_templates", [])
    partial = VisionProfile.from_dict({**data, "numeric_templates": []}, base_dir=out_dir)
    for item in numeric:
        region, _, _ = partial.numeric_region(item["field"])
        item["template"] = image(item["template"], region.to_list())
        # Digit references keep their strict limits. References cover only a
        # few values, so a loosened limit lets a wrong reference (turn "3")
        # match a value it has no reference for (turn 5) and block the OCR
        # reading. Unmatched references fall back to OCR, which scales well.
    data["numeric_templates"] = numeric
    scaled_path = out_dir / "profile.json"
    scaled_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return VisionProfile.load(scaled_path)


def scale_tooltip_layout(data: dict, factor: float) -> dict:
    """Scale a tooltip layout file's geometry (``load_tooltip_layout`` format)."""
    layout = dict(data["layout"])
    for name in ("search_rect", "title_offset_rect"):
        layout[name] = [round(value * factor) for value in layout[name]]
    for name in ("panel_width_range", "panel_height_range"):
        low, high = layout[name]
        layout[name] = [max(1, int(low * factor)), max(1, round(high * factor + .5))]
    return {**data, "image_size": [round(value * factor) for value in data["image_size"]], "layout": layout}
