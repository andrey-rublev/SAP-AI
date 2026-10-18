"""Private, offline sprite-to-desktop species experiment; never controls the game.

Fit partial-crop geometry for the original sprite-composite experiment, or use
train-real for explicitly labeled real crops with disjoint sessions and poses.
Both report held-out results offline and never modify the desktop vision profile.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_vision import SlotConfig, decode_profile_json

UNKNOWN = "__unknown__"
EMPTY = "__empty__"
REAL_SPLITS = ("train", "calibration", "test")
# Fixed before examining results; backgrounds or almost unchanged portraits cannot
# establish transfer merely because they came from another full screenshot.
REAL_NEAR_DUPLICATE_MAE = .01
INPUT_SIZE = (64, 32)
PRIVATE_ROOT = ROOT / ".local"
# Capture the implementation loaded for this process, never a later file revision.
IMPLEMENTATION_SOURCE = Path(__file__).read_bytes()
IMPLEMENTATION_SHA256 = hashlib.sha256(IMPLEMENTATION_SOURCE).hexdigest()
PREPROCESSING = "RGB; BILINEAR resize to 64x32; CHW float32 / 255"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_path(path, *, writable=False):
    """Confine artifacts to repo/.local and refuse redirected filesystem paths."""
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(PRIVATE_ROOT))
    if path == root or root not in path.parents:
        raise ValueError("artifact must be under the repository's private .local directory")
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("linked or redirected artifact paths are not allowed")
        if item == path and writable and (not stat.S_ISREG(info.st_mode) or info.st_nlink > 1):
            raise ValueError("output must be an unlinked regular file")
    return path


def write_output(path, writer):
    path = private_path(path, writable=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        private_path(path, writable=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dump(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_output(path, lambda handle: handle.write(payload))


def preprocess(image):
    image = image.convert("RGB").resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255


def fit_sprite(crop, sprite, background, sizes=range(144, 289, 8)):
    """Find scale, mirror, and partial-crop offset by RGB reconstruction error."""
    import cv2

    crop = crop.convert("RGB")
    target = np.asarray(crop, dtype=np.float32)
    color = tuple(int(v) for v in np.median(np.asarray(background).reshape(-1, 3), axis=0))
    best = None
    padding = max(crop.size)
    for mirror in (False, True):
        original = ImageOps.mirror(sprite) if mirror else sprite
        for size in sizes:
            rendered = original.resize((size, size), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (size + 2 * padding, size + 2 * padding), color)
            canvas.paste(rendered, (padding, padding), rendered)
            distances = cv2.matchTemplate(np.asarray(canvas, dtype=np.float32), target, cv2.TM_SQDIFF)
            minimum, _, position, _ = cv2.minMaxLoc(distances)
            score = max(0., minimum) / (target.size * 255 ** 2)
            if best is None or score < best["mse"]:
                best = {"size": size, "x": position[0] - padding, "y": position[1] - padding,
                        "width": crop.width, "height": crop.height, "mirror": mirror, "mse": score}
    return best


def render(sprite, background, fit, rng=None):
    """Compose an actual partial sprite over an independently sampled background."""
    rng = rng or random.Random(0)
    jitter = fit.get("jitter", 0)
    scale = rng.uniform(1 - jitter, 1 + jitter)
    size = max(1, round(fit["size"] * scale))
    sprite = ImageOps.mirror(sprite) if fit["mirror"] else sprite
    sprite = sprite.resize((size, size), Image.Resampling.BICUBIC)
    width, height = fit["width"], fit["height"]
    canvas = background.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    x = round(fit["x"] * scale + rng.uniform(-4, 4) * bool(jitter))
    y = round(fit["y"] * scale + rng.uniform(-4, 4) * bool(jitter))
    canvas.paste(sprite, (-x, -y), sprite)
    if jitter:
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(.88, 1.12))
        canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(.85, 1.15))
    return canvas


def prepare(profile_path, assets_path, negatives_path, output):
    profile_path, assets_path, negatives_path, output = map(Path, (profile_path, assets_path, negatives_path, output))
    output = private_path(output)
    if output.exists():
        raise ValueError("prepare requires a new output directory")
    profile_bytes = profile_path.read_bytes()
    profile = decode_profile_json(profile_bytes)
    backgrounds = sorted({str((profile_path.parent / slot["empty_template"]).resolve())
                          for row in ("shop", "team") for slot in profile[row] if slot.get("empty_template")})
    # Equal-pixel aliases must not cross the background split.
    unique = {}
    for path in backgrounds:
        with Image.open(path) as image:
            pixels = image.convert("RGB")
            key = (pixels.size, hashlib.sha256(pixels.tobytes()).hexdigest())
        unique.setdefault(key, path)
    backgrounds = list(unique.values())
    if len(backgrounds) < 2:
        raise ValueError("need at least two distinct recorded empty backgrounds")
    references = {}
    for row in ("shop", "team"):
        for slot in profile[row]:
            # Geometry fitting retains the first reference for each species;
            # alternate portraits belong to calibrated template matching.
            for species, path in SlotConfig.from_dict(slot).species_references():
                references.setdefault(species, str((profile_path.parent / path).resolve()))
    entries, previews = [], []
    for negative, directory in ((False, assets_path), (True, negatives_path)):
        manifest = json.loads((directory / "manifest.json").read_text())
        for entry in manifest["textures"]:
            species = entry["candidate_species"]
            path = str((directory / entry["file"]).resolve())
            if not negative and species not in references:
                continue
            sprite = Image.open(path).convert("RGBA")
            if negative:
                best_known = {}
                for known in entries:
                    if known["label"] != UNKNOWN and (known["label"] not in best_known or
                            known["fit"]["mse"] < best_known[known["label"]]["mse"]):
                        best_known[known["label"]] = known["fit"]
                if not best_known:
                    raise ValueError("unknown sprites require fitted known-pet geometry")
                fit = {key: int(np.median([value[key] for value in best_known.values()]))
                       for key in ("size", "x", "y", "width", "height")}
                fit.update(mirror=True, mse=None)
            else:
                crop = Image.open(references[species]).convert("RGB")
                fit = fit_sprite(crop, sprite, Image.open(backgrounds[0]).convert("RGB"))
                previews.append((species, entry["path_id"], fit["mse"], crop,
                                 render(sprite, Image.open(backgrounds[0]), fit)))
            entries.append({**entry, "path": path, "label": UNKNOWN if negative else species,
                            "manifest_sha256": digest(directory / "manifest.json"), "fit": fit})
    # Variant IDs stay disjoint. Exact RGBA duplicates are grouped before splitting.
    grouped = {}
    for entry in entries:
        grouped.setdefault((entry["label"], entry["rgba_sha256"]), []).append(entry)
    by_label = {}
    for (label, _), group in grouped.items():
        by_label.setdefault(label, []).append(group)
    for groups in by_label.values():
        groups.sort(key=lambda group: (group[0]["fit"]["mse"] or 0, group[0]["path_id"]))
        if len(groups) < 2:
            raise ValueError("need two distinct sprite variants per class")
        for index, group in enumerate(groups):
            for entry in group:
                entry["split"] = "validation" if index % 2 else "train"
    plan = {"version": 1, "profile": str((output / "profile-snapshot.json").resolve()),
            "source_profile": str(profile_path.resolve()), "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "references": {label: {"path": path, "sha256": digest(path)} for label, path in references.items()},
            "backgrounds": [{"path": path, "sha256": digest(path), "split": "validation" if i % 2 else "train"}
                            for i, path in enumerate(backgrounds)],
            "labels": sorted({entry["label"] for entry in entries}), "entries": entries,
            "limitations": ["Synthetic validation is not real-world accuracy.",
                            "Variants share species artwork and are not independent species samples.",
                            "Unknown training species differ from real held-out unknown species."]}
    output.mkdir(parents=True)
    write_output(output / "profile-snapshot.json", lambda handle: handle.write(profile_bytes))
    dump(output / "plan.json", plan)
    sheet = Image.new("RGB", (650, len(previews) * 95), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for i, (label, path_id, mse, crop, reconstruction) in enumerate(previews):
        draw.text((5, i * 95 + 6), f"{label} ID{path_id} MSE={mse:.4f}", fill="black")
        sheet.paste(crop, (255, i * 95 + 5))
        sheet.paste(reconstruction, (425, i * 95 + 5))
    write_output(output / "geometry-fit.png", lambda handle: sheet.save(handle, format="PNG"))
    return plan


class SyntheticCrops:
    def __init__(self, plan, split):
        if split not in ("train", "validation"):
            raise ValueError("invalid dataset split")
        self.plan, self.split = plan, split
        self.labels = plan["labels"]
        self.entries = [entry for entry in plan["entries"] if entry["split"] == split]
        self.backgrounds = [Image.open(item["path"]).convert("RGB") for item in plan["backgrounds"] if item["split"] == split]
        self.sprites = {entry["path"]: Image.open(entry["path"]).convert("RGBA") for entry in self.entries}
        for entry in self.entries:
            if "rgba_sha256" in entry and hashlib.sha256(self.sprites[entry["path"]].tobytes()).hexdigest() != entry["rgba_sha256"]:
                raise ValueError("sprite pixels changed since fitting")
        for item in plan["backgrounds"]:
            if "sha256" in item and digest(item["path"]) != item["sha256"]:
                raise ValueError("background changed since fitting")
        if not self.backgrounds or {entry["label"] for entry in self.entries} != set(self.labels):
            raise ValueError("incomplete dataset split")

    def sample(self, index):
        seed = int.from_bytes(hashlib.sha256(f"species-v1|{self.split}|{index}".encode()).digest()[:8], "little")
        rng = random.Random(seed)
        label = self.labels[index % len(self.labels)]
        options = [entry for entry in self.entries if entry["label"] == label]
        entry, background = rng.choice(options), rng.choice(self.backgrounds)
        fit = {**entry["fit"], "jitter": .10}
        # Flipping the rendered crop is also safe for a partial portrait.
        image = render(self.sprites[entry["path"]], background, fit, rng)
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if label == UNKNOWN and rng.random() < .2:
            image = background
        return preprocess(image), self.labels.index(label)


def model_for(label_count):
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 4)),
                         nn.Flatten(), nn.Linear(48 * 8, label_count))


def real_results(model, plan, labels_path, threshold=.95):
    import torch
    labels_path = Path(labels_path)
    labels_bytes = labels_path.read_bytes()
    ground_truth = json.loads(labels_bytes)
    profile_bytes = Path(plan["profile"]).read_bytes()
    profile = decode_profile_json(profile_bytes)
    if hashlib.sha256(profile_bytes).hexdigest() != plan["profile_sha256"]:
        raise ValueError("profile changed since fitting")
    if not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between zero and one")
    records, excluded = [], []
    for frame in ground_truth["frames"]:
        if frame.get("phase") != "shop":
            excluded.append(frame["image"])
            continue
        source = (labels_path.parent / frame["image"]).resolve()
        frame_bytes = source.read_bytes()
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        if list(image.size) != profile["image_size"]:
            raise ValueError("real frame does not match profile size")
        for row in ("shop", "team"):
            for i, expected in enumerate(frame[row]):
                x, y, width, height = profile[row][i]["portrait"]
                crop = image.crop((x, y, x + width, y + height))
                array = preprocess(crop)
                with torch.no_grad():
                    probs = model(torch.from_numpy(array.copy()).unsqueeze(0)).softmax(dim=1)[0]
                probability, index = probs.max(dim=0)
                guess = plan["labels"][int(index)]
                prediction = guess if float(probability) >= threshold else UNKNOWN
                records.append({"image": frame["image"], "frame_sha256": hashlib.sha256(frame_bytes).hexdigest(), "slot": f"{row}.{i}",
                                "crop_sha256": hashlib.sha256(crop.tobytes()).hexdigest(), "expected": expected,
                                "prediction": prediction, "top_label": guess, "confidence": float(probability)})
    known = [r for r in records if r["expected"] != UNKNOWN]
    unknown = [r for r in records if r["expected"] == UNKNOWN]
    accepted = [r for r in records if r["prediction"] != UNKNOWN]
    return {"threshold": threshold, "preprocessing": PREPROCESSING, "records": records, "excluded_non_shop_frames": excluded,
            "samples": len(records), "unique_pixel_crops": len({r["crop_sha256"] for r in records}),
            "known_samples": len(known), "unknown_samples": len(unknown), "accepted": len(accepted),
            "correct_accepted": sum(r["prediction"] == r["expected"] for r in accepted),
            "known_correct": sum(r["prediction"] == r["expected"] for r in known),
            "unknown_false_accepts": sum(r["prediction"] != UNKNOWN for r in unknown),
            "label_source": ground_truth["label_source"], "labels_sha256": hashlib.sha256(labels_bytes).hexdigest()}


def train(plan_path, labels_path, seconds=3300, max_steps=200000, batch_size=64, threads=4):
    import torch
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
            or any(type(value) is not int or value <= 0 for value in (max_steps, batch_size, threads))):
        raise ValueError("training bounds must be finite and positive; counts must be integers")
    plan_path = private_path(plan_path)
    for filename in ("model.pt", "progress.json", "report.json"):
        private_path(plan_path.parent / filename, writable=True)
    torch.set_num_threads(threads)
    torch.manual_seed(48271)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    provenance = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                  "implementation_sha256": IMPLEMENTATION_SHA256, "preprocessing": PREPROCESSING}
    training, validation = SyntheticCrops(plan, "train"), SyntheticCrops(plan, "validation")
    model = model_for(len(plan["labels"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_function = torch.nn.CrossEntropyLoss()
    validation_samples = [validation.sample(i) for i in range(1200)]
    vx = torch.from_numpy(np.stack([x for x, _ in validation_samples]))
    vy = torch.tensor([y for _, y in validation_samples])
    start, last_checkpoint = time.monotonic(), time.monotonic()
    history, step, loss = [], 0, None
    try:
        while step < max_steps and time.monotonic() - start < seconds:
            model.train()
            samples = [training.sample(step * batch_size + i) for i in range(batch_size)]
            x = torch.from_numpy(np.stack([a for a, _ in samples]))
            y = torch.tensor([b for _, b in samples])
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1
            if time.monotonic() - last_checkpoint >= 60:
                model.eval()
                with torch.no_grad():
                    predictions = torch.cat([model(batch).argmax(1) for batch in vx.split(128)])
                history.append({"step": step, "seconds": time.monotonic() - start,
                                "loss": float(loss.detach()), "synthetic_accuracy": float((predictions == vy).float().mean())})
                write_output(plan_path.parent / "model.pt", lambda handle: torch.save(
                    {"state_dict": model.state_dict(), "labels": plan["labels"], **provenance}, handle))
                dump(plan_path.parent / "progress.json", history)
                print(json.dumps(history[-1]), flush=True)
                last_checkpoint = time.monotonic()
    except KeyboardInterrupt:
        pass
    model.eval()
    write_output(plan_path.parent / "model.pt", lambda handle: torch.save(
        {"state_dict": model.state_dict(), "labels": plan["labels"], **provenance}, handle))
    with torch.no_grad():
        predictions = torch.cat([model(batch).argmax(1) for batch in vx.split(128)])
    report = {"steps": step, "seconds": time.monotonic() - start, "history": history,
              "synthetic_validation_accuracy": float((predictions == vy).float().mean()),
              "real_holdout": real_results(model, plan, labels_path), **provenance, "device": "cpu", "seed": 48271,
              "limitations": plan["limitations"] + ["Small same-session real holdout; no production readiness claim."]}
    dump(plan_path.parent / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "real_holdout")}), flush=True)
    return report


def evaluate_checkpoint(plan_path, labels_path, model_path, output):
    """Re-evaluate fixed weights, recording training and evaluator provenance separately."""
    import torch
    output = private_path(output, writable=True)
    if output.exists():
        raise ValueError("evaluation requires a new report path")
    plan_bytes = private_path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    model_bytes = private_path(model_path).read_bytes()
    checkpoint = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    if checkpoint["plan_sha256"] != plan_hash or checkpoint["labels"] != plan["labels"]:
        raise ValueError("model does not match the loaded plan")
    torch.set_num_threads(4)
    model = model_for(len(plan["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    report = {"kind": "species_checkpoint_reevaluation", "plan_sha256": plan_hash,
              "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
              "training_implementation_sha256": checkpoint.get("implementation_sha256"),
              "evaluation_implementation_sha256": IMPLEMENTATION_SHA256,
              "real_holdout": real_results(model, plan, labels_path),
              "limitations": plan["limitations"] + ["No retraining; small same-session real holdout with repeated pets."]}
    dump(output, report)
    return report


def _canonical(value):
    return isinstance(value, str) and bool(value) and value == " ".join(value.split()).casefold()


def _pixel_key(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def real_input_distance(first, second):
    """Training includes mirror augmentation, so mirrored aliases also leak."""
    return min(float(np.abs(first - second).mean()), float(np.abs(first[:, :, ::-1] - second).mean()))


def load_real_crops(manifest_path, *, _manifest_bytes=None):
    """Validate independently labeled real crops before any fitting or scoring.

    Each sample explicitly supplies image, rect, expected, occupied, split,
    session, pose_group and label_source. Unknown occupied pets also name their
    source_species: their species stay disjoint between splits. A pose_group
    identifies the same portrait appearance across frames/sessions, not a slot.
    This gate can detect leakage; metadata cannot prove independent labeling.
    """
    manifest_path = private_path(manifest_path)
    manifest_bytes = manifest_path.read_bytes() if _manifest_bytes is None else _manifest_bytes
    manifest = json.loads(manifest_bytes)
    known = manifest.get("known_species", [])
    if (manifest.get("version") != 1 or not known or len(known) != len(set(known))
            or any(not _canonical(name) or name.startswith("__") for name in known)):
        raise ValueError("real manifest needs version 1 and unique canonical known_species")
    labels = [UNKNOWN, EMPTY, *sorted(known)]
    samples, arrays = [], []
    groups, identities, unknown_species = {}, {}, {}
    source_hashes = {}
    for entry in manifest.get("samples", []):
        expected, occupied, split = entry.get("expected"), entry.get("occupied"), entry.get("split")
        if (expected not in labels or type(occupied) is not bool or (expected == EMPTY) == occupied
                or split not in REAL_SPLITS):
            raise ValueError("invalid real sample label, occupied flag or split")
        for key in ("session", "pose_group", "label_source"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"real sample needs explicit {key}")
        source_species = entry.get("source_species", expected)
        if expected == UNKNOWN and (not _canonical(source_species) or source_species in labels
                                    or source_species.startswith("__")):
            raise ValueError("occupied unknown needs an independently labeled unfamiliar source_species")
        if expected not in (UNKNOWN, EMPTY) and source_species != expected:
            raise ValueError("known source_species must equal expected")
        if expected == UNKNOWN:
            previous = unknown_species.setdefault(source_species, split)
            if previous != split:
                raise ValueError("unknown source species crosses real splits")
        path = private_path(manifest_path.parent / entry["image"])
        image_bytes = path.read_bytes()
        frame_hash = hashlib.sha256(image_bytes).hexdigest()
        if source_hashes.setdefault(str(path), frame_hash) != frame_hash:
            raise ValueError("real source image changed while loading samples")
        with Image.open(io.BytesIO(image_bytes)) as source:
            source = source.convert("RGB")
            rect = entry.get("rect")
            if (not isinstance(rect, list) or len(rect) != 4 or any(type(n) is not int for n in rect)
                    or min(rect[:2]) < 0 or min(rect[2:]) <= 0
                    or rect[0] + rect[2] > source.width or rect[1] + rect[3] > source.height):
                raise ValueError("real crop rectangle must be inside its source image")
            x, y, width, height = rect
            crop = source.crop((x, y, x + width, y + height))
            frame_pixels = _pixel_key(source)
        array = preprocess(crop)
        input_hash = hashlib.sha256(array.tobytes()).hexdigest()
        mirror_hash = hashlib.sha256(array[:, :, ::-1].tobytes()).hexdigest()
        row = {**entry, "source_species": source_species, "image": str(path),
               "frame_sha256": source_hashes[str(path)], "frame_pixel_sha256": frame_pixels,
               "crop_sha256": _pixel_key(crop), "input_sha256": input_hash,
               "mirror_group_sha256": min(input_hash, mirror_hash)}
        # Frame bytes/pixels, declared sessions/poses, raw crops and effective
        # model inputs all stay together, including copied or resized aliases.
        for key, value in (("session", row["session"]), ("pose_group", row["pose_group"]),
                           ("frame", frame_pixels), ("crop", row["crop_sha256"]), ("input", row["input_sha256"]),
                           ("mirror", row["mirror_group_sha256"])):
            previous = groups.setdefault((key, value), split)
            if previous != split:
                raise ValueError(f"{key} crosses real splits")
        for key in (row["crop_sha256"], row["mirror_group_sha256"]):
            previous = identities.setdefault(key, expected)
            if previous != expected:
                raise ValueError("identical real crop has conflicting labels")
        for previous, previous_array in zip(samples, arrays):
            if previous["split"] != split and real_input_distance(previous_array, array) <= REAL_NEAR_DUPLICATE_MAE:
                raise ValueError("near-duplicate model inputs cross real splits")
        samples.append(row)
        arrays.append(array)
    if not samples:
        raise ValueError("real manifest has no samples")
    if manifest.get("test_withheld") is True and any(row["split"] == "test" for row in samples):
        raise ValueError("test_withheld manifest cannot contain test samples")
    required_splits = REAL_SPLITS[:2] if manifest.get("test_withheld") is True else REAL_SPLITS
    for split in required_splits:
        present = {row["expected"] for row in samples if row["split"] == split}
        required = set(labels) if split == "train" else {UNKNOWN, EMPTY}
        if not required <= present or not (set(known) & present):
            raise ValueError(f"incomplete real {split}: need known pets, occupied unknowns and explicit empties")
    references = []
    for name in manifest.get("reference_images", []):
        path = private_path(manifest_path.parent / name)
        image_bytes = path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as reference:
            array = preprocess(reference)
        references.append({"path": str(path), "sha256": hashlib.sha256(image_bytes).hexdigest()})
        for row, candidate in zip(samples, arrays):
            if row["split"] != "train" and real_input_distance(candidate, array) <= REAL_NEAR_DUPLICATE_MAE:
                raise ValueError("held-out real crop overlaps a calibration/template reference")
    # Within-split aliases are reported and receive one vote/weight per input.
    unique = {}
    for row, array in zip(samples, arrays):
        unique.setdefault((row["split"], row["mirror_group_sha256"]), (row, array))
    provenance = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "source_sha256": source_hashes, "reference_images": references,
                  "preprocessing": PREPROCESSING, "near_duplicate_mae": REAL_NEAR_DUPLICATE_MAE,
                  "dataset_metadata": {key: value for key, value in manifest.items() if key not in ("samples", "reference_images")},
                  "sample_count": len(samples), "unique_input_count": len(unique),
                  "counts": {split: {label: sum(row["split"] == split and row["expected"] == label
                                                for row, _ in unique.values()) for label in labels}
                             for split in REAL_SPLITS}}
    return {"labels": labels, "samples": samples, "unique": list(unique.values()), "provenance": provenance,
            "manifest_bytes": manifest_bytes,
            "test_withheld": manifest.get("test_withheld") is True}


def score_real_predictions(records, threshold):
    """Only an accepted known species is an identity claim; empty stays distinct."""
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not .95 <= threshold <= 1:
        raise ValueError("real confidence threshold must be between .95 and one")
    result = {"threshold": threshold, "samples": len(records), "known_samples": 0, "known_correct": 0,
              "known_wrong_accepts": 0, "unknown_samples": 0, "unknown_false_accepts": 0,
              "empty_samples": 0, "empty_false_accepts": 0, "empty_correct": 0,
              "occupied_false_empty": 0, "accepted_species": 0, "per_species": {}}
    scored = []
    for original in records:
        row = dict(original)
        top, expected = row["top_label"], row["expected"]
        prediction = top if row["confidence"] >= threshold else UNKNOWN
        accepted = prediction not in (UNKNOWN, EMPTY)
        row["prediction"] = prediction
        scored.append(row)
        result["accepted_species"] += int(accepted)
        if expected == EMPTY:
            result["empty_samples"] += 1
            result["empty_false_accepts"] += int(accepted)
            result["empty_correct"] += int(prediction == EMPTY)
        elif expected == UNKNOWN:
            result["unknown_samples"] += 1
            result["unknown_false_accepts"] += int(accepted)
        else:
            result["known_samples"] += 1
            result["known_correct"] += int(prediction == expected)
            result["known_wrong_accepts"] += int(accepted and prediction != expected)
            entry = result["per_species"].setdefault(expected, {"samples": 0, "correct": 0})
            entry["samples"] += 1
            entry["correct"] += int(prediction == expected)
        result["occupied_false_empty"] += int(expected != EMPTY and prediction == EMPTY)
    result["records"] = scored
    result["known_recall"] = result["known_correct"] / result["known_samples"] if result["known_samples"] else None
    result["accepted_species_precision"] = result["known_correct"] / result["accepted_species"] if result["accepted_species"] else None
    # These are observed rates only. Correlated crops cannot support a
    # population error bound by treating every screenshot as independent.
    for name in ("unknown", "empty"):
        total, errors = result[f"{name}_samples"], result[f"{name}_false_accepts"]
        result[f"{name}_false_accept_rate"] = errors / total if total else None
    return result


def calibrate_real_threshold(records):
    """Select once on calibration only, never lowering the original .95 floor.

    Empty predictions on occupied crops also count as unsafe. If even confidence
    one cannot reject a wrong certainty, calibration fails closed explicitly.
    Zero observed errors on this small set is not a statistical guarantee.
    """
    candidates = sorted({.95, 1., *(float(np.nextafter(row["confidence"], 1.))
                                   for row in records if .95 <= row["confidence"] < 1.)})
    for threshold in candidates:
        report = score_real_predictions(records, threshold)
        if not sum(report[key] for key in ("known_wrong_accepts", "unknown_false_accepts",
                                           "empty_false_accepts", "occupied_false_empty")):
            return {"status": "calibrated", "threshold": threshold, "report": report}
    return {"status": "unsafe_at_all_thresholds", "threshold": 1.,
            "report": score_real_predictions(records, 1.)}


def train_real(manifest_path, output, seconds=900, max_steps=20000, threads=3, batch_size=32):
    """Train solely on deduplicated real training crops; touch test once at end."""
    import torch
    if (isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds)
            or not 0 < seconds <= 2700 or any(type(n) is not int or n <= 0 for n in (threads, max_steps, batch_size))
            or threads > 3):
        raise ValueError("real training bounds must be positive, at most 2700 seconds and three threads")
    output = private_path(output)
    if output.exists():
        raise ValueError("real training requires a new output directory")
    data = load_real_crops(manifest_path)
    output.mkdir(parents=True)
    write_output(output / "training-source.py", lambda handle: handle.write(IMPLEMENTATION_SOURCE))
    write_output(output / "training-manifest.json", lambda handle: handle.write(data["manifest_bytes"]))
    torch.set_num_threads(threads)
    torch.manual_seed(48271)
    rng = random.Random(48271)
    labels = data["labels"]
    training = [(row, array) for row, array in data["unique"] if row["split"] == "train"]
    by_label = [[array for row, array in training if row["expected"] == label] for label in labels]
    model = model_for(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_function = torch.nn.CrossEntropyLoss()
    start, last, step, history = time.monotonic(), time.monotonic(), 0, []
    loss = None
    while step < max_steps and time.monotonic() - start < seconds:
        model.train()
        targets = [(step * batch_size + i) % len(labels) for i in range(batch_size)]
        batch = []
        for target in targets:
            array = rng.choice(by_label[target]).copy()
            if rng.random() < .5:
                array = array[:, :, ::-1].copy()
            # Mild photometry only. No composites, new labels or pose claims.
            batch.append(np.clip(array * rng.uniform(.94, 1.06), 0., 1.))
        optimizer.zero_grad()
        loss = loss_function(model(torch.from_numpy(np.stack(batch))), torch.tensor(targets))
        loss.backward()
        optimizer.step()
        step += 1
        if time.monotonic() - last >= 60:
            history.append({"step": step, "seconds": time.monotonic() - start, "training_loss": float(loss.detach())})
            dump(output / "progress.json", history)
            print(json.dumps(history[-1]), flush=True)
            last = time.monotonic()
    model.eval()
    def predictions(split):
        rows = []
        with torch.no_grad():
            for row, array in data["unique"]:
                if row["split"] != split:
                    continue
                probabilities = model(torch.from_numpy(array.copy()).unsqueeze(0)).softmax(1)[0]
                probability, index = probabilities.max(0)
                rows.append({**row, "top_label": labels[int(index)], "confidence": float(probability)})
        return rows
    calibration = calibrate_real_threshold(predictions("calibration"))
    # Fixed schedule; calibration never selects weights or stopping time.
    test_predictions = predictions("test") if not data["test_withheld"] else None
    report = {"kind": "real_crop_species_experiment", "steps": step, "seconds": time.monotonic() - start,
              "threads": threads, "seed": 48271, "device": "cpu", "labels": labels, "history": history,
              "implementation_sha256": IMPLEMENTATION_SHA256, **data["provenance"],
              "calibration": calibration, "test_state": "withheld" if data["test_withheld"] else "evaluated",
              "test": score_real_predictions(test_predictions, calibration["threshold"]) if test_predictions is not None else None,
              "test_at_original_threshold": score_real_predictions(test_predictions, .95) if test_predictions is not None else None,
              "training": score_real_predictions(predictions("train"), .95),
              "eligible_for_promotion": False,
              "limitations": ["Small manually labeled dataset; session/pose metadata requires independent review.",
                              "Near-duplicate filtering cannot prove novel poses or independent labeling.",
                              "Calibration and test report unique inputs; nearby within-split crops remain correlated.",
                              "Fixed training schedule; test never selects thresholds, weights or stopping time.",
                              "Offline only; no recognizer/profile integration or production readiness claim."]}
    report["missing_known_species"] = {split: sorted(set(labels) - {UNKNOWN, EMPTY} -
                                                   {row["expected"] for row, _ in data["unique"] if row["split"] == split})
                                       for split in ("calibration", "test")}
    write_output(output / "model.pt", lambda handle: torch.save(
        {"state_dict": model.state_dict(), "labels": labels, "manifest_sha256": data["provenance"]["manifest_sha256"],
         "implementation_sha256": IMPLEMENTATION_SHA256, "preprocessing": PREPROCESSING,
         "calibration_threshold": calibration["threshold"], "calibration_status": calibration["status"],
         "data_provenance": data["provenance"]}, handle))
    dump(output / "data-audit.json", {**data["provenance"], "samples": data["samples"]})
    dump(output / "report.json", report)
    print(json.dumps({"steps": step, "seconds": report["seconds"], "calibration": calibration["status"],
                      "threshold": calibration["threshold"], "test": {k: v for k, v in report["test"].items() if k != "records"}
                      if report["test"] is not None else None}), flush=True)
    return report


def evaluate_real_checkpoint(training_manifest_path, manifest_path, model_path, output, threads=3):
    """Open a final test only after weights, calibration and training are frozen."""
    import torch
    if type(threads) is not int or not 0 < threads <= 3:
        raise ValueError("evaluation needs one to three threads")
    output = private_path(output, writable=True)
    if output.exists():
        raise ValueError("real evaluation requires a new report path")
    training = load_real_crops(training_manifest_path)
    model_bytes = private_path(model_path).read_bytes()
    checkpoint = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    if (checkpoint["manifest_sha256"] != training["provenance"]["manifest_sha256"]
            or checkpoint["labels"] != training["labels"]
            or checkpoint.get("preprocessing") != PREPROCESSING):
        raise ValueError("real model does not match frozen training manifest/preprocessing")
    frozen = checkpoint.get("data_provenance")
    if (not isinstance(frozen, dict) or frozen.get("manifest_sha256") != checkpoint["manifest_sha256"]
            or any(key not in frozen for key in ("source_sha256", "reference_images"))):
        raise ValueError("real checkpoint lacks frozen source/reference provenance; retrain")
    if any(frozen[key] != training["provenance"][key] for key in ("source_sha256", "reference_images")):
        raise ValueError("training images or references changed since the frozen checkpoint")
    if not training["test_withheld"]:
        raise ValueError("final evaluation needs a withheld training manifest")
    # Check the actual frozen weights and rejection policy before touching test
    # imagery. An unsafe calibration keeps the final test unopened.
    score_real_predictions([], checkpoint["calibration_threshold"])
    if checkpoint["calibration_status"] != "calibrated":
        raise ValueError("real checkpoint calibration is unsafe; final test remains withheld")
    torch.set_num_threads(threads)
    model = model_for(len(training["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Compare declarations using JSON alone: even a changed label_source, rect
    # or pose in a non-test row must fail before Image.open can see final test.
    original = json.loads(training["manifest_bytes"])
    final_manifest_path = private_path(manifest_path)
    final_bytes = final_manifest_path.read_bytes()
    declarations = json.loads(final_bytes)
    if (declarations.get("known_species") != original["known_species"]
            or declarations.get("version") != original["version"] or declarations.get("test_withheld") is True
            or not isinstance(declarations.get("samples"), list)):
        raise ValueError("final evaluation needs withheld training and a complete evaluation manifest")
    if ([row for row in declarations["samples"] if row.get("split") != "test"] != original["samples"]
            or declarations.get("reference_images", []) != original.get("reference_images", [])):
        raise ValueError("training/calibration declarations or reference paths changed before final test")
    # Identical relative declarations can point at different files when the
    # final manifest lives elsewhere. Check their effective paths without
    # opening any imagery before the full loader sees test rows.
    declared_sources = [private_path(final_manifest_path.parent / row["image"])
                        for row in declarations["samples"] if row["split"] != "test"]
    declared_references = [private_path(final_manifest_path.parent / name)
                           for name in declarations.get("reference_images", [])]
    if (declared_sources != [Path(row["image"]) for row in training["samples"]]
            or declared_references != [Path(row["path"]) for row in training["provenance"]["reference_images"]]):
        raise ValueError("training/calibration source or reference paths changed before final test")
    # Reuse those exact bytes so a second manifest read cannot bypass preflight.
    evaluation = load_real_crops(manifest_path, _manifest_bytes=final_bytes)
    if (training["samples"] != [row for row in evaluation["samples"] if row["split"] != "test"]
            or training["provenance"]["reference_images"] != evaluation["provenance"]["reference_images"]):
        raise ValueError("training/calibration samples or references changed before final test")
    records = []
    with torch.no_grad():
        for row, array in evaluation["unique"]:
            if row["split"] == "test":
                probability, index = model(torch.from_numpy(array.copy()).unsqueeze(0)).softmax(1)[0].max(0)
                records.append({**row, "top_label": training["labels"][int(index)], "confidence": float(probability)})
    report = {"kind": "frozen_real_crop_final_test", "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
              "training_manifest_sha256": checkpoint["manifest_sha256"], **evaluation["provenance"],
              "training_implementation_sha256": checkpoint["implementation_sha256"],
              "evaluation_implementation_sha256": IMPLEMENTATION_SHA256,
              "calibration_status": checkpoint["calibration_status"],
              "test": score_real_predictions(records, checkpoint["calibration_threshold"]),
              "test_at_original_threshold": score_real_predictions(records, .95), "eligible_for_promotion": False,
              "missing_known_species": {split: sorted(set(training["labels"]) - {UNKNOWN, EMPTY} -
                                                       {row["expected"] for row in evaluation["samples"] if row["split"] == split})
                                         for split in ("calibration", "test")},
              "limitations": ["Single fixed checkpoint/threshold; test does not select either.",
                              "Small independent test; no production readiness claim."]}
    dump(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("profile", "assets", "negatives", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--plan", type=Path, required=True)
    training.add_argument("--labels", type=Path, required=True)
    training.add_argument("--seconds", type=float, default=3300)
    training.add_argument("--steps", type=int, default=200000)
    training.add_argument("--threads", type=int, default=4)
    evaluation = commands.add_parser("evaluate")
    for name in ("plan", "labels", "model", "output"):
        evaluation.add_argument("--" + name, type=Path, required=True)
    real = commands.add_parser("train-real", help="offline real crops with session/pose-disjoint calibration and test")
    real.add_argument("--manifest", type=Path, required=True)
    real.add_argument("--output", type=Path, required=True)
    real.add_argument("--seconds", type=float, default=900)
    real.add_argument("--steps", type=int, default=20000)
    real.add_argument("--threads", type=int, default=3)
    final = commands.add_parser("evaluate-real", help="score a frozen real model on a newly labeled final test")
    for name in ("training-manifest", "manifest", "model", "output"):
        final.add_argument("--" + name, type=Path, required=True)
    final.add_argument("--threads", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare(args.profile, args.assets, args.negatives, args.output)
        print(json.dumps({"entries": len(plan["entries"]), "labels": plan["labels"]}))
    elif args.command == "train":
        train(args.plan, args.labels, args.seconds, args.steps, threads=args.threads)
    elif args.command == "evaluate":
        report = evaluate_checkpoint(args.plan, args.labels, args.model, args.output)
        print(json.dumps({key: value for key, value in report["real_holdout"].items() if key != "records"}))
    elif args.command == "train-real":
        train_real(args.manifest, args.output, args.seconds, args.steps, args.threads)
    else:
        report = evaluate_real_checkpoint(args.training_manifest, args.manifest, args.model, args.output, args.threads)
        print(json.dumps({key: value for key, value in report["test"].items() if key != "records"}))


if __name__ == "__main__":
    main()
