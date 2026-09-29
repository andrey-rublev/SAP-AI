"""Private, offline sprite-to-desktop species experiment; never controls the game.

Fit partial-crop geometry on calibration templates, train a tiny CPU classifier on
composited sprites, and report held-out real-frame results separately. This is not
a production recognizer and does not modify the desktop vision profile.
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
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

UNKNOWN = "__unknown__"
INPUT_SIZE = (64, 32)
PRIVATE_ROOT = Path(__file__).resolve().parents[1] / ".local"
# Capture the implementation loaded for this process, never a later file revision.
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
    profile = json.loads(profile_bytes)
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
            for species, path in slot["species_templates"].items():
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
    profile = json.loads(profile_bytes)
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
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare(args.profile, args.assets, args.negatives, args.output)
        print(json.dumps({"entries": len(plan["entries"]), "labels": plan["labels"]}))
    elif args.command == "train":
        train(args.plan, args.labels, args.seconds, args.steps, threads=args.threads)
    else:
        report = evaluate_checkpoint(args.plan, args.labels, args.model, args.output)
        print(json.dumps({key: value for key, value in report["real_holdout"].items() if key != "records"}))


if __name__ == "__main__":
    main()
