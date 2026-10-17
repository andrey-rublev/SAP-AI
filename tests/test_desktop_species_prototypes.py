import json

import numpy as np
import pytest

from tools import eval_desktop_species_prototypes as prototypes


def dataset():
    fish = np.random.default_rng(78).uniform(.2, .8, (3, 32, 64)).astype(np.float32)
    labels = [prototypes.crops.UNKNOWN, prototypes.crops.EMPTY, "fish", "horse"]
    values = [np.zeros_like(fish), np.ones_like(fish), fish, np.full_like(fish, .1)]
    pairs = [({"split": "train", "expected": label, "input_sha256": str(i)}, array)
             for i, (label, array) in enumerate(zip(labels, values))]
    return {"labels": labels, "unique": pairs}, fish


def test_mirror_matching_and_same_class_aliases_do_not_create_cross_class_ambiguity():
    data, fish = dataset()
    data["unique"].append(({"split": "train", "expected": "fish", "input_sha256": "alternate"}, fish + .001))
    classifier = prototypes.Prototypes(data)
    direct = classifier.distances(fish[:, :, ::-1], mirrored=False)
    mirrored = classifier.distances(fish[:, :, ::-1], mirrored=True)
    assert direct["distance"] > .1
    assert mirrored["top_label"] == "fish" and mirrored["distance"] == 0
    assert mirrored["runner_up_label"] != "fish"
    assert mirrored["cross_class_margin"] > .015


def test_calibration_tightens_rejection_without_accepting_unfamiliar_pet():
    records = [{"expected": "fish", "top_label": "fish", "distance": .01, "cross_class_margin": .06},
               {"expected": prototypes.crops.UNKNOWN, "top_label": "fish", "distance": .02, "cross_class_margin": .05},
               {"expected": prototypes.crops.EMPTY, "top_label": prototypes.crops.EMPTY,
                "distance": .005, "cross_class_margin": .06}]
    labels = [prototypes.crops.UNKNOWN, prototypes.crops.EMPTY, "fish", "horse"]
    assert prototypes.score(records, labels)["unknown_false_accepts"] == 1
    calibrated = prototypes.calibrate(records, labels)
    assert calibrated["report"]["known_correct"] == 1
    assert not any(calibrated["report"][key] for key in prototypes.UNSAFE)
    assert calibrated["policy"]["max_distance"] <= .04
    assert calibrated["policy"]["margin"] >= .015
    assert calibrated["report"]["missing_known_species"] == ["horse"]


@pytest.mark.parametrize("maximum,margin", [(.041, .015), (.04, .014), (float("nan"), .015), (True, .015)])
def test_prototype_policy_cannot_loosen_distance_or_margin(maximum, margin):
    with pytest.raises(ValueError, match="distance <=.04"):
        prototypes.validate_policy(maximum, margin)


def test_prototype_job_refuses_final_test_before_loading_any_images(tmp_path, monkeypatch):
    root = tmp_path / ".local"
    root.mkdir()
    monkeypatch.setattr(prototypes.crops, "PRIVATE_ROOT", root)
    manifest = root / "final.json"
    manifest.write_text(json.dumps({"test_withheld": False, "samples": [{"split": "test"}]}))
    monkeypatch.setattr(prototypes.crops, "load_real_crops", lambda *args, **kwargs: pytest.fail("must not open any test imagery"))
    with pytest.raises(ValueError, match="withheld final test"):
        prototypes.run(manifest, root / "refused")
    assert not (root / "refused").exists()
