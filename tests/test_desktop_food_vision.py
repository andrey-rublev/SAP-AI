"""Optional food observations from synthetic calibrated images only."""
import copy
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from sap_ai.desktop.state import FoodSlot, Phase
from sap_ai.desktop.cli import validate_profile_assets
from sap_ai.desktop.vision import FoodSlotConfig, Perceptor, Rect, VisionProfile


def save_template(path, value, size=(8, 8)):
    Image.new("RGB", size, (value, value, value)).save(path)


@pytest.fixture
def scene(tmp_path):
    for name, value in (("shop", 200), ("empty", 0), ("apple", 150), ("pear", 230)):
        save_template(tmp_path / f"{name}.png", value)
    frame = np.zeros((24, 64, 3), dtype=np.uint8)
    frame[:8, :8] = 200
    frame[10:18, 10:18] = 150
    frame[10, 20] = 3
    config = {"version": 1, "image_size": [64, 24], "calibrated": True,
              "phase_templates": [{"phase": "shop", "region": [0, 0, 8, 8], "template": "shop.png"}],
              "food": [{"portrait": [10, 10, 8, 8], "cost": [20, 10, 1, 1],
                        "empty_template": "empty.png", "food_templates": {"apple": "apple.png", "pear": "pear.png"}}]}
    def ocr(crop):
        code = int(crop[0, 0, 0])
        return "?" if code == 255 else str(code)
    return config, frame, tmp_path, ocr


def observe(scene, *, ocr=None):
    config, frame, base, default_ocr = scene
    return Perceptor(VisionProfile.from_dict(config, base_dir=base), ocr=default_ocr if ocr is None else ocr).observe(frame)


def test_food_identity_and_price_require_separate_explicit_evidence(scene):
    food = observe(scene).food[0]
    assert food == FoodSlot(True, food_id="apple", cost=3)
    assert food.effect is food.attack_bonus is food.health_bonus is None
    scene[0]["food"][0].pop("cost")
    assert observe(scene).food[0] == FoodSlot(True, food_id="apple")


@pytest.mark.parametrize("raw", ["?", "", "0", "-1", "3 gold", "100", None])
def test_unreadable_or_out_of_range_price_stays_unknown_without_erasing_identity(scene, raw):
    assert observe(scene, ocr=lambda image: raw).food[0] == FoodSlot(True, food_id="apple")


def test_explicit_empty_wins_only_without_conflicting_price(scene):
    scene[1][10:18, 10:18] = 0
    assert observe(scene).food[0] == FoodSlot(None)
    scene[1][10, 20] = 255
    assert observe(scene).food[0] == FoodSlot(False)


def test_empty_mismatch_or_readable_price_does_not_establish_unfamiliar_food_occupancy(scene):
    scene[1][10:18, 10:18] = 70
    assert observe(scene).food[0] == FoodSlot(None)
    scene[0]["food"][0]["food_templates"] = {}
    assert observe(scene).food[0] == FoodSlot(None)


@pytest.mark.parametrize("frame_value", [0, 2])
def test_near_empty_and_food_evidence_abstains_in_both_directions(scene, frame_value):
    save_template(scene[2] / "apple.png", 2)
    scene[1][10:18, 10:18] = frame_value
    scene[1][10, 20] = 255
    assert observe(scene).food[0] == FoodSlot(None)


def test_confident_food_must_beat_empty_even_if_empty_is_within_distance_limit(scene):
    save_template(scene[2] / "empty.png", 140)
    assert observe(scene).food[0] == FoodSlot(True, food_id="apple", cost=3)


def test_ambiguous_food_identity_retains_only_template_grounded_occupancy_and_price(scene):
    save_template(scene[2] / "pear.png", 151)
    assert observe(scene).food[0] == FoodSlot(True, cost=3)


def test_alternate_same_identity_and_canonical_aliases_share_one_competitor(scene):
    save_template(scene[2] / "apple-alternate.png", 80)
    scene[0]["food"][0]["food_templates"].update({" APPLE ": ["apple-alternate.png"], "Apple": "apple.png"})
    scene[1][10:18, 10:18] = 80
    assert observe(scene).food[0].food_id == "apple"
    scene[0]["food"][0]["food_templates"]["pear"] = ["pear.png", "apple-alternate.png"]
    assert observe(scene).food[0] == FoodSlot(True, cost=3)


def test_food_and_empty_observation_can_be_configured_independently(scene):
    scene[0]["food"][0].pop("empty_template")
    assert observe(scene).food[0].food_id == "apple"
    scene[0]["food"][0]["food_templates"] = {}
    assert observe(scene).food[0] == FoodSlot(None)
    scene[0]["food"][0]["empty_template"] = "empty.png"
    scene[1][10:18, 10:18] = 0
    scene[1][10, 20] = 255
    assert observe(scene).food[0] == FoodSlot(False)


def test_price_numeric_reference_is_explicit_and_ocr_disagreement_abstains(scene):
    save_template(scene[2] / "price.png", 3, size=(1, 1))
    scene[0]["numeric_templates"] = [{"field": "food.0.cost", "value": 3, "template": "price.png"}]
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[2])
    assert profile.numeric_region("food.0.cost") == (Rect(20, 10, 1, 1), 1, 99)
    assert Perceptor(profile).observe(scene[1]).food[0].cost == 3
    assert observe(scene, ocr=lambda crop: "?").food[0].cost == 3
    assert observe(scene, ocr=lambda crop: "4").food[0].cost is None


@pytest.mark.parametrize("change", ["missing_region", "missing_slot", "zero_price", "unsupported_stat"])
def test_invalid_numeric_food_references_fail_profile_validation(scene, change):
    item = {"field": "food.0.cost", "value": 3, "template": "price.png"}
    if change == "missing_region":
        scene[0]["food"][0].pop("cost")
    elif change == "missing_slot":
        item["field"] = "food.1.cost"
    elif change == "zero_price":
        item["value"] = 0
    else:
        item["field"] = "food.0.attack"
    scene[0]["numeric_templates"] = [item]
    with pytest.raises(ValueError, match="numeric"):
        VisionProfile.from_dict(scene[0], base_dir=scene[2])


def test_optional_food_roundtrip_keeps_legacy_profiles_and_food_geometry(scene, tmp_path):
    legacy = copy.deepcopy(scene[0])
    legacy.pop("food")
    legacy_profile = VisionProfile.from_dict(legacy, base_dir=scene[2])
    assert "food" not in legacy_profile.to_dict()
    assert Perceptor(legacy_profile).observe(scene[1]).food == ()
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[2])
    assert isinstance(profile.food[0], FoodSlotConfig)
    assert profile.food[0].portrait.center == (14, 14)
    path = tmp_path / "food-profile.json"
    profile.save(path)
    loaded = VisionProfile.load(path)
    assert loaded == profile
    assert loaded.to_dict()["food"] == profile.to_dict()["food"]


def test_non_shop_frame_reads_neither_food_templates_nor_price_ocr(scene):
    scene[1][:8, :8] = 0
    scene[0]["food"][0]["food_templates"]["apple"] = "missing.png"
    engine = Mock(side_effect=AssertionError("non-shop OCR"))
    board = observe(scene, ocr=engine)
    assert board.phase == Phase.UNKNOWN and board.food == ()
    engine.assert_not_called()


@pytest.mark.parametrize("references", [None, [], [""], [1], (), True, {"path": "apple.png"}])
def test_malformed_food_references_are_rejected(scene, references):
    scene[0]["food"][0]["food_templates"]["apple"] = references
    with pytest.raises(ValueError, match="food reference"):
        VisionProfile.from_dict(scene[0], base_dir=scene[2])


@pytest.mark.parametrize("path", ["../outside.png", "/tmp/outside.png"])
def test_food_template_paths_must_stay_inside_profile_directory(scene, path):
    scene[0]["food"][0]["food_templates"]["apple"] = ["apple.png", path]
    with pytest.raises(ValueError, match="template paths"):
        VisionProfile.from_dict(scene[0], base_dir=scene[2])


@pytest.mark.parametrize("field,value", [("portrait", [60, 10, 8, 8]), ("cost", [64, 10, 1, 1]),
                                        ("max_distance", float("nan")), ("margin", -.1),
                                        ("effect", "single_target_stats"), ("attack_bonus", 1), ("cost", 3)])
def test_invalid_or_inferred_food_configuration_is_rejected(scene, field, value):
    scene[0]["food"][0][field] = value
    with pytest.raises(ValueError):
        VisionProfile.from_dict(scene[0], base_dir=scene[2])


@pytest.mark.parametrize("food", [{}, None, [None], [dict(portrait=[10, 10, 8, 8])] * 6])
def test_invalid_food_collection_is_rejected(scene, food):
    scene[0]["food"] = food
    with pytest.raises(ValueError, match="food"):
        VisionProfile.from_dict(scene[0], base_dir=scene[2])


def test_missing_or_missized_food_reference_is_a_configuration_error(scene):
    save_template(scene[2] / "wrong.png", 150, size=(7, 8))
    scene[0]["food"][0]["food_templates"]["apple"] = "wrong.png"
    with pytest.raises(ValueError, match="dimensions"):
        observe(scene)
    scene[0]["food"][0]["food_templates"]["apple"] = "missing.png"
    with pytest.raises(FileNotFoundError):
        observe(scene)


@pytest.mark.parametrize("reference", ["empty", "alternate", "cost"])
def test_all_food_assets_are_checked_before_window_activation(scene, reference):
    config = scene[0]["food"][0]
    if reference == "empty":
        config["empty_template"] = "missing.png"
    elif reference == "alternate":
        # Even an unused variant must be checked before starting the controller.
        config["food_templates"]["apple"] = ["apple.png", "missing.png"]
    else:
        scene[0]["numeric_templates"] = [{"field": "food.0.cost", "value": 3,
                                           "template": "missing.png"}]
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[2])
    with pytest.raises(FileNotFoundError):
        validate_profile_assets(Perceptor(profile))


def test_food_price_engine_failure_invalidates_observation_without_identity_fallback(scene):
    engine = Mock(side_effect=TimeoutError("price OCR timeout"))
    with pytest.raises(TimeoutError, match="price OCR timeout"):
        observe(scene, ocr=engine)
    assert engine.call_count == 1
