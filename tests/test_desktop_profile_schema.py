"""Reject discarded calibration options before assets or desktop IO are used."""
import copy
import json
from unittest.mock import Mock

import pytest

import desktop
from desktop_vision import NumberTemplate, SlotConfig, VisionProfile


def profile_data():
    return {"version": 1, "image_size": [40, 20], "calibrated": True,
            "hud": {"gold": [5, 0, 1, 1]},
            "shop": [{"portrait": [0, 5, 4, 4]}],
            "team": [{"portrait": [10, 5, 4, 4]}],
            "numeric_templates": [{"field": "gold", "value": 3, "template": "three.png"}]}


@pytest.mark.parametrize("row", ["shop", "team"])
@pytest.mark.parametrize("key,value", [
    ("available_from_tun", 9), ("max_distnce", .001),
    ("level_region", [7, 5, 1, 1]), ("species_template", "fish.png"),
    ("unsupported", None),
])
def test_pet_slot_rejects_misspelled_or_unsupported_options(row, key, value):
    data = profile_data()
    data[row][0][key] = value
    with pytest.raises(ValueError, match="pet slot must contain only supported observation fields"):
        VisionProfile.from_dict(data)


@pytest.mark.parametrize("key,value", [
    ("max_distnce", .0001), ("margn", .1),
    ("region", [5, 0, 1, 1]), ("unsupported", None),
])
def test_numeric_template_rejects_misspelled_or_unsupported_options(key, value):
    data = profile_data()
    data["numeric_templates"][0][key] = value
    with pytest.raises(ValueError, match="numeric template must contain only supported fields"):
        VisionProfile.from_dict(data)


@pytest.mark.parametrize("value", [None, [], [["field", "gold"]], "invalid", 7, True])
@pytest.mark.parametrize("parser,label", [(SlotConfig.from_dict, "pet slot"),
                                         (NumberTemplate.from_dict, "numeric template")])
def test_nested_parsers_require_dictionary_objects(parser, label, value):
    with pytest.raises(ValueError, match=label):
        parser(value)


def test_legacy_optional_regions_and_threshold_defaults_roundtrip(tmp_path):
    data = profile_data()
    data["shop"][0].update(attack=None, health=None, level=None, empty_template=None)
    original = copy.deepcopy(data)
    profile = VisionProfile.from_dict(data, base_dir=tmp_path)
    assert data == original
    slot = profile.shop[0]
    assert slot.attack is slot.health is slot.level is slot.empty_template is None
    assert (slot.max_distance, slot.margin, slot.available_from_turn) == (.06, .015, 1)
    assert slot.species_templates == {}
    number = profile.numeric_templates[0]
    assert (number.max_distance, number.margin) == (.01, .01)
    path = tmp_path / "legacy-profile.json"
    profile.save(path)
    assert VisionProfile.load(path) == profile
    assert "food" not in profile.to_dict()


def test_all_current_serialized_fields_remain_supported(tmp_path):
    data = profile_data()
    data["shop"][0].update(
        attack=[5, 5, 1, 1], health=[6, 5, 1, 1], level=[7, 5, 1, 1],
        empty_template="empty.png", species_templates={" FISH ": ["fish.png", "fish-alt.png"]},
        max_distance=.025, margin=.02, available_from_turn=9,
    )
    data["numeric_templates"][0].update(max_distance=.002, margin=.003)
    profile = VisionProfile.from_dict(data, base_dir=tmp_path)
    encoded = profile.to_dict()
    assert encoded["shop"] == data["shop"]
    assert encoded["numeric_templates"] == data["numeric_templates"]
    assert VisionProfile.from_dict(encoded, base_dir=tmp_path) == profile


@pytest.mark.parametrize("collection,option", [
    ("shop", ("available_from_tun", 9)),
    ("team", ("level_region", [7, 5, 1, 1])),
    ("numeric_templates", ("max_distnce", .0001)),
])
def test_play_check_rejects_discarded_options_before_assets_or_native_dependencies(
        tmp_path, monkeypatch, collection, option):
    data = profile_data()
    data[collection][0][option[0]] = option[1]
    path = tmp_path / "invalid-profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    perceptor = Mock(side_effect=AssertionError("perception constructed before schema validation"))
    assets = Mock(side_effect=AssertionError("template assets read before schema validation"))
    dependencies = Mock(side_effect=AssertionError("native dependencies checked before schema validation"))
    window = Mock(side_effect=AssertionError("desktop window constructed before schema validation"))
    monkeypatch.setattr(desktop, "Perceptor", perceptor)
    monkeypatch.setattr(desktop, "validate_profile_assets", assets)
    monkeypatch.setattr(desktop, "validate_live_dependencies", dependencies)
    monkeypatch.setattr(desktop, "WindowsGameWindow", window)
    with pytest.raises(ValueError, match="must contain only supported"):
        desktop.main(["play", "--check", "--profile", str(path)])
    for dependency in (perceptor, assets, dependencies, window):
        dependency.assert_not_called()
