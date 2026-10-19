"""Strict local profile JSON loading, without templates or desktop IO."""
import json
from unittest.mock import Mock

import pytest

from sap_ai.desktop.vision import VisionProfile, decode_profile_json


def raw_profile(fields):
    return '{"version":1,"image_size":[40,20],' + fields + '}'


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("fields,key", [
    ('"calibrated":false,"calibrated":true', "calibrated"),
    ('"calibrated":true,"calibrated":true', "calibrated"),
    (r'"calibrated":false,"\u0063alibrated":true', "calibrated"),
    ('"shop":[{"portrait":[0,5,4,4],"level":null,"level":[7,5,1,1]}]', "level"),
    ('"shop":[{"portrait":[0,5,4,4],"available_from_turn":9,"available_from_turn":1}]',
     "available_from_turn"),
    ('"team":[{"portrait":[0,5,4,4],"max_distance":0.001,"max_distance":0.06}]',
     "max_distance"),
    ('"hud":{"gold":[5,0,1,1]},"numeric_templates":[{"field":"gold","value":3,'
     '"template":"three.png","margin":0.01,"margin":0.01}]', "margin"),
    ('"hud":{"gold":[5,0,1,1],"turn":[6,0,1,1]},"numeric_colors":['
     '{"field":"gold","field":"turn","min_rgb":[200,0,0],"max_rgb":[255,50,50]}]', "field"),
    ('"shop":[{"portrait":[0,5,4,4],"species_templates":{"fish":"one.png","fish":"two.png"}}]',
     "fish"),
])
def test_duplicate_decoded_keys_are_rejected_at_any_depth(tmp_path, encoding, fields, key):
    path = tmp_path / "duplicate-profile.json"
    path.write_text(raw_profile(fields), encoding=encoding)
    with pytest.raises(ValueError, match=f"duplicate JSON key in vision profile: '{key}'"):
        VisionProfile.load(path)


def test_duplicate_json_is_rejected_before_profile_conversion(tmp_path, monkeypatch):
    path = tmp_path / "duplicate-profile.json"
    path.write_text(raw_profile('"calibrated":false,"calibrated":true'), encoding="utf-8")
    conversion = Mock(side_effect=AssertionError("ambiguous JSON reached profile conversion"))
    monkeypatch.setattr(VisionProfile, "from_dict", conversion)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        VisionProfile.load(path)
    conversion.assert_not_called()


def test_same_field_names_in_separate_objects_remain_valid(tmp_path):
    data = {"version": 1, "image_size": [40, 20], "calibrated": True,
            "hud": {"gold": [5, 0, 1, 1]},
            "shop": [{"portrait": [0, 5, 4, 4]}, {"portrait": [10, 5, 4, 4]}],
            "numeric_templates": [
                {"field": "gold", "value": 2, "template": "two.png"},
                {"field": "gold", "value": 3, "template": "three.png"},
            ]}
    path = tmp_path / "separate-objects.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    expected = VisionProfile.from_dict(data, base_dir=tmp_path)
    assert VisionProfile.load(path) == expected
    assert len(expected.shop) == len(expected.numeric_templates) == 2


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_unicode_and_legacy_defaults_load_and_roundtrip_with_optional_bom(tmp_path, encoding):
    data = {"version": 1, "image_size": [40, 20], "calibrated": True,
            "shop": [{"portrait": [0, 5, 4, 4], "level": None,
                      "species_templates": {"猫": ["猫.png", "é.png"]}}]}
    path = tmp_path / "unicode-profile.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding=encoding)
    expected = VisionProfile.from_dict(data, base_dir=tmp_path)
    profile = VisionProfile.load(path)
    assert profile == expected
    assert profile.shop[0].species_templates == {"猫": ["猫.png", "é.png"]}
    assert profile.shop[0].level is None
    assert (profile.shop[0].max_distance, profile.shop[0].margin,
            profile.shop[0].available_from_turn) == (.06, .015, 1)
    assert not profile.numeric_templates and not profile.numeric_colors
    saved = tmp_path / "saved-profile.json"
    profile.save(saved)
    assert VisionProfile.load(saved) == profile


@pytest.mark.parametrize("text", ['{"version":1,', '{"version":1,"image_size":[40,20],}'])
def test_invalid_json_syntax_remains_rejected(tmp_path, text):
    path = tmp_path / "invalid-profile.json"
    path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(json.JSONDecodeError):
        VisionProfile.load(path)


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("bom", [False, True])
def test_shared_decoder_preserves_partial_unicode_profiles(as_bytes, bom):
    data = {"shop": [{"species_templates": {"猫": "é.png"}}]}
    text = ("\ufeff" if bom else "") + json.dumps(data, ensure_ascii=False)
    payload = text.encode("utf-8") if as_bytes else text
    assert decode_profile_json(payload) == data


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("text", [
    '{"shop":[{"portrait":null,"portrait":null}]}',
    r'{"shop":[{"portrait":null,"portrai\u0074":null}]}',
])
def test_shared_decoder_rejects_nested_equal_and_escaped_duplicates(as_bytes, text):
    payload = text.encode("utf-8") if as_bytes else text
    with pytest.raises(ValueError, match="duplicate JSON key.*portrait"):
        decode_profile_json(payload)


@pytest.mark.parametrize("payload", [None, {}, [], 1, True])
def test_shared_decoder_requires_text_or_bytes(payload):
    with pytest.raises(TypeError, match="profile JSON must be UTF-8 text or bytes"):
        decode_profile_json(payload)


@pytest.mark.parametrize("as_bytes", [False, True])
def test_shared_decoder_rejects_two_leading_boms(as_bytes):
    text = "\ufeff\ufeff" + '{"shop":[]}'
    with pytest.raises(json.JSONDecodeError):
        decode_profile_json(text.encode("utf-8") if as_bytes else text)


def test_file_loader_does_not_strip_a_second_bom(tmp_path):
    path = tmp_path / "double-bom-profile.json"
    path.write_bytes(("\ufeff\ufeff" + raw_profile('"calibrated":true')).encode("utf-8"))
    with pytest.raises(json.JSONDecodeError):
        VisionProfile.load(path)
