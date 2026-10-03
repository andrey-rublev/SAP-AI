"""Synthetic image tests; never capture the screen or move the mouse."""
import copy

import numpy as np
import pytest

from desktop_state import Action, DesktopPolicy, Phase, legal_action
from desktop_vision import PhaseTemplate, Perceptor, Rect, VisionProfile, _white_text_distance, image_distance, read_number


def save_template(path, color, size=(4, 4)):
    from PIL import Image
    Image.new("RGB", size, (color, color, color)).save(path)


@pytest.fixture
def scene(tmp_path):
    save_template(tmp_path / "shop.png", 200)
    save_template(tmp_path / "battle.png", 100)
    save_template(tmp_path / "result.png", 50)
    save_template(tmp_path / "empty.png", 0)
    save_template(tmp_path / "ant.png", 150)
    save_template(tmp_path / "fish.png", 230)
    config = {
        "version": 1,
        "image_size": [40, 20],
        "calibrated": True,
        "phase_templates": [
            {"phase": phase.value, "region": [0, 0, 4, 4], "template": f"{phase.value}.png"}
            for phase in (Phase.SHOP, Phase.BATTLE, Phase.RESULT)
        ],
        "hud": {"gold": [5, 0, 1, 1], "turn": [6, 0, 1, 1], "wins": [7, 0, 1, 1], "lives": [8, 0, 1, 1]},
        "shop": [{"portrait": [0, 5, 4, 4], "attack": [5, 5, 1, 1], "health": [6, 5, 1, 1], "level": [7, 5, 1, 1], "empty_template": "empty.png", "species_templates": {"ant": "ant.png", "fish": "fish.png"}}],
        "team": [{"portrait": [10, 5, 4, 4], "attack": [15, 5, 1, 1], "health": [16, 5, 1, 1], "empty_template": "empty.png"}],
        "buttons": {"roll": [1, 15], "end_turn": [30, 15], "sell": [20, 15], "continue": [30, 10]},
    }
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    frame[:4, :4] = 200
    frame[5:9, :4] = 150
    for x, text_code in ((5, 10), (6, 1), (7, 0), (8, 5)):
        frame[0, x] = text_code
    frame[5, 5] = 2
    frame[5, 6] = 3
    frame[5, 7] = 1
    frame[5, 15:17] = 255  # unreadable stats on explicitly empty team slot
    ocr = lambda image: str(int(image[0, 0, 0])) if int(image[0, 0, 0]) != 255 else "?"
    return config, frame, ocr, tmp_path


def observe(scene, **kwargs):
    config, frame, ocr, directory = scene
    return Perceptor(VisionProfile.from_dict(config, base_dir=directory), ocr=kwargs.get("ocr", ocr)).observe(frame)


def test_reads_one_calibrated_shop_frame(scene):
    board = observe(scene)
    assert board.phase == Phase.SHOP
    assert (board.gold, board.turn, board.wins, board.lives) == (10, 1, 0, 5)
    assert (board.shop[0].occupied, board.shop[0].species, board.shop[0].attack, board.shop[0].health, board.shop[0].level) == (True, "ant", 2, 3, 1)
    assert board.team[0].occupied is False


@pytest.mark.parametrize("phase,color", [(Phase.BATTLE, 100), (Phase.RESULT, 50), (Phase.UNKNOWN, 255)])
def test_other_phases_never_read_shop_ocr(scene, phase, color):
    scene[1][:4, :4] = color
    def unexpected_ocr(image):
        pytest.fail("non-shop frames must not attempt shop OCR")
    board = observe(scene, ocr=unexpected_ocr)
    assert board.phase == phase
    assert board.gold is None and not board.shop and not board.team


def test_ambiguous_phase_returns_unknown(scene):
    config, frame, ocr, directory = scene
    config["phase_templates"][1]["template"] = "shop.png"
    assert observe(scene).phase == Phase.UNKNOWN


def test_multiple_references_of_same_phase_do_not_compete(scene):
    scene[0]["phase_templates"].append(copy.deepcopy(scene[0]["phase_templates"][0]))
    assert observe(scene).phase == Phase.SHOP


@pytest.fixture
def white_text_scene(tmp_path):
    from PIL import Image
    # A fixed PAUSE silhouette in a roomy crop; no OCR or real game pixels.
    glyphs = ("110 101 110 100 100", "010 101 111 101 101",
              "101 101 101 101 111", "111 100 111 001 111",
              "111 100 110 100 111")
    letters = [np.array([[pixel == "1" for pixel in row] for row in glyph.split()])
               for glyph in glyphs]
    word = np.concatenate([np.pad(letter, ((0, 0), (0, 1))) for letter in letters], axis=1)
    mask = np.zeros((20, 60), dtype=bool)
    mask[5:15, 8:48] = np.repeat(np.repeat(word, 2, axis=0), 2, axis=1)
    crop = np.full((20, 60, 3), 35, dtype=np.uint8)
    crop[mask] = 255
    Image.fromarray(crop).save(tmp_path / "pause.png")
    frame = np.zeros((30, 80, 3), dtype=np.uint8)
    frame[3:23, 2:62] = crop
    config = {"version": 1, "image_size": [80, 30], "calibrated": True,
              "phase_templates": [{"phase": "battle", "region": [2, 3, 60, 20],
                                   "template": "pause.png", "match_mode": "white_text"}]}
    return config, frame, mask, tmp_path


def observe_white_text(scene):
    def unexpected_ocr(crop):
        pytest.fail("phase text matching must not invoke OCR")
    return Perceptor(VisionProfile.from_dict(scene[0], base_dir=scene[3]), ocr=unexpected_ocr).observe(scene[1])


def assert_unknown_without_input(board):
    assert board.phase == Phase.UNKNOWN
    assert board.gold is None and not board.shop and not board.team
    assert DesktopPolicy().choose_action(board) is None
    assert not legal_action(board, Action("roll"))
    assert not legal_action(board, Action("end_turn"))


def test_white_text_survives_changing_background_and_moving_moon(white_text_scene):
    _, frame, mask, directory = white_text_scene
    crop = frame[3:23, 2:62]
    for moon_x in (0, 25, 45):
        crop[:] = (130, 160, 185)
        crop[:8, moon_x:moon_x + 10] = (200, 210, 215)
        crop[mask] = 255
        assert observe_white_text(white_text_scene).phase == Phase.BATTLE
    # The same calibrated RGB reference still rejects this background change.
    white_text_scene[0]["phase_templates"][0].pop("match_mode")
    assert_unknown_without_input(observe_white_text(white_text_scene))


@pytest.mark.parametrize("change", ["missing_letter", "extra_strokes", "different_word", "colored_text"])
def test_white_text_rejects_changed_or_missing_text(white_text_scene, change):
    crop = white_text_scene[1][3:23, 2:62]
    if change == "missing_letter":
        crop[:, 8:14] = 35
    elif change == "extra_strokes":
        crop[3:8, 50:58] = 255
    elif change == "different_word":
        crop[:] = 35
        crop[np.roll(white_text_scene[2], 6, axis=1)] = 255
    else:
        crop[white_text_scene[2]] = (255, 230, 200)
    assert_unknown_without_input(observe_white_text(white_text_scene))


@pytest.mark.parametrize("target", ["reference", "observation", "both"])
@pytest.mark.parametrize("ink", ["blank", "sparse", "filled"])
def test_white_text_invalid_ink_never_matches_even_at_max_threshold(white_text_scene, target, ink):
    from PIL import Image
    crop = np.full((20, 60, 3), 35, dtype=np.uint8)
    if ink == "sparse":
        crop[0, :7] = 255
    elif ink == "filled":
        crop[:] = 255
    if target in {"reference", "both"}:
        Image.fromarray(crop).save(white_text_scene[3] / "pause.png")
    if target in {"observation", "both"}:
        white_text_scene[1][3:23, 2:62] = crop
    white_text_scene[0]["phase_templates"][0]["max_distance"] = 1
    assert_unknown_without_input(observe_white_text(white_text_scene))


@pytest.mark.parametrize("option,value", [("min_ink_pixels", 1000), ("min_ink_fraction", 0.4), ("max_ink_fraction", 0.02)])
def test_white_text_ink_limits_use_calibrated_options(white_text_scene, option, value):
    template = white_text_scene[0]["phase_templates"][0]
    template["white_text"] = {option: value}
    assert_unknown_without_input(observe_white_text(white_text_scene))


def test_white_text_channel_threshold_and_spread_use_calibrated_options(white_text_scene):
    from PIL import Image
    crop = white_text_scene[1][3:23, 2:62]
    crop[white_text_scene[2]] = (215, 225, 230)
    Image.fromarray(crop).save(white_text_scene[3] / "pause.png")
    template = white_text_scene[0]["phase_templates"][0]
    assert_unknown_without_input(observe_white_text(white_text_scene))
    template["white_text"] = {"min_channel": 210, "max_channel_spread": 15}
    assert observe_white_text(white_text_scene).phase == Phase.BATTLE
    template["white_text"]["max_channel_spread"] = 14
    assert_unknown_without_input(observe_white_text(white_text_scene))


def test_white_text_error_is_symmetric_and_normalized_by_ink_union():
    left = np.zeros((10, 10, 3), dtype=np.uint8)
    right = left.copy()
    left[0, :] = 255
    right[0, :5] = right[1, :5] = 255
    options = PhaseTemplate(Phase.BATTLE, Rect(0, 0, 10, 10), "pause.png", match_mode="white_text").matching_options()
    assert _white_text_distance(left, right, options) == pytest.approx(2 / 3)
    assert _white_text_distance(right, left, options) == pytest.approx(2 / 3)
    assert _white_text_distance(left, left, options) == 0


@pytest.mark.parametrize("same_phase", [False, True])
def test_white_text_runner_up_margin_and_same_phase_variants(white_text_scene, same_phase):
    from PIL import Image
    reference = white_text_scene[1][3:23, 2:62].copy()
    y, x = np.argwhere(white_text_scene[2])[0]
    reference[y, x] = 35  # Almost identical, below the configured margin.
    Image.fromarray(reference).save(white_text_scene[3] / "variant.png")
    variant = {**white_text_scene[0]["phase_templates"][0], "template": "variant.png",
               "phase": "battle" if same_phase else "result"}
    white_text_scene[0]["phase_templates"].append(variant)
    board = observe_white_text(white_text_scene)
    if same_phase:
        assert board.phase == Phase.BATTLE
    else:
        assert_unknown_without_input(board)


@pytest.mark.parametrize("evidence", ["abstain", "agree", "conflict", "ambiguous", "eligible_runner"])
def test_mixed_phase_modes_require_all_eligible_evidence_to_agree(white_text_scene, evidence):
    from PIL import Image
    config, frame, mask, directory = white_text_scene
    crop = frame[3:23, 2:62]
    crop[~mask] = 150  # RGB pause.png cannot match, white mask still can.
    rgb_reference = crop.copy()
    if evidence == "abstain":
        rgb_reference[~mask] = 35
    elif evidence == "eligible_runner":
        rgb_reference[~mask] = 170
    Image.fromarray(rgb_reference).save(directory / "rgb.png")
    rgb = {"phase": "battle" if evidence == "agree" else "result", "region": [2, 3, 60, 20],
           "template": "rgb.png"}
    config["phase_templates"].append(rgb)
    if evidence == "ambiguous":
        config["phase_templates"].append({**rgb, "phase": "battle"})
    elif evidence == "eligible_runner":
        # Lowest RGB error fails its stricter threshold; a slightly worse
        # rival passes its own threshold. That group must veto a white winner.
        rgb["max_distance"] = 0.01
        rgb_reference[~mask] = 172
        Image.fromarray(rgb_reference).save(directory / "runner.png")
        config["phase_templates"].append({**rgb, "phase": "shop", "template": "runner.png", "max_distance": 0.1})
    board = observe_white_text(white_text_scene)
    if evidence in {"abstain", "agree"}:
        assert board.phase == Phase.BATTLE
    else:
        assert_unknown_without_input(board)


@pytest.mark.parametrize("fields", [
    {"match_mode": "ocr"}, {"match_mode": True}, {"match_mode": None},
    {"match_mode": ["white_text"]}, {"white_text": {}}, {"unexpected": 1},
    {"white_text": []}, {"white_text": {"unknown": 1}},
    {"white_text": {"min_channel": True}}, {"white_text": {"min_channel": 256}},
    {"white_text": {"max_channel_spread": -1}}, {"white_text": {"max_channel_spread": 2.5}},
    {"white_text": {"min_ink_pixels": 0}}, {"white_text": {"min_ink_pixels": True}},
    {"white_text": {"min_ink_fraction": 0}}, {"white_text": {"min_ink_fraction": float("nan")}},
    {"white_text": {"max_ink_fraction": 1}}, {"white_text": {"max_ink_fraction": float("inf")}},
    {"white_text": {"min_ink_fraction": 0.6, "max_ink_fraction": 0.6}},
])
def test_invalid_phase_mode_or_white_text_options_are_rejected(white_text_scene, fields):
    template = white_text_scene[0]["phase_templates"][0]
    if fields == {"white_text": {}}:
        template["match_mode"] = "rgb"
    template.update(fields)
    with pytest.raises(ValueError):
        VisionProfile.from_dict(white_text_scene[0], base_dir=white_text_scene[3])


def test_phase_modes_roundtrip_and_legacy_v1_remains_rgb(scene, white_text_scene):
    legacy = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    assert all(item.match_mode == "rgb" and item.white_text is None for item in legacy.phase_templates)
    assert all("match_mode" not in item and "white_text" not in item for item in legacy.to_dict()["phase_templates"])
    template = white_text_scene[0]["phase_templates"][0]
    template["white_text"] = {"min_channel": 240, "max_channel_spread": 12, "min_ink_pixels": 20,
                              "min_ink_fraction": 0.03, "max_ink_fraction": 0.5}
    profile = VisionProfile.from_dict(white_text_scene[0], base_dir=white_text_scene[3])
    path = white_text_scene[3] / "white-text.json"
    profile.save(path)
    loaded = VisionProfile.load(path)
    assert loaded == profile
    assert loaded.to_dict()["version"] == 1
    assert loaded.to_dict()["phase_templates"][0]["white_text"] == template["white_text"]
    assert Perceptor(loaded).observe(white_text_scene[1]).phase == Phase.BATTLE


def test_no_phase_references_cannot_authorize_input(scene):
    scene[0]["phase_templates"] = []
    assert_unknown_without_input(observe(scene))


@pytest.mark.parametrize("attack,health", [(255, 3), (2, 255), (255, 255), (2, 0), (100, 3)])
def test_unreadable_or_invalid_stats_are_unknown_not_pets(scene, attack, health):
    scene[1][5, 5] = attack
    scene[1][5, 6] = health
    assert observe(scene).shop[0].occupied is None


def test_unknown_species_does_not_erase_verified_stats(scene):
    scene[1][5:9, :4] = 80
    slot = observe(scene).shop[0]
    assert slot.occupied is True and slot.attack == 2 and slot.species is None


def test_ambiguous_species_does_not_invent_merge_identity(scene):
    config = scene[0]
    config["shop"][0]["species_templates"]["fish"] = "ant.png"
    slot = observe(scene).shop[0]
    assert slot.occupied is True and slot.species is None


def test_empty_portrait_with_stats_is_contradictory(scene):
    scene[1][5:9, :4] = 0
    assert observe(scene).shop[0].occupied is None


@pytest.mark.parametrize("species_color,frame_color", [(2, 2), (2, 0), (0, 0), (10, 10)])
def test_empty_requires_margin_over_species_even_when_ocr_is_blank(scene, species_color, frame_color):
    # Both references match the threshold: species may tie, nearly tie, or win.
    save_template(scene[3] / "ant.png", species_color)
    scene[1][5:9, :4] = frame_color
    scene[1][5, 5:8] = 255
    slot = observe(scene).shop[0]
    assert slot.occupied is None
    assert slot.species is None


def test_species_pixels_and_missing_ocr_never_authorize_an_empty_target(scene):
    save_template(scene[3] / "ant.png", 2)
    scene[1][5:9, :4] = 2
    assert observe(scene, ocr=None).shop[0].occupied is None


def test_empty_can_win_by_a_clear_margin_with_blank_ocr(scene):
    save_template(scene[3] / "ant.png", 10)
    scene[1][5:9, :4] = 0
    scene[1][5, 5:8] = 255
    assert observe(scene).shop[0].occupied is False


def test_empty_requires_visual_evidence(scene):
    scene[0]["team"][0].pop("empty_template")
    assert observe(scene).team[0].occupied is None


def test_ocr_absence_or_exception_preserves_unknown(scene):
    board = observe(scene, ocr=None)
    assert board.gold is None and board.shop[0].occupied is None
    assert board.team[0].occupied is False
    def broken_ocr(image):
        raise RuntimeError("OCR subprocess unavailable")
    assert observe(scene, ocr=broken_ocr).gold is None


@pytest.mark.parametrize("text", ["", "?", "3/5", "1O", "2 3", "-1", "+1", "1.0", "100", "١", None, 3])
def test_strict_numbers(text):
    assert read_number(text, 0, 99) is None


def test_numbers_preserve_real_zero_and_whitespace():
    assert read_number(" 0\n", 0, 99) == 0
    assert read_number("0", 1, 99) is None
    assert read_number("50", 0, 99) == 50


@pytest.mark.parametrize("frame", [np.zeros((10, 20, 3), dtype=np.uint8), np.zeros((20, 40, 4), dtype=np.uint8), np.zeros((20, 40, 3), dtype=np.float32)])
def test_wrong_size_format_or_dtype_rejected_before_ocr(scene, frame):
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    with pytest.raises(ValueError, match="RGB uint8"):
        Perceptor(profile).observe(frame)


def test_wrong_template_size_rejected(scene):
    save_template(scene[3] / "shop.png", 200, size=(2, 2))
    with pytest.raises(ValueError, match="dimensions"):
        observe(scene)


def test_missing_template_rejected(scene):
    scene[0]["phase_templates"][0]["template"] = "missing.png"
    with pytest.raises(FileNotFoundError):
        observe(scene)


def test_profile_roundtrip_relative_templates(scene):
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    path = scene[3] / "profile.json"
    profile.save(path)
    loaded = VisionProfile.load(path)
    assert loaded == profile
    assert loaded.buttons["roll"] == (1, 15)
    assert loaded.shop[0].portrait.center == (2, 7)
    assert Perceptor(loaded, ocr=scene[2]).observe(scene[1]).gold == 10


@pytest.mark.parametrize("key,value", [("version", 2), ("version", True), ("image_size", [40.0, 20]), ("image_size", [0, 20]), ("calibrated", "true"), ("unexpected", 1)])
def test_invalid_profile_rejected(scene, key, value):
    scene[0][key] = value
    with pytest.raises(ValueError):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("rect", [[-1, 0, 4, 4], [39, 0, 4, 4], [0, 0, 0, 4], [0, 0, 4.0, 4], [0, 0, 4]])
def test_invalid_regions_rejected(scene, rect):
    scene[0]["hud"]["gold"] = rect
    with pytest.raises(ValueError):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("name", ["../private.png", "/tmp/private.png", "", None])
def test_template_paths_cannot_escape_profile_directory(scene, name):
    scene[0]["phase_templates"][0]["template"] = name
    with pytest.raises(ValueError, match="template paths"):
        VisionProfile.from_dict(scene[0], base_dir=scene[3])


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_matching_thresholds_rejected(scene, threshold):
    scene[0]["phase_templates"][0]["max_distance"] = threshold
    with pytest.raises(ValueError, match="max_distance"):
        VisionProfile.from_dict(scene[0])


def test_uncalibrated_profile_can_be_inspected_but_not_enabled(scene):
    scene[0]["calibrated"] = False
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    with pytest.raises(ValueError, match="explicitly calibrated"):
        profile.validate(require_calibrated=True)
    assert Perceptor(profile).observe(scene[1]).phase == Phase.SHOP


def test_distance_uses_signed_arithmetic():
    dark = np.zeros((2, 2, 3), dtype=np.uint8)
    light = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert image_distance(dark, light) == image_distance(light, dark) == 1
    assert image_distance(dark, dark) == 0


def test_crop_does_not_allow_negative_or_overflow_geometry():
    with pytest.raises(ValueError):
        Rect(-1, 0, 1, 1).crop(np.zeros((2, 2, 3), dtype=np.uint8))


def add_number_reference(scene, *, field="gold", value=10, color=10, name="number.png", **thresholds):
    save_template(scene[3] / name, color, size=(1, 1))
    item = {"field": field, "value": value, "template": name, **thresholds}
    scene[0].setdefault("numeric_templates", []).append(item)
    return item


def test_calibrated_numeric_reference_resolves_blank_ocr(scene):
    add_number_reference(scene)
    assert observe(scene, ocr=lambda crop: "?").gold == 10
    assert observe(scene, ocr=None).gold == 10


def test_reference_can_confirm_a_real_zero_but_never_defaults_to_zero(scene):
    scene[1][0, 5] = 0
    add_number_reference(scene, value=0, color=0)
    assert observe(scene, ocr=None).gold == 0
    scene[1][0, 5] = 255
    assert observe(scene, ocr=None).gold is None


def test_conflicting_ocr_and_numeric_reference_stay_unknown(scene):
    add_number_reference(scene)
    assert observe(scene, ocr=lambda crop: "7").gold is None


def test_ambiguous_number_references_cannot_be_resolved_by_ocr(scene):
    add_number_reference(scene)
    add_number_reference(scene, value=7, name="seven.png")
    assert observe(scene).gold is None


def test_numeric_margin_prevents_near_tie_even_when_ocr_agrees(scene):
    add_number_reference(scene)
    add_number_reference(scene, value=7, color=11, name="seven.png")
    assert observe(scene).gold is None


def test_same_value_reference_variants_do_not_compete(scene):
    add_number_reference(scene)
    add_number_reference(scene, color=11, name="same-number-variant.png")
    assert observe(scene, ocr=None).gold == 10


def test_unmatched_numeric_references_allow_ordinary_ocr(scene):
    add_number_reference(scene, value=7, color=200)
    assert observe(scene).gold == 10


def test_numeric_references_do_not_override_ocr_runtime_failure(scene):
    add_number_reference(scene)
    def broken(crop):
        raise RuntimeError("OCR unavailable")
    assert observe(scene, ocr=broken).gold is None


def test_numeric_references_do_not_hide_ocr_timeout(scene):
    add_number_reference(scene)
    def timeout(crop):
        raise TimeoutError("OCR timeout")
    with pytest.raises(TimeoutError):
        observe(scene, ocr=timeout)


def test_level_reference_attaches_only_to_its_calibrated_slot(scene):
    add_number_reference(scene, field="shop.0.level", value=1, color=1)
    ocr = lambda crop: "?" if int(crop[0, 0, 0]) == 1 else scene[2](crop)
    board = observe(scene, ocr=ocr)
    assert board.shop[0].level == 1
    assert board.team[0].level is None


@pytest.mark.parametrize("field,value", [("bogus", 1), ("team.4.level", 1), ("team.0.level", 1), ("shop.0.level", 0), ("shop.0.level", 4), ("gold", True), ("gold", "10"), ("gold", 100), ("turn", 0), ("shop.0.health", 0)])
def test_invalid_numeric_reference_field_or_range_is_rejected(scene, field, value):
    add_number_reference(scene, field=field, value=value)
    with pytest.raises(ValueError):
        VisionProfile.from_dict(scene[0], base_dir=scene[3])


def test_numeric_templates_roundtrip_and_stay_relative(scene):
    add_number_reference(scene)
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    path = scene[3] / "with-numbers.json"
    profile.save(path)
    loaded = VisionProfile.load(path)
    assert loaded == profile
    assert Perceptor(loaded).observe(scene[1]).gold == 10
    scene[0]["numeric_templates"][0]["template"] = "../outside.png"
    with pytest.raises(ValueError, match="template paths"):
        VisionProfile.from_dict(scene[0], base_dir=scene[3])


def test_missing_numeric_template_is_configuration_error(scene):
    item = add_number_reference(scene)
    item["template"] = "missing-number.png"
    with pytest.raises(FileNotFoundError):
        observe(scene)


def test_numeric_template_must_have_exact_crop_dimensions(scene):
    add_number_reference(scene)
    save_template(scene[3] / "number.png", 10, size=(2, 2))
    with pytest.raises(ValueError, match="dimensions"):
        observe(scene)


@pytest.mark.parametrize("phase", [phase for phase in Phase if phase != Phase.UNKNOWN])
def test_explicit_phase_templates_support_all_known_phases(scene, phase):
    scene[0]["phase_templates"] = [{"phase": phase.value, "region": [0, 0, 4, 4], "template": "shop.png"}]
    assert observe(scene).phase == phase


def test_transition_buttons_are_explicit_optional_calibration(scene):
    scene[0]["buttons"].update({name: [3, 4] for name in ("name_adjective", "name_noun", "confirm_name", "continue_round", "dismiss_tier", "confirm_end_turn")})
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    assert profile.buttons["dismiss_tier"] == (3, 4)
    assert profile.buttons["confirm_end_turn"] == (3, 4)


def gated_shop(scene, turns=(1, 1, 1, 5, 9)):
    template = scene[0]["shop"][0]
    scene[0]["shop"] = [{**copy.deepcopy(template), "available_from_turn": turn} for turn in turns]


@pytest.mark.parametrize("turn,count", [(1, 3), (4, 3), (5, 4), (8, 4), (9, 5), (12, 5)])
def test_shop_geometry_uses_calibrated_turn_boundaries(scene, turn, count):
    gated_shop(scene)
    scene[1][0, 6] = turn
    board = observe(scene)
    assert board.turn == turn and len(board.shop) == count
    assert all(slot.occupied is True and slot.attack == 2 and slot.health == 3 for slot in board.shop)


def test_turn_gating_uses_profile_values_without_builtin_game_schedule(scene):
    gated_shop(scene, (1, 2, 7))
    scene[1][0, 6] = 2
    assert len(observe(scene).shop) == 2


def test_unknown_turn_with_gated_slots_blocks_actions_and_skips_shop_readings(scene, monkeypatch):
    gated_shop(scene)
    scene[1][0, 6] = 255
    scene[0]["team"] *= 5
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    perceptor = Perceptor(profile, ocr=scene[2])
    original = perceptor._slot
    prefixes = []

    def record(frame, slot, field_prefix=None):
        prefixes.append(field_prefix)
        return original(frame, slot, field_prefix)

    monkeypatch.setattr(perceptor, "_slot", record)
    board = perceptor.observe(scene[1])
    assert board.turn is None and len(board.shop) == 5
    assert all(slot.occupied is None for slot in board.shop)
    assert prefixes == [f"team.{i}" for i in range(5)]
    assert DesktopPolicy().choose_action(board) is None
    assert not legal_action(board, Action("roll"))
    assert not legal_action(board, Action("end_turn"))


def test_inactive_slot_references_are_not_loaded(scene):
    gated_shop(scene)
    scene[0]["shop"][3]["species_templates"] = {"future": "not-yet-captured.png"}
    scene[1][0, 6] = 4
    assert len(observe(scene).shop) == 3


@pytest.mark.parametrize("turn", [0, -1, True, 1.5, "5", None])
def test_invalid_available_turn_is_rejected(scene, turn):
    scene[0]["shop"][0]["available_from_turn"] = turn
    with pytest.raises(ValueError, match="available_from_turn"):
        VisionProfile.from_dict(scene[0])


def test_active_shop_must_be_a_prefix_and_team_slots_are_always_available(scene):
    gated_shop(scene, (1, 5, 1))
    with pytest.raises(ValueError, match="nondecreasing"):
        VisionProfile.from_dict(scene[0])
    gated_shop(scene, (1, 5, 9))
    scene[0]["team"][0]["available_from_turn"] = 2
    with pytest.raises(ValueError, match="team slots"):
        VisionProfile.from_dict(scene[0])


@pytest.mark.parametrize("row", ["shop", "team"])
def test_profile_rejects_more_slots_than_board_can_represent(scene, row):
    scene[0][row] *= 6
    with pytest.raises(ValueError, match="at most five"):
        VisionProfile.from_dict(scene[0])


def test_gated_slots_roundtrip_and_legacy_slots_default_to_turn_one(scene):
    legacy = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    assert legacy.shop[0].available_from_turn == legacy.team[0].available_from_turn == 1
    scene[1][0, 6] = 255
    assert observe(scene).shop[0].occupied is True  # Legacy unknown-turn behavior.
    gated_shop(scene)
    profile = VisionProfile.from_dict(scene[0], base_dir=scene[3])
    path = scene[3] / "gated.json"
    profile.save(path)
    restored = VisionProfile.load(path)
    assert restored == profile
    assert [slot.available_from_turn for slot in restored.shop] == [1, 1, 1, 5, 9]
