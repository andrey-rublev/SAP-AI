"""Uniform profile scaling and display-mode choice; no desktop access."""
import json

import numpy as np
import pytest
from PIL import Image

from sap_ai.desktop.display import choose_client_size
from sap_ai.desktop.scaling import fit_size, scale_factor, scale_profile, scale_tooltip_layout
from sap_ai.desktop.vision import Perceptor, VisionProfile


def write_profile(tmp_path):
    templates = tmp_path / "t"
    templates.mkdir()
    menu = np.zeros((40, 80, 3), dtype=np.uint8)
    menu[10:30, 20:60] = (200, 120, 40)
    Image.fromarray(menu).save(templates / "menu.png")
    Image.fromarray(np.full((20, 40, 3), 90, dtype=np.uint8)).save(templates / "empty.png")
    Image.fromarray(np.full((20, 20, 3), 255, dtype=np.uint8)).save(templates / "gold-5.png")
    data = {"version": 1, "image_size": [400, 200], "calibrated": True,
            "phase_templates": [{"phase": "main_menu", "region": [100, 40, 80, 40], "template": "t/menu.png",
                                 "max_distance": .02, "margin": .01}],
            "hud": {"gold": [10, 10, 20, 20]},
            "shop": [{"portrait": [200, 120, 40, 20], "empty_template": "t/empty.png",
                      "species_templates": {"fish": ["t/empty.png"]}}],
            "team": [{"portrait": [200, 160, 40, 20]}] * 5,
            "buttons": {"open_play": [140, 60]}, "pointer_rest": [399, 199],
            "numeric_templates": [{"field": "gold", "value": 5, "template": "t/gold-5.png",
                                   "max_distance": .01, "margin": .01}]}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, menu


def test_scaled_profile_maps_regions_points_and_references(tmp_path):
    path, menu = write_profile(tmp_path)
    scaled = scale_profile(path, (300, 150), tmp_path / "cache")
    assert scaled.image_size == (300, 150) and scaled.buttons["open_play"] == (105, 45)
    assert scaled.phase_templates[0].region.to_list() == [75, 30, 60, 30]
    assert scaled.hud["gold"].to_list() == [8, 8, 14, 14] and scaled.pointer_rest == (299, 149)
    # A frame rendered at the new size is recognized with the resized reference.
    frame = np.zeros((150, 300, 3), dtype=np.uint8)
    frame[30:60, 75:135] = np.asarray(Image.fromarray(menu).resize((60, 30), Image.Resampling.LANCZOS))
    assert Perceptor(scaled).observe(frame).phase.value == "main_menu"
    assert scale_profile(path, (400, 200), tmp_path / "cache") == VisionProfile.load(path)


def test_different_aspect_ratios_are_refused(tmp_path):
    path, _ = write_profile(tmp_path)
    with pytest.raises(ValueError, match="aspect"):
        scale_profile(path, (300, 200), tmp_path / "cache")
    assert scale_factor((2560, 1440), (1920, 1080)) == pytest.approx(.75)


def test_display_choice_keeps_native_fullscreen_or_fits_a_window():
    assert choose_client_size((2560, 1440), (2560, 1440), (2560, 1392)) == ((2560, 1440), False)
    assert choose_client_size((2560, 1440), (1920, 1200), (1920, 1200)) == ((1904, 1071), True)
    assert choose_client_size((2560, 1440), (3840, 2160), (3840, 2112)) == ((2560, 1440), True)
    assert fit_size((2560, 1440), (1280, 1000)) == (1280, 720)


def test_tooltip_layout_scales_with_the_profile():
    data = {"version": 1, "calibrated": True, "image_size": [2560, 1440],
            "layout": {"search_rect": [0, 30, 2560, 730], "panel_width_range": [600, 700],
                       "panel_height_range": [180, 450], "title_offset_rect": [120, 12, 420, 70]}}
    scaled = scale_tooltip_layout(data, .75)
    assert scaled["image_size"] == [1920, 1080]
    assert scaled["layout"]["search_rect"] == [0, 22, 1920, 548]
    assert scaled["layout"]["panel_width_range"] == [450, 526]


def test_phase_sign_matches_on_an_unseen_background(tmp_path):
    """Two references of one sign on different backgrounds match a third background."""
    templates = tmp_path / "t"
    templates.mkdir()
    sign = np.zeros((20, 40, 3), dtype=np.uint8)
    sign[2:18, 2:38] = (230, 160, 60)
    sign[8:12, 10:30] = (90, 40, 20)
    def on(background):
        image = np.full((20, 40, 3), background, dtype=np.uint8)
        image[2:18, 2:38] = sign[2:18, 2:38]
        return image
    Image.fromarray(on((40, 160, 40))).save(templates / "day.png")
    Image.fromarray(on((240, 230, 200))).save(templates / "cream.png")
    data = {"version": 1, "image_size": [100, 60], "calibrated": True,
            "phase_templates": [{"phase": "shop", "region": [10, 10, 40, 20], "template": f"t/{name}.png",
                                 "max_distance": .045, "margin": .015} for name in ("day", "cream")],
            "team": [{"portrait": [60, 40, 10, 10]}] * 5}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    perceptor = Perceptor(VisionProfile.load(path))
    frame = np.zeros((60, 100, 3), dtype=np.uint8)
    frame[10:30, 10:50] = on((200, 90, 140))  # evening background never calibrated
    assert perceptor._phase(frame).value == "shop"
    frame[10:30, 10:50] = 0  # no sign at all
    assert perceptor._phase(frame).value == "unknown"
