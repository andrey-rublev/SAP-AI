"""Adversarial synthetic panel inventories; no file or desktop IO."""
from dataclasses import replace

import numpy as np
import pytest

import desktop_tooltip_layout as detector
from desktop_tooltip_layout import TooltipLayout, inspect_tooltip_panel, inventory_tooltip_panels


def scene():
    image = np.full((150, 340, 3), 90, dtype=np.uint8)
    layout = TooltipLayout((10, 10, 310, 130), (20, 32), (38, 42), (3, 3, 24, 10),
                           dark_perimeter_max_channel=60)
    return image, layout


def panel(image, rect, *, title=True):
    x, y, width, height = rect
    image[y - 1:y + height + 1, x - 1:x + width + 1] = 0
    image[y:y + height, x:x + width] = 247
    if title:
        image[y + 5:y + 9, x + 5:x + 15] = (190, 100, 15)


def test_inventory_retains_every_candidate_instead_of_stopping_at_ambiguity():
    image, layout = scene()
    bounds = ((20, 20, 30, 40), (80, 20, 30, 40), (140, 20, 30, 40))
    for rect in bounds:
        panel(image, rect)
    inventory = inventory_tooltip_panels(image, layout)
    assert {item.panel_rect for item in inventory.candidates} == set(bounds)
    assert inventory.boundary_rects == inventory.unresolved_reasons == ()
    strict = inspect_tooltip_panel(image, layout)
    assert (strict.status, strict.title, strict.reason) == ("ambiguous", None, None)


def test_inventory_retains_all_unresolved_reason_kinds_and_still_returns_a_valid_candidate():
    image, layout = scene()
    valid, bad_edge, bad_perimeter, bad_title = ((20, 20, 30, 40), (80, 20, 30, 40),
                                               (140, 20, 30, 40), (200, 20, 20, 40))
    for rect in (valid, bad_edge, bad_perimeter, bad_title):
        panel(image, rect)
    image[20, 80:95] = 0  # Dense panel, but only half of its upper edge survives.
    image[19, 140:170] = 90  # Valid white panel with an uncalibrated exterior.
    image[100, 10] = 255  # Unrelated tiny boundary evidence is retained too.
    inventory = inventory_tooltip_panels(image, layout)
    assert [item.panel_rect for item in inventory.candidates] == [valid]
    assert inventory.boundary_rects == ((10, 100, 1, 1),)
    assert set(inventory.unresolved_reasons) == {"panel_edges", "panel_perimeter", "title_geometry", "search_boundary"}
    assert len(inventory.unresolved_reasons) == 4
    strict = inspect_tooltip_panel(image, layout)
    assert (strict.status, strict.title, strict.reason) == ("unresolved", None, "search_boundary")


def test_strict_ambiguity_keeps_priority_without_hiding_inventory_boundary_evidence():
    image, layout = scene()
    panel(image, (20, 20, 30, 40))
    panel(image, (80, 20, 30, 40))
    image[100, 10] = 255
    inventory = inventory_tooltip_panels(image, layout)
    assert len(inventory.candidates) == 2
    assert inventory.boundary_rects == ((10, 100, 1, 1),)
    assert inventory.unresolved_reasons == ("search_boundary",)
    strict = inspect_tooltip_panel(image, layout)
    assert (strict.status, strict.title, strict.reason) == ("ambiguous", None, None)


def test_every_boundary_component_uses_absolute_bounds_even_below_panel_size_limits():
    image, layout = scene()
    image[20, 10] = image[20, 319] = image[10, 30] = image[139, 30] = 255
    inventory = inventory_tooltip_panels(image, layout)
    assert inventory.candidates == ()
    assert inventory.boundary_rects == ((10, 20, 1, 1), (30, 10, 1, 1), (30, 139, 1, 1), (319, 20, 1, 1))
    assert inventory.unresolved_reasons == ("search_boundary",)
    assert inspect_tooltip_panel(image, layout).status == "unresolved"


def test_diagonal_interior_pixel_does_not_enlarge_a_four_connected_boundary_component():
    image, layout = scene()
    image[10, 10] = image[11, 11] = 255
    assert inventory_tooltip_panels(image, layout).boundary_rects == ((10, 10, 1, 1),)


def test_candidate_title_crops_are_detached_from_input_and_each_other():
    image, layout = scene()
    for rect in ((20, 20, 30, 40), (80, 20, 30, 40)):
        panel(image, rect)
    inventory = inventory_tooltip_panels(image, layout)
    original = [item.title_crop.copy() for item in inventory.candidates]
    image[:] = 0
    assert all(np.array_equal(item.title_crop, expected) for item, expected in zip(inventory.candidates, original))
    inventory.candidates[0].title_crop[:] = 0
    assert np.array_equal(inventory.candidates[1].title_crop, original[1])


def test_work_limit_discards_partial_candidates_instead_of_proving_clearance(monkeypatch):
    image, layout = scene()
    panel(image, (20, 20, 30, 40), title=False)
    panel(image, (80, 75, 30, 40), title=False)
    monkeypatch.setattr(detector, "MAX_WHITE_RUNS", 40)
    inventory = inventory_tooltip_panels(image, layout)
    assert inventory.candidates == inventory.boundary_rects == ()
    assert inventory.unresolved_reasons == ("fragmentation_limit",)
    strict = inspect_tooltip_panel(image, layout)
    assert (strict.status, strict.title, strict.reason) == ("unresolved", None, "fragmentation_limit")


def test_exact_component_work_budget_is_allowed_but_one_less_refuses(monkeypatch):
    image, layout = scene()
    panel(image, (20, 20, 30, 40), title=False)
    monkeypatch.setattr(detector, "MAX_WHITE_RUNS", 40)
    assert len(inventory_tooltip_panels(image, layout).candidates) == 1
    monkeypatch.setattr(detector, "MAX_WHITE_RUNS", 39)
    assert inventory_tooltip_panels(image, layout).unresolved_reasons == ("fragmentation_limit",)


def test_inventory_uses_same_color_and_geometry_gates_as_strict_inspection():
    image, layout = scene()
    panel(image, (20, 20, 30, 40))
    image[20:60, 20:50] = (250, 10, 10)
    assert inventory_tooltip_panels(image, layout).candidates == ()
    assert inspect_tooltip_panel(image, layout).status == "absent"
    panel(image, (20, 20, 30, 40))
    inventory = inventory_tooltip_panels(image, layout)
    strict = inspect_tooltip_panel(image, layout)
    assert strict.status == "unique"
    assert strict.title.panel_rect == inventory.candidates[0].panel_rect
    invalid_title = replace(layout, title_offset_rect=(3, 3, 35, 10))
    assert inventory_tooltip_panels(image, invalid_title).unresolved_reasons == ("title_geometry",)


@pytest.mark.parametrize("bad", [np.zeros((10, 10), dtype=np.uint8), np.zeros((150, 340, 4), dtype=np.uint8),
                                  np.zeros((150, 340, 3), dtype=float)])
def test_malformed_inventory_inputs_remain_explicit_errors(bad):
    _, layout = scene()
    with pytest.raises(ValueError, match="uint8 RGB"):
        inventory_tooltip_panels(bad, layout)
