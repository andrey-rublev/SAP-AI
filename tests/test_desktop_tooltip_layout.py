"""Calibrated synthetic white-panel scenes; no game images or screen access."""
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

import desktop_tooltip_layout as detector
from desktop_tooltip_layout import TooltipLayout, find_tooltip_title, inspect_tooltip_panel


def panel(image, rect, text="Fish"):
    x, y, width, height = rect
    image[y - 1:y + height + 1, x - 1:x + width + 1] = 0
    image[y:y + height, x:x + width] = (247, 247, 245)
    title = Image.fromarray(image)
    ImageDraw.Draw(title).text((x + 12, y + 8), text, fill=(190, 100, 15), font=ImageFont.load_default())
    image[:] = np.asarray(title)


@pytest.fixture(params=[((180, 240), (58, 42, 100, 85), (20, 15, 190, 150), (8, 7, 65, 15)),
                        ((260, 340), (95, 96, 140, 115), (60, 55, 240, 180), (12, 7, 95, 18))])
def scene(request):
    shape, bounds, search, title = request.param
    image = np.full((*shape, 3), (90, 110, 130), dtype=np.uint8)
    panel(image, bounds)
    layout = TooltipLayout(search, (bounds[2] - 5, bounds[2] + 5), (bounds[3] - 5, bounds[3] + 5),
                           title, dark_perimeter_max_channel=60)
    return image, bounds, layout


def test_two_calibrated_layouts_return_exact_bounds_and_a_detached_title_crop(scene):
    image, bounds, layout = scene
    original = image.copy()
    found = find_tooltip_title(image, layout)
    assert found is not None and found.panel_rect == bounds
    tx, ty, tw, th = layout.title_offset_rect
    x, y = bounds[0] + tx, bounds[1] + ty
    assert found.title_rect == (x, y, tw, th)
    assert np.array_equal(found.title_crop, image[y:y + th, x:x + tw])
    assert .85 < found.white_fraction < 1
    assert found.white_edge_fraction == found.dark_perimeter_fraction == 1
    found.title_crop[:] = 0
    assert np.array_equal(image, original)


def test_blank_or_white_hud_of_wrong_dimensions_is_not_a_panel(scene):
    image, _, layout = scene
    image[:] = (90, 110, 130)
    assert find_tooltip_title(image, layout) is None
    x, y, _, _ = layout.search_rect
    panel(image, (x + 8, y + 8, 35, 25))
    assert find_tooltip_title(image, layout) is None


def test_sparse_white_outline_is_not_a_dense_panel(scene):
    image, (x, y, width, height), layout = scene
    image[y:y + height, x:x + width] = 0
    image[y, x:x + width] = 255
    image[y + height - 1, x:x + width] = 255
    image[y:y + height, x] = 255
    image[y:y + height, x + width - 1] = 255
    assert find_tooltip_title(image, layout) is None


def test_dense_fill_alone_does_not_replace_four_white_edges(scene):
    image, (x, y, width, _), layout = scene
    image[y, x:x + width // 2] = 0
    assert find_tooltip_title(image, layout) is None


def test_optional_dark_perimeter_is_enforced_only_when_calibrated(scene):
    image, (x, y, width, _), layout = scene
    image[y - 1, x:x + width] = (140, 140, 140)
    assert find_tooltip_title(image, layout) is None
    found = find_tooltip_title(image, replace(layout, dark_perimeter_max_channel=None))
    assert found is not None and found.dark_perimeter_fraction is None


def test_chromatic_bright_rectangle_is_not_near_white(scene):
    image, (x, y, width, height), layout = scene
    image[y:y + height, x:x + width] = (255, 245, 120)
    assert find_tooltip_title(image, layout) is None


def test_multiple_eligible_panels_are_ambiguous():
    image = np.full((150, 300, 3), (90, 110, 130), dtype=np.uint8)
    panel(image, (20, 30, 95, 80))
    panel(image, (170, 30, 95, 80), "Ant")
    layout = TooltipLayout((0, 0, 300, 150), (90, 100), (75, 85), (8, 7, 65, 15),
                           dark_perimeter_max_channel=60)
    assert find_tooltip_title(image, layout) is None
    inspected = inspect_tooltip_panel(image, layout)
    assert inspected.status == "ambiguous" and inspected.title is None


def test_clear_and_unique_evidence_have_distinct_inspection_results(scene):
    image, bounds, layout = scene
    inspected = inspect_tooltip_panel(image, layout)
    assert inspected.status == "unique" and inspected.reason is None
    assert inspected.title.panel_rect == bounds
    inspected.title.title_crop[:] = 0
    assert find_tooltip_title(image, layout).title_crop.any()
    image[:] = (90, 110, 130)
    inspected = inspect_tooltip_panel(image, layout)
    assert inspected.status == "absent" and inspected.title is inspected.reason is None


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_boundary_white_evidence_cannot_establish_clear_search_or_unique_panel(scene, edge):
    image, _, layout = scene
    x, y, width, height = layout.search_rect
    point = {"left": (y + 2, x), "right": (y + 2, x + width - 1),
             "top": (y, x + 2), "bottom": (y + height - 1, x + 2)}[edge]
    image[point] = 255
    inspected = inspect_tooltip_panel(image, layout)
    assert (inspected.status, inspected.reason) == ("unresolved", "search_boundary")
    assert inspected.title is None and find_tooltip_title(image, layout) is None
    image[:] = (90, 110, 130)
    image[point] = 255
    assert inspect_tooltip_panel(image, layout).status == "unresolved"


def test_fragmentation_budget_decline_is_never_a_negative_observation(scene, monkeypatch):
    image, _, layout = scene
    monkeypatch.setattr(detector, "MAX_WHITE_RUNS", 1)
    inspected = inspect_tooltip_panel(image, layout)
    assert (inspected.status, inspected.reason, inspected.title) == (
        "unresolved", "fragmentation_limit", None,
    )


def test_plausible_panel_with_invalid_title_or_perimeter_is_unresolved(scene):
    image, (x, y, width, _), layout = scene
    invalid = replace(layout, title_offset_rect=(width - 3, 8, 15, 12))
    assert inspect_tooltip_panel(image, invalid).reason == "title_geometry"
    image[y - 1, x:x + width] = (140, 140, 140)
    inspected = inspect_tooltip_panel(image, layout)
    assert (inspected.status, inspected.reason) == ("unresolved", "panel_perimeter")


def test_a_narrow_clipped_beveled_panel_cannot_be_dismissed_as_a_small_hud_glyph():
    image = np.full((120, 150, 3), 90, dtype=np.uint8)
    image[20:95, 20:120] = 247
    image[95:100, 30:110] = 247
    # The full connected panel satisfies today's .8 edge gate. Its bottom
    # strip is narrower than the minimum full width, so an orthogonal-width
    # heuristic would wrongly report a clear search after clipping its top.
    layout = TooltipLayout((0, 0, 150, 120), (95, 105), (75, 85), (10, 10, 60, 20))
    assert inspect_tooltip_panel(image, layout).status == "unique"
    clipped = image[95:110].copy()
    result = inspect_tooltip_panel(clipped, replace(layout, search_rect=(0, 0, 150, 15)))
    assert (result.status, result.reason, result.title) == ("unresolved", "search_boundary", None)


def test_panel_cut_by_image_edge_is_not_completed_or_guessed(scene):
    image, (x, _, width, _), layout = scene
    cut = image[:, :x + width - 10].copy()
    layout = replace(layout, search_rect=(0, 0, cut.shape[1], cut.shape[0]))
    assert find_tooltip_title(cut, layout) is None


def test_panel_cut_by_search_edge_is_rejected(scene):
    image, (x, _, _, _), layout = scene
    clipped_search = (x + 3, 0, image.shape[1] - x - 3, image.shape[0])
    assert find_tooltip_title(image, replace(layout, search_rect=clipped_search)) is None


def test_title_offset_must_stay_inside_detected_white_panel(scene):
    image, (_, _, width, _), layout = scene
    assert find_tooltip_title(image, replace(layout, title_offset_rect=(width - 3, 8, 15, 12))) is None


def test_search_must_stay_inside_supplied_image(scene):
    image, _, layout = scene
    with pytest.raises(ValueError, match="outside"):
        find_tooltip_title(image, replace(layout, search_rect=(0, 0, image.shape[1] + 1, image.shape[0])))


@pytest.mark.parametrize("bad_image", [np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 3, 4), dtype=np.uint8),
                                       np.zeros((0, 3, 3), dtype=np.uint8), np.zeros((2, 3, 3), dtype=float)])
def test_rgb_shape_and_dtype_are_explicit(bad_image):
    layout = TooltipLayout((0, 0, 3, 2), (1, 3), (1, 2), (0, 0, 1, 1))
    with pytest.raises(ValueError, match="uint8 RGB"):
        find_tooltip_title(bad_image, layout)


@pytest.mark.parametrize("changes", [{"search_rect": (-1, 0, 20, 20)}, {"title_offset_rect": (0, 0, 0, 10)},
                                     {"panel_width_range": (30, 20)}, {"panel_height_range": (True, 40)},
                                     {"min_white_channel": 256}, {"min_white_fraction": .5},
                                     {"min_white_edge_fraction": float("nan")}, {"dark_perimeter_max_channel": 255},
                                     {"search_rect": (0, 0, 4000, 4000)}])
def test_invalid_calibration_is_rejected(changes):
    options = dict(search_rect=(0, 0, 80, 80), panel_width_range=(20, 40), panel_height_range=(20, 40),
                   title_offset_rect=(2, 2, 15, 10))
    options.update(changes)
    with pytest.raises(ValueError):
        TooltipLayout(**options)
