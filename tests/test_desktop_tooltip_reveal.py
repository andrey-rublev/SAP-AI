"""Synthetic temporal evidence; no game pixels, capture, OCR or native input."""
from dataclasses import replace

import numpy as np
import pytest

from sap_ai.desktop.state import Board, FoodSlot, PetSlot, Phase
from sap_ai.desktop.tooltip_layout import TooltipLayout, inspect_tooltip_panel
import sap_ai.desktop.tooltip_layout as detector
import sap_ai.desktop.tooltip_reveal as verifier
from sap_ai.desktop.tooltip_reveal import TooltipChangeLimits, verify_revealed_tooltip


def paint_panel(frame, rect):
    x, y, width, height = rect
    frame[y - 1:y + height + 1, x - 1:x + width + 1] = 0
    frame[y:y + height, x:x + width] = 247
    frame[y + 10:y + 20, x + 15:x + 45] = (190, 100, 15)


@pytest.fixture(params=[1, 2])
def sequence(request):
    scale = request.param
    clear = np.full((180 * scale, 300 * scale, 3), (90, 110, 130), dtype=np.uint8)
    # An unrelated clipped HUD glyph keeps the strict search unresolved.
    clear[0:5, 15:24] = 247
    rect = (60 * scale, 45 * scale, 100 * scale, 80 * scale)
    hovered = clear.copy()
    paint_panel(hovered, rect)
    layout = TooltipLayout((0, 0, clear.shape[1], clear.shape[0]),
                           (rect[2], rect[2]), (rect[3], rect[3]), (10, 8, 65, 20),
                           dark_perimeter_max_channel=60)
    board = Board(Phase.SHOP, gold=7, turn=2, shop=(PetSlot(True, attack=2, health=3),),
                  team=(PetSlot(True, attack=1, health=2),))
    return dict(frames=[frame.copy() for frame in (clear, clear, hovered, hovered, clear, clear)],
                clear_boards=[board] * 4, layout=layout, target_row="shop", target_index=0,
                change_limits=TooltipChangeLimits(30, .6, .6))


def test_static_boundary_does_not_establish_absence_but_can_survive_separate_reveal(sequence):
    assert all(inspect_tooltip_panel(frame, sequence["layout"]).status == "unresolved"
               for frame in sequence["frames"])
    found = verify_revealed_tooltip(**sequence)
    assert found.status == "revealed" and found.reason is None
    assert found.unchanged_boundary_components == 1
    assert found.changed_fraction > .95 and found.new_white_fraction > .95
    assert "no global tooltip absence" in found.limitation and "target identity" in found.limitation
    assert sequence["clear_boards"][0].shop[0].species is None
    source = sequence["frames"][2].copy()
    found.title.title_crop[:] = 0
    assert np.array_equal(sequence["frames"][2], source)


def test_a_clear_search_can_use_the_same_reveal_contract(sequence):
    for frame in sequence["frames"]:
        frame[0:5, 15:24] = (90, 110, 130)
    found = verify_revealed_tooltip(**sequence)
    assert found.status == "revealed" and found.unchanged_boundary_components == 0


@pytest.mark.parametrize("index", [0, 1, 4, 5])
def test_a_stale_or_unrelated_clear_panel_refuses(sequence, index):
    sequence["frames"][index] = sequence["frames"][2].copy()
    assert verify_revealed_tooltip(**sequence).reason == "clear_panel_present"


@pytest.mark.parametrize("index", [2, 3])
def test_missing_hover_panel_refuses(sequence, index):
    sequence["frames"][index] = sequence["frames"][0].copy()
    assert verify_revealed_tooltip(**sequence).reason == "hover_panel_count"


def test_two_bounded_hover_panels_refuse(sequence):
    rect = inspect_tooltip_panel(sequence["frames"][2], sequence["layout"])
    assert rect.title is None  # Still unresolved due to the unrelated HUD glyph.
    width = sequence["layout"].panel_width_range[0]
    height = sequence["layout"].panel_height_range[0]
    for index in (2, 3):
        paint_panel(sequence["frames"][index], (180 * (width // 100), 45 * (height // 80), width, height))
    assert verify_revealed_tooltip(**sequence).reason == "hover_panel_count"


def test_moving_or_resegmented_hover_panel_refuses(sequence):
    moved = sequence["frames"][0].copy()
    width = sequence["layout"].panel_width_range[0]
    height = sequence["layout"].panel_height_range[0]
    paint_panel(moved, (60 * (width // 100) + 1, 45 * (height // 80), width, height))
    sequence["frames"][3] = moved
    assert verify_revealed_tooltip(**sequence).reason == "hover_geometry_changed"


@pytest.mark.parametrize("location", ["title", "perimeter"])
def test_hover_pixels_including_perimeter_must_be_stable(sequence, location):
    width = sequence["layout"].panel_width_range[0]
    x, y = 60 * (width // 100), 45 * (width // 100)
    point = (y + 12, x + 20) if location == "title" else (y - 1, x + 20)
    sequence["frames"][3][point] = (10, 20, 30)
    assert verify_revealed_tooltip(**sequence).reason == "hover_pixels_changed"


@pytest.mark.parametrize("index", [1, 4, 5])
def test_clear_patch_must_restore_exactly_before_and_after(sequence, index):
    width = sequence["layout"].panel_width_range[0]
    sequence["frames"][index][45 * (width // 100), 60 * (width // 100)] = 89
    assert verify_revealed_tooltip(**sequence).reason == "clear_pixels_not_restored"


@pytest.mark.parametrize("index", range(1, 6))
def test_changed_boundary_pixels_cannot_hide_behind_same_geometry(sequence, index):
    sequence["frames"][index][1, 16] = 246  # Still the same near-white connected component.
    assert verify_revealed_tooltip(**sequence).reason == "boundary_pixels_changed"


def test_new_clipped_hover_panel_or_changed_boundary_geometry_refuses(sequence):
    for index in (2, 3):
        sequence["frames"][index][0:8, 35:70] = 247
    assert verify_revealed_tooltip(**sequence).reason == "boundary_geometry_changed"


def test_unchanged_boundary_bounding_patch_cannot_overlap_revealed_panel(sequence):
    scale = sequence["layout"].panel_width_range[0] // 100
    for frame in sequence["frames"]:
        # A C-shaped boundary component surrounds, without touching, the panel.
        frame[:160 * scale, 0:2] = 247
        frame[10:12, :290 * scale] = 247
        frame[158 * scale:160 * scale, :290 * scale] = 247
    assert verify_revealed_tooltip(**sequence).reason == "boundary_overlaps_panel"


def test_work_limit_is_never_tolerated_as_an_unchanged_hud(sequence, monkeypatch):
    monkeypatch.setattr(detector, "MAX_WHITE_RUNS", 1)
    assert verify_revealed_tooltip(**sequence).reason == "unresolved_interior_or_work_limit"


def test_interior_failure_is_not_hidden_by_boundary_reason(sequence):
    for index in (2, 3):
        width = sequence["layout"].panel_width_range[0]
        x, y = 60 * (width // 100), 45 * (width // 100)
        sequence["frames"][index][y - 1, x:x + width] = 90
    assert verify_revealed_tooltip(**sequence).reason == "unresolved_interior_or_work_limit"


def test_tiny_eligibility_change_is_not_a_new_tooltip():
    clear = np.full((140, 180, 3), 90, dtype=np.uint8)
    clear[30:109, 30:130] = 247  # An already visible panel, one row below the calibrated height.
    hovered = clear.copy()
    hovered[109, 30:130] = 247
    board = Board(Phase.SHOP, shop=(PetSlot(True),))
    found = verify_revealed_tooltip(
        frames=[frame.copy() for frame in (clear, clear, hovered, hovered, clear, clear)],
        clear_boards=[board] * 4, target_row="shop", target_index=0,
        layout=TooltipLayout((0, 0, 180, 140), (100, 100), (80, 80), (10, 10, 60, 20)),
        change_limits=TooltipChangeLimits(30, .6, .6))
    assert found.reason == "insufficient_panel_appearance"
    assert found.changed_fraction == found.new_white_fraction == 1 / 80


def test_large_color_change_without_new_white_area_is_not_a_new_tooltip():
    clear = np.full((140, 180, 3), 90, dtype=np.uint8)
    clear[30:109, 30:130] = 255
    hovered = clear.copy()
    hovered[30:110, 30:130] = 230
    board = Board(Phase.SHOP, shop=(PetSlot(True),))
    found = verify_revealed_tooltip(
        frames=[frame.copy() for frame in (clear, clear, hovered, hovered, clear, clear)],
        clear_boards=[board] * 4, target_row="shop", target_index=0,
        layout=TooltipLayout((0, 0, 180, 140), (100, 100), (80, 80), (10, 10, 60, 20)),
        change_limits=TooltipChangeLimits(20, .6, .6))
    assert found.reason == "insufficient_panel_appearance"
    assert found.changed_fraction == 1 and found.new_white_fraction == 1 / 80


@pytest.mark.parametrize("change", [dict(gold=6), dict(turn=3), dict(phase=Phase.BATTLE),
                                    dict(team=(PetSlot(True, attack=4, health=2),)),
                                    dict(food=(FoodSlot(True, "apple", 3),))])
def test_changed_typed_clear_board_or_phase_refuses(sequence, change):
    sequence["clear_boards"][3] = replace(sequence["clear_boards"][0], **change)
    assert verify_revealed_tooltip(**sequence).status == "refused"


@pytest.mark.parametrize("occupied", [None, False])
def test_target_must_be_known_occupied(sequence, occupied):
    board = replace(sequence["clear_boards"][0], shop=(PetSlot(occupied),))
    sequence["clear_boards"] = [board] * 4
    assert verify_revealed_tooltip(**sequence).reason == "target_not_known_occupied"


def test_missing_target_slot_refuses(sequence):
    sequence["target_index"] = 4
    assert verify_revealed_tooltip(**sequence).reason == "target_not_known_occupied"


def test_reused_capture_storage_is_not_temporal_evidence(sequence):
    sequence["frames"][1] = sequence["frames"][0].view()
    assert verify_revealed_tooltip(**sequence).reason == "shared_frame_storage"


def test_verification_uses_snapshots_after_accepting_independent_storage(sequence, monkeypatch):
    original = verifier.inventory_tooltip_panels

    def mutate_callers_after_snapshot(frame, layout):
        for supplied in sequence["frames"]:
            supplied[:] = 0
        return original(frame, layout)

    monkeypatch.setattr(verifier, "inventory_tooltip_panels", mutate_callers_after_snapshot)
    found = verify_revealed_tooltip(**sequence)
    assert found.status == "revealed" and found.title.title_crop.any()
    assert all(not frame.any() for frame in sequence["frames"])


def test_full_frame_budget_is_checked_before_copying(sequence, monkeypatch):
    shape = sequence["frames"][0].shape
    monkeypatch.setattr(verifier, "MAX_FRAME_PIXELS", shape[0] * shape[1] - 1)
    with pytest.raises(ValueError, match="bounded pixel budget"):
        verify_revealed_tooltip(**sequence)


def test_interleaved_storage_is_conservatively_refused_without_exact_overlap_search(sequence):
    shape = sequence["frames"][0].shape
    buffer = np.empty((shape[0] * 2, *shape[1:]), dtype=np.uint8)
    buffer[::2] = sequence["frames"][0]
    buffer[1::2] = sequence["frames"][1]
    sequence["frames"][:2] = (buffer[::2], buffer[1::2])
    assert verify_revealed_tooltip(**sequence).reason == "shared_frame_storage"


@pytest.mark.parametrize("field,value", [("min_channel_change", True), ("min_channel_change", 0),
                                         ("min_channel_change", 256), ("min_changed_fraction", 0),
                                         ("min_changed_fraction", float("nan")),
                                         ("min_new_white_fraction", 1.01),
                                         ("min_new_white_fraction", True)])
def test_change_limits_are_explicit_finite_and_nontrivial(field, value):
    options = dict(min_channel_change=30, min_changed_fraction=.6, min_new_white_fraction=.6)
    options[field] = value
    with pytest.raises(ValueError):
        TooltipChangeLimits(**options)


@pytest.mark.parametrize("mutation", ["few_frames", "few_boards", "bad_board", "float_frame", "different_shape"])
def test_invalid_sequence_inputs_raise(sequence, mutation):
    if mutation == "few_frames":
        sequence["frames"].pop()
    elif mutation == "few_boards":
        sequence["clear_boards"].pop()
    elif mutation == "bad_board":
        sequence["clear_boards"][0] = {}
    elif mutation == "float_frame":
        sequence["frames"][0] = sequence["frames"][0].astype(float)
    else:
        sequence["frames"][0] = sequence["frames"][0][:-1].copy()
    with pytest.raises(ValueError):
        verify_revealed_tooltip(**sequence)
