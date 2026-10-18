"""Pure six-frame tooltip appearance sequences; no native or filesystem IO."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from desktop_runtime import DesktopUnavailable
from desktop_state import Board, PetSlot, Phase
from desktop_tooltip_layout import TooltipLayout
from desktop_vision import Rect, SlotConfig, VisionProfile
from tools import probe_desktop_hover as probe


def frame(*panels, gold=10):
    image = np.full((100, 100, 3), 90, dtype=np.uint8)
    image[0, 0] = (gold, 0, 0)
    for x, y in panels:
        image[y - 1:y + 41, x - 1:x + 31] = 0
        image[y:y + 40, x:x + 30] = (247, 247, 245)
        image[y + 5:y + 10, x + 5:x + 19] = (190, 100, 15)
    return image


def setup(frames=None):
    profile = VisionProfile(image_size=(100, 100), calibrated=True, pointer_rest=(50, 95),
        shop=(SlotConfig(Rect(10, 60, 10, 10)),),
        team=tuple(SlotConfig(Rect(10 + index * 15, 20, 10, 10)) for index in range(5)))
    layout = TooltipLayout((0, 0, 100, 100), (28, 32), (38, 42), (3, 3, 20, 10),
                           dark_perimeter_max_channel=60)
    board = Board(Phase.SHOP, gold=10, turn=2, shop=(PetSlot(True, None, 2, 3, 1),),
                  team=(PetSlot(False),) * 5)
    values = frames or [frame(), frame(), frame((55, 35)), frame((55, 35)), frame(), frame()]
    capture, hover, wait, check = Mock(side_effect=iter(values)), Mock(), Mock(), Mock()
    def observe(image):
        # Even an eager injected reader must never see tooltip-covered pixels.
        assert np.array_equal(image[40, 60], [90, 90, 90])
        return replace(board, gold=int(image[0, 0, 0]))
    observe = Mock(side_effect=observe)
    return profile, layout, capture, hover, observe, wait, check


def collect(parts, evidence=None, **options):
    profile, layout, capture, hover, observe, wait, check = parts
    return probe.collect_association(profile, probe.HoverTarget("shop", 0), layout,
        capture=capture, hover=hover, observe=observe, wait=wait, require_running=check,
        evidence=evidence, **options)


def test_complete_sequence_is_diagnostic_evidence_without_identity_enrichment():
    parts = setup()
    result = collect(parts)
    _, _, capture, hover, observe, wait, _ = parts
    assert result["appearance_sequence_verified"] is True and result["pointer_restored"] is True
    assert capture.call_count == 6 and observe.call_count == 4
    assert hover.call_args_list == [call((50, 95)), call((15, 65)), call((50, 95))]
    assert wait.call_args_list == [call(.15), call(.1), call(.75), call(.1), call(.15), call(.1)]
    assert result["before"] == result["after"]
    assert result["before"]["shop"][0]["species"] is None
    hover_samples = [row for row in result["samples"] if row["stage"] == "hover"]
    assert len(hover_samples) == 2 and all("board" not in row for row in hover_samples)
    assert hover_samples[0]["title_pixel_sha256"] == hover_samples[1]["title_pixel_sha256"]
    assert not any("identity" in key or "species" in key for row in hover_samples for key in row)


@pytest.mark.parametrize("failure", ["stale", "ambiguous", "unresolved"])
def test_parking_requires_proven_absence_not_a_missing_unique_title(failure):
    first = frame((55, 35))
    if failure == "ambiguous":
        first = frame((10, 10), (55, 35))
    elif failure == "unresolved":
        first = frame()
        first[:40, :30] = 247  # Search-boundary contact is not proven absence.
    parts, evidence = setup([first]), {}
    with pytest.raises(DesktopUnavailable, match="absence not established"):
        collect(parts, evidence)
    assert evidence["appearance_sequence_verified"] is False and evidence["hover_attempted"] is False
    assert parts[2].call_count == 1 and parts[4].call_count == 0
    assert parts[3].call_args_list == [call((50, 95))]


def test_detector_work_bound_is_not_mistaken_for_tooltip_absence(monkeypatch):
    parts, evidence = setup(), {}
    monkeypatch.setattr(probe, "inspect_tooltip_panel", Mock(return_value=SimpleNamespace(
        status="unresolved", title=None, reason="white_component_work_bound")))
    with pytest.raises(DesktopUnavailable, match="absence not established.*unresolved"):
        collect(parts, evidence)
    assert parts[4].call_count == 0 and parts[3].call_count == 1
    assert evidence["samples"][0]["tooltip_reason"] == "white_component_work_bound"
    assert evidence["appearance_sequence_verified"] is False


@pytest.mark.parametrize("bad_hover", ["absent", "disappeared", "ambiguous", "unresolved"])
def test_settled_hover_requires_one_panel_in_both_frames_without_retries(bad_hover):
    absent, shown = frame(), frame((55, 35))
    invalid = frame((10, 10), (55, 35)) if bad_hover == "ambiguous" else absent.copy()
    if bad_hover == "unresolved":
        invalid[:40, :30] = 247
    frames = [absent, absent, shown, invalid] if bad_hover != "absent" else [absent, absent, absent]
    parts, evidence = setup(frames), {}
    with pytest.raises(DesktopUnavailable, match="settled hover tooltip"):
        collect(parts, evidence)
    assert evidence["appearance_sequence_verified"] is False and evidence["hover_attempted"] is True
    assert parts[4].call_count == 2
    assert parts[3].call_args_list == [call((50, 95)), call((15, 65))]
    assert parts[2].call_count == len(frames)


@pytest.mark.parametrize("change", ["title", "body", "position"])
def test_changing_title_body_or_panel_position_cannot_form_an_association(change):
    shown = frame((55, 35))
    changed = shown.copy()
    if change == "title":
        changed[40:45, 60:74] = (210, 80, 20)
    elif change == "body":
        changed[60, 70] = (120, 100, 50)
    else:
        changed = frame((58, 35))
    parts, evidence = setup([frame(), frame(), shown, changed]), {}
    with pytest.raises(DesktopUnavailable, match="panel or title changed"):
        collect(parts, evidence)
    assert evidence["appearance_sequence_verified"] is False and parts[2].call_count == 4
    assert parts[3].call_count == 2 and parts[4].call_count == 2
    if change == "body":
        rows = [row for row in evidence["samples"] if row["stage"] == "hover"]
        assert rows[0]["title_pixel_sha256"] == rows[1]["title_pixel_sha256"]


def test_reused_camera_buffer_cannot_hide_a_changing_tooltip_body():
    shown = frame((55, 35))
    changed = shown.copy()
    changed[60, 70] = (120, 100, 50)
    images = iter([frame(), frame(), shown, changed, frame(), frame()])
    shared = np.empty_like(shown)
    parts, evidence = setup(), {}
    def capture():
        shared[:] = next(images)
        return shared
    parts[2].side_effect = capture
    with pytest.raises(DesktopUnavailable, match="panel or title changed"):
        collect(parts, evidence)
    assert evidence["appearance_sequence_verified"] is False
    assert parts[2].call_count == 4 and parts[3].call_count == 2


@pytest.mark.parametrize("change", ["gold", "other_slot", "target_missing", "unreadable"])
def test_hidden_board_changes_or_uncertainty_after_clearing_reject_association(change):
    parts, evidence = setup(), {}
    original = parts[4].side_effect
    calls = 0
    def observe(image):
        nonlocal calls
        calls += 1
        board = original(image)
        if calls <= 2:
            return board
        if change == "gold":
            return replace(board, gold=9)
        if change == "other_slot":
            return replace(board, team=(PetSlot(True, None, 2, 3, 1),) + board.team[1:])
        if change == "target_missing":
            return replace(board, shop=(PetSlot(False),))
        return replace(board, turn=None)
    parts[4].side_effect = observe
    with pytest.raises(DesktopUnavailable, match="underlying shop changed|not known occupied"):
        collect(parts, evidence)
    assert evidence["appearance_sequence_verified"] is False
    assert parts[3].call_count == 3 and parts[4].call_count <= 4 and parts[2].call_count <= 6


def test_tooltip_that_stays_visible_after_final_parking_cannot_pass_clear_check():
    shown = frame((55, 35))
    parts, evidence = setup([frame(), frame(), shown, shown, shown]), {}
    with pytest.raises(DesktopUnavailable, match="absence not established"):
        collect(parts, evidence)
    assert parts[4].call_count == 2 and parts[2].call_count == 5
    assert evidence["appearance_sequence_verified"] is False


def test_changing_board_before_hover_prevents_target_movement():
    parts, evidence = setup([frame(), frame(gold=9)]), {}
    with pytest.raises(DesktopUnavailable, match="shop changed during probe preflight"):
        collect(parts, evidence)
    assert parts[3].call_args_list == [call((50, 95))]


def test_stop_during_settle_prevents_capture_and_followup_pointer_movement():
    parts, evidence = setup(), {}
    stopped = False
    def wait(seconds):
        nonlocal stopped
        if seconds == .75:
            stopped = True
    def check():
        if stopped:
            raise probe.ProbeStopped("test stop")
    parts[5].side_effect, parts[6].side_effect = wait, check
    with pytest.raises(probe.ProbeStopped, match="test stop"):
        collect(parts, evidence)
    assert parts[2].call_count == parts[4].call_count == 2
    assert parts[3].call_count == 2 and evidence["appearance_sequence_verified"] is False


@pytest.mark.parametrize("bad", ["profile", "rest", "layout", "target", "settle"])
def test_invalid_explicit_calibration_fails_before_any_supplied_io(bad):
    parts = list(setup())
    options = {}
    if bad == "profile":
        parts[0] = replace(parts[0], calibrated=False)
    elif bad == "rest":
        parts[0] = replace(parts[0], pointer_rest=None)
    elif bad == "layout":
        parts[1] = replace(parts[1], search_rect=(0, 0, 101, 100))
    elif bad == "target":
        parts[0] = replace(parts[0], shop=())
    else:
        options["settle_seconds"] = 3
    with pytest.raises(ValueError):
        collect(parts, **options)
    for callback in parts[2:]:
        callback.assert_not_called()
