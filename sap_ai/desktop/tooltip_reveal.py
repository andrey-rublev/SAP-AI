"""Pure six-frame tooltip appearance evidence, separate from strict clearance.

No capture, input, OCR, species assignment or gameplay integration occurs here.
The caller must supply independent snapshots in parked/parked/hover/hover/
parked/parked order, four corresponding clear Boards, and predeclared change
limits. A successful result does not prove which target caused a panel or that
no clipped tooltip exists elsewhere. Unknown Board fields remain unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from sap_ai.desktop.state import Board, Phase
from sap_ai.desktop.tooltip_layout import MAX_SEARCH_PIXELS, TooltipLayout, TooltipTitle, inventory_tooltip_panels


MAX_FRAME_PIXELS = MAX_SEARCH_PIXELS


REVEAL_LIMIT = (
    "Bounded development appearance evidence only; no global tooltip absence, "
    "target identity or causation is established. Unchanged clipped components "
    "outside the revealed panel and synchronized unrelated overlays remain possible. "
    "Unknown Board fields remain unknown."
)


@dataclass(frozen=True)
class TooltipChangeLimits:
    """Caller-calibrated substantial change, measured within the white panel.

    A changed pixel differs by at least min_channel_change in any RGB channel.
    A newly white pixel must also pass that change gate and change from outside
    to inside the layout's near-white color gate. Both fractions use full panel
    area, preventing a tiny title/edge change from establishing a new panel.
    """

    min_channel_change: int
    min_changed_fraction: float
    min_new_white_fraction: float

    def __post_init__(self):
        if type(self.min_channel_change) is not int or not 1 <= self.min_channel_change <= 255:
            raise ValueError("min_channel_change must be an integer from 1 through 255")
        for name in ("min_changed_fraction", "min_new_white_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite, above zero and at most one")


@dataclass(frozen=True)
class TooltipRevealInspection:
    status: Literal["revealed", "refused"]
    reason: str | None = None
    title: TooltipTitle | None = None
    changed_fraction: float | None = None
    new_white_fraction: float | None = None
    unchanged_boundary_components: int = 0
    limitation: str = REVEAL_LIMIT


def _patch(frame, rect):
    x, y, width, height = rect
    return frame[y:y + height, x:x + width]


def _overlap(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _white(patch, layout):
    minimum = patch.min(axis=2).astype(np.int16)
    maximum = patch.max(axis=2).astype(np.int16)
    return (minimum >= layout.min_white_channel) & (maximum - minimum <= layout.max_white_spread)


def verify_revealed_tooltip(*, frames, clear_boards, layout: TooltipLayout,
                            target_row: str, target_index: int,
                            change_limits: TooltipChangeLimits) -> TooltipRevealInspection:
    """Check a complete supplied sequence without interpreting tooltip text.

    Only unchanged boundary components disjoint from the panel and its one-pixel
    perimeter are tolerated. They remain unresolved; they are never relabeled
    absent. Every other unresolved inspection, visible clear candidate, changed
    clear Board, unstable hover or failed restoration refuses the sequence.
    Invalid arguments raise; inadequate observation evidence returns a refusal.
    """
    if not isinstance(layout, TooltipLayout) or not isinstance(change_limits, TooltipChangeLimits):
        raise TypeError("explicit TooltipLayout and TooltipChangeLimits are required")
    if (not isinstance(target_row, str) or target_row not in {"shop", "team"}
            or type(target_index) is not int or not 0 <= target_index < 5):
        raise ValueError("target must name a shop/team slot from zero through four")
    if not isinstance(frames, (list, tuple)) or len(frames) != 6:
        raise ValueError("six independent frames in clear/clear/hover/hover/clear/clear order are required")
    if (not isinstance(clear_boards, (list, tuple)) or len(clear_boards) != 4
            or not all(isinstance(board, Board) for board in clear_boards)):
        raise ValueError("four typed clear Boards are required")
    arrays = tuple(np.asarray(frame) for frame in frames)
    for frame in arrays:
        if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
                or min(frame.shape[:2]) <= 0 or frame.shape != arrays[0].shape):
            raise ValueError("six equally sized nonempty uint8 RGB frames are required")
        if frame.shape[0] * frame.shape[1] > MAX_FRAME_PIXELS:
            raise ValueError("tooltip frames exceed the bounded pixel budget")
    # Reused capture storage loses history before this pure verifier can see it.
    # Never count six aliases of the latest buffer as independent observations.
    # The conservative bounds check also avoids unbounded exact overlap work
    # on unusually strided arrays. Suspected sharing is not fresh evidence.
    if any(np.may_share_memory(frame, earlier) for index, frame in enumerate(arrays) for earlier in arrays[:index]):
        return TooltipRevealInspection("refused", "shared_frame_storage")
    arrays = tuple(frame.copy() for frame in arrays)
    if any(board.phase != Phase.SHOP for board in clear_boards):
        return TooltipRevealInspection("refused", "clear_phase")
    if any(board.fingerprint() != clear_boards[0].fingerprint() for board in clear_boards[1:]):
        return TooltipRevealInspection("refused", "clear_board_changed")
    slots = getattr(clear_boards[0], target_row)
    if target_index >= len(slots) or slots[target_index].occupied is not True:
        return TooltipRevealInspection("refused", "target_not_known_occupied")

    inventories = tuple(inventory_tooltip_panels(frame, layout) for frame in arrays)
    if any(set(item.unresolved_reasons) - {"search_boundary"} for item in inventories):
        return TooltipRevealInspection("refused", "unresolved_interior_or_work_limit")
    clear_indices = (0, 1, 4, 5)
    if any(inventories[index].candidates for index in clear_indices):
        return TooltipRevealInspection("refused", "clear_panel_present")
    if any(len(inventories[index].candidates) != 1 for index in (2, 3)):
        return TooltipRevealInspection("refused", "hover_panel_count")
    title, repeated = (inventories[index].candidates[0] for index in (2, 3))
    if title.panel_rect != repeated.panel_rect or title.title_rect != repeated.title_rect:
        return TooltipRevealInspection("refused", "hover_geometry_changed")
    x, y, width, height = title.panel_rect
    perimeter = (x - 1, y - 1, width + 2, height + 2)
    boundary = inventories[0].boundary_rects
    if any(item.boundary_rects != boundary for item in inventories[1:]):
        return TooltipRevealInspection("refused", "boundary_geometry_changed")
    for rect in boundary:
        if _overlap(rect, perimeter):
            return TooltipRevealInspection("refused", "boundary_overlaps_panel")
        if any(not np.array_equal(_patch(arrays[0], rect), _patch(frame, rect)) for frame in arrays[1:]):
            return TooltipRevealInspection("refused", "boundary_pixels_changed")
    if not np.array_equal(_patch(arrays[2], perimeter), _patch(arrays[3], perimeter)):
        return TooltipRevealInspection("refused", "hover_pixels_changed")
    baseline = _patch(arrays[0], perimeter)
    if any(not np.array_equal(baseline, _patch(arrays[index], perimeter)) for index in clear_indices[1:]):
        return TooltipRevealInspection("refused", "clear_pixels_not_restored")

    before, after = (_patch(arrays[index], title.panel_rect) for index in (0, 2))
    changed = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=2) >= change_limits.min_channel_change
    newly_white = changed & _white(after, layout) & ~_white(before, layout)
    changed_fraction, new_white_fraction = float(changed.mean()), float(newly_white.mean())
    if (changed_fraction < change_limits.min_changed_fraction
            or new_white_fraction < change_limits.min_new_white_fraction):
        return TooltipRevealInspection("refused", "insufficient_panel_appearance",
                                       changed_fraction=changed_fraction, new_white_fraction=new_white_fraction)
    return TooltipRevealInspection("revealed", title=title, changed_fraction=changed_fraction,
                                   new_white_fraction=new_white_fraction,
                                   unchanged_boundary_components=len(boundary))
