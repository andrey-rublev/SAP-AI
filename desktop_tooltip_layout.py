"""Locate a calibrated white tooltip panel in one explicitly supplied RGB image.

All rectangles are (x, y, width, height). The title offset is relative to the
connected near-white panel's bounds, excluding any surrounding dark border.
This detector has no game coordinates, OCR, capture, or production integration;
its geometry and color limits require validation for the caller's layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MAX_SEARCH_PIXELS = 4_000_000
MAX_WHITE_RUNS = 100_000


def _rect(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 4 or any(type(n) is not int for n in value)
            or min(value[:2]) < 0 or min(value[2:]) <= 0):
        raise ValueError(f"{name} must be nonnegative x/y and positive integer width/height")
    return tuple(value)


@dataclass(frozen=True)
class TooltipLayout:
    search_rect: tuple[int, int, int, int]
    panel_width_range: tuple[int, int]
    panel_height_range: tuple[int, int]
    title_offset_rect: tuple[int, int, int, int]
    min_white_channel: int = 230
    max_white_spread: int = 25
    min_white_fraction: float = .85
    min_white_edge_fraction: float = .8
    dark_perimeter_max_channel: int | None = None
    min_dark_perimeter_fraction: float = .8

    def __post_init__(self):
        for name in ("search_rect", "title_offset_rect"):
            object.__setattr__(self, name, _rect(getattr(self, name), name))
        for name in ("panel_width_range", "panel_height_range"):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(type(n) is not int for n in value) or not 0 < value[0] <= value[1]):
                raise ValueError(f"{name} must be a positive inclusive integer range")
            object.__setattr__(self, name, tuple(value))
        if (type(self.min_white_channel) is not int or not 1 <= self.min_white_channel <= 255
                or type(self.max_white_spread) is not int or not 0 <= self.max_white_spread <= 255):
            raise ValueError("white color limits must be integer channel values")
        for name in ("min_white_fraction", "min_white_edge_fraction", "min_dark_perimeter_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not .5 < value <= 1:
                raise ValueError(f"{name} must be finite, above one half and at most one")
        if (self.dark_perimeter_max_channel is not None
                and (type(self.dark_perimeter_max_channel) is not int
                     or not 0 <= self.dark_perimeter_max_channel < self.min_white_channel)):
            raise ValueError("dark perimeter limit must be below the white channel limit or None")
        if self.search_rect[2] * self.search_rect[3] > MAX_SEARCH_PIXELS:
            raise ValueError("calibrated search rectangle exceeds the bounded search size")


@dataclass(frozen=True)
class TooltipTitle:
    title_crop: np.ndarray
    panel_rect: tuple[int, int, int, int]
    title_rect: tuple[int, int, int, int]
    white_fraction: float
    white_edge_fraction: float
    dark_perimeter_fraction: float | None


def _white_components(mask):
    """Exact four-connected components using white row runs, without pixel BFS."""
    parents, runs, previous = [], [], []

    def root(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        current, cursor = [], 0
        for start, end in zip(edges[::2], edges[1::2]):
            if len(runs) >= MAX_WHITE_RUNS:
                return []  # Excessively fragmented evidence is not a panel.
            label = len(parents)
            parents.append(label)
            while cursor < len(previous) and previous[cursor][1] <= start:
                cursor += 1
            index = cursor
            while index < len(previous) and previous[index][0] < end:
                other = root(previous[index][2])
                parents[other] = root(label)
                index += 1
            current.append((int(start), int(end), label))
            runs.append((int(start), y, int(end), label))
        previous = current
    components = {}
    for start, y, end, label in runs:
        key = root(label)
        if key not in components:
            components[key] = [start, y, end, y + 1, end - start]
        else:
            bounds = components[key]
            bounds[0] = min(bounds[0], start)
            bounds[1] = min(bounds[1], y)
            bounds[2] = max(bounds[2], end)
            bounds[3] = max(bounds[3], y + 1)
            bounds[4] += end - start
    return components.values()


def find_tooltip_title(image, layout: TooltipLayout) -> TooltipTitle | None:
    """Return a copied title crop/evidence only for exactly one eligible panel.

    A panel must fit the calibrated dimensions, have dense white fill and all
    four white edges, and optionally have a dark one-pixel exterior perimeter.
    Components touching the search boundary are potentially clipped and fail.
    Invalid RGB/search geometry raises ValueError; absent, ambiguous, fragmented
    or clipped panel evidence and title offsets outside a panel return None.
    """
    if not isinstance(layout, TooltipLayout):
        raise TypeError("layout must be a TooltipLayout")
    frame = np.asarray(image)
    if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) <= 0):
        raise ValueError("tooltip layout needs a nonempty uint8 RGB image")
    sx, sy, sw, sh = layout.search_rect
    if sx + sw > frame.shape[1] or sy + sh > frame.shape[0]:
        raise ValueError("calibrated search rectangle is outside the supplied image")
    region = frame[sy:sy + sh, sx:sx + sw]
    maximum = region.max(axis=2).astype(np.int16)
    minimum = region.min(axis=2).astype(np.int16)
    white = (minimum >= layout.min_white_channel) & (maximum - minimum <= layout.max_white_spread)
    dark = maximum <= layout.dark_perimeter_max_channel if layout.dark_perimeter_max_channel is not None else None
    eligible = []
    for x0, y0, x1, y1, area in _white_components(white):
        width, height = x1 - x0, y1 - y0
        if (not layout.panel_width_range[0] <= width <= layout.panel_width_range[1]
                or not layout.panel_height_range[0] <= height <= layout.panel_height_range[1]
                or x0 == 0 or y0 == 0 or x1 == sw or y1 == sh):
            continue
        fraction = area / (width * height)
        edge_fraction = min(float(edge.mean()) for edge in (white[y0, x0:x1], white[y1 - 1, x0:x1],
                                                            white[y0:y1, x0], white[y0:y1, x1 - 1]))
        if fraction < layout.min_white_fraction or edge_fraction < layout.min_white_edge_fraction:
            continue
        perimeter_fraction = None
        if dark is not None:
            perimeter_fraction = min(float(edge.mean()) for edge in (dark[y0 - 1, x0:x1], dark[y1, x0:x1],
                                                                     dark[y0:y1, x0 - 1], dark[y0:y1, x1]))
            if perimeter_fraction < layout.min_dark_perimeter_fraction:
                continue
        tx, ty, tw, th = layout.title_offset_rect
        if tx + tw > width or ty + th > height:
            continue
        absolute_x, absolute_y = sx + x0 + tx, sy + y0 + ty
        eligible.append(TooltipTitle(
            title_crop=frame[absolute_y:absolute_y + th, absolute_x:absolute_x + tw].copy(),
            panel_rect=(sx + x0, sy + y0, width, height), title_rect=(absolute_x, absolute_y, tw, th),
            white_fraction=fraction, white_edge_fraction=edge_fraction, dark_perimeter_fraction=perimeter_fraction))
        if len(eligible) > 1:
            return None
    return eligible[0] if eligible else None
