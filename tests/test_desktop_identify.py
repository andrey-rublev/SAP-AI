"""Hover-based species identification with synthetic frames and a fake mouse."""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sap_ai.desktop.identify import PET_NAMES, SpeciesIdentifier, load_tooltip_layout
from sap_ai.desktop.runtime import DesktopRuntime
from sap_ai.desktop.state import Action, Board, PetSlot, Phase
from sap_ai.desktop.tooltip_layout import TooltipLayout
from sap_ai.desktop.vision import Rect

LAYOUT = TooltipLayout(search_rect=(0, 0, 200, 200), panel_width_range=(60, 100),
                       panel_height_range=(40, 80), title_offset_rect=(5, 5, 40, 10),
                       min_white_fraction=.75)
REST = (190, 190)


def parked_frame():
    return np.full((200, 200, 3), 40, dtype=np.uint8)


def tooltip_frame():
    frame = parked_frame()
    frame[50:100, 50:130] = 255
    frame[57:63, 56:80] = (220, 30, 30)  # one line of colored title text
    return frame


def shop_board(*, team_species=None, shop_attack=2, gold=10):
    team = (PetSlot(True, species=team_species, attack=3, health=4, level=1),) + (PetSlot(False),) * 4
    return Board(Phase.SHOP, gold=gold, turn=1, shop=(PetSlot(True, attack=shop_attack, health=3),), team=team)


class FakeGame:
    """Shows a tooltip naming the hovered pet; parked pointer shows none."""

    def __init__(self, names):
        self.names, self.pointer, self.hovers = names, REST, []
        self.click = Mock()

    def hover(self, point, size):
        self.pointer = tuple(point)
        self.hovers.append(tuple(point))

    def capture(self):
        return parked_frame() if self.pointer == REST else tooltip_frame()

    def ocr(self, image):
        return self.names.get(self.pointer, "")


def make_runtime(board, names, *, execute=True):
    profile = SimpleNamespace(
        validate=Mock(), image_size=(200, 200), pointer_rest=REST, phase_templates=(),
        buttons={"roll": (5, 190), "end_turn": (195, 5), "sell": (100, 190)},
        shop=[SimpleNamespace(portrait=Rect(10, 150, 10, 10))],
        team=[SimpleNamespace(portrait=Rect(10 + i * 20, 20, 10, 10)) for i in range(5)])
    game = FakeGame(names)
    perceptor = SimpleNamespace(observe=Mock(side_effect=lambda frame: board()))
    identifier = SpeciesIdentifier(LAYOUT, ocr=game.ocr, sleep=lambda seconds: None)
    driver = DesktopRuntime(profile, perceptor, game, execute=execute, identifier=identifier)
    return driver, game, identifier


def test_settled_shop_is_identified_once_and_pointer_parked():
    driver, game, identifier = make_runtime(shop_board, {(15, 25): "RHINO", (15, 155): "TIGER"})
    first = driver.observe()  # unsettled: never hover
    assert game.hovers == [REST] and first.team[0].species is None
    second = driver.observe()
    assert second.team[0].species == "rhino" and second.shop[0].species == "tiger"
    assert game.pointer == REST and identifier.identified == 2
    hovered = len(game.hovers)
    third = driver.observe()
    assert third == second and len(game.hovers) == hovered  # cached, no further hovering


def test_preview_never_moves_the_pointer():
    driver, game, _ = make_runtime(shop_board, {(15, 25): "RHINO"}, execute=False)
    driver.observe(), driver.observe()
    assert game.hovers == []


def test_visible_tooltip_in_parked_frame_prevents_hovering():
    driver, game, _ = make_runtime(shop_board, {(15, 25): "RHINO"})
    game.capture = lambda: tooltip_frame()
    driver.observe(), driver.observe()
    assert game.hovers == [REST]


def test_unreadable_titles_stop_after_bounded_attempts():
    driver, game, identifier = make_runtime(shop_board, {})
    for _ in range(6):
        board = driver.observe()
    assert board.team[0].species is None
    assert identifier.hovers == 2 * identifier.max_attempts  # two pets, two attempts each


def test_cached_species_keeps_action_preflight_stable():
    driver, game, _ = make_runtime(lambda: shop_board(), {(15, 25): "RHINO", (15, 155): "TIGER"})
    driver.observe(), driver.observe()
    driver.act(Action("roll"))  # preflight must not see a "changed" board
    game.click.assert_called_once()


def test_actions_invalidate_only_affected_slots():
    identifier = SpeciesIdentifier(LAYOUT, ocr=lambda image: "", sleep=lambda seconds: None)
    board = shop_board()
    identifier.annotate(board)
    identifier.record("shop", 0, board.shop[0], "Tiger")
    identifier.record("team", 0, board.team[0], "Rhino")
    identifier.forget(Action("roll"))
    annotated = identifier.annotate(board)
    assert annotated.shop[0].species is None and annotated.team[0].species == "rhino"
    identifier.forget(Action("sell", 0))
    assert identifier.annotate(board).team[0].species is None


def test_changed_stats_or_new_turn_invalidate_cached_identity():
    identifier = SpeciesIdentifier(LAYOUT, ocr=lambda image: "", sleep=lambda seconds: None)
    board = shop_board()
    identifier.annotate(board)
    identifier.record("shop", 0, board.shop[0], "Tiger")
    assert identifier.annotate(shop_board(shop_attack=5)).shop[0].species is None
    next_turn = Board(Phase.SHOP, gold=10, turn=2, shop=board.shop, team=board.team)
    assert identifier.annotate(next_turn).shop[0].species is None


def test_template_species_is_never_overridden():
    identifier = SpeciesIdentifier(LAYOUT, ocr=lambda image: "", sleep=lambda seconds: None)
    board = shop_board(team_species="pig")
    identifier.record("team", 0, board.team[0], "Rhino")
    assert identifier.annotate(board).team[0].species == "pig"
    assert ("team", 0) not in identifier.targets(board)


def test_identifier_requires_pointer_rest():
    profile = SimpleNamespace(validate=Mock(), pointer_rest=None)
    with pytest.raises(ValueError, match="pointer_rest"):
        DesktopRuntime(profile, Mock(), Mock(), identifier=SpeciesIdentifier(LAYOUT))


def test_tooltip_layout_file_must_match_profile_size(tmp_path):
    path = tmp_path / "tooltip.json"
    data = {"version": 1, "calibrated": True, "image_size": [200, 200],
            "layout": {"search_rect": [0, 0, 200, 200], "panel_width_range": [60, 100],
                       "panel_height_range": [40, 80], "title_offset_rect": [5, 5, 40, 10]}}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_tooltip_layout(path, (200, 200)).title_offset_rect == (5, 5, 40, 10)
    with pytest.raises(ValueError, match="aspect"):
        load_tooltip_layout(path, (300, 200))
    half = load_tooltip_layout(path, (100, 100))  # same aspect: scales with the profile
    assert half.search_rect == (0, 0, 100, 100) and half.title_offset_rect == (2, 2, 20, 5)
    data["calibrated"] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tooltip_layout(path, (200, 200))


def test_vocabulary_names_are_unique_and_valid():
    assert len(PET_NAMES) == len(set(PET_NAMES))
    SpeciesIdentifier(LAYOUT)  # read_pet_title accepts every vocabulary entry
    assert {"Pig", "Rhino", "Tiger", "Fish"} <= set(PET_NAMES)


def test_identify_keeps_names_read_on_a_new_turn():
    names = {"pointer": None}
    identifier = SpeciesIdentifier(LAYOUT, ocr=lambda image: names["pointer"], sleep=lambda seconds: None)
    board = Board(Phase.SHOP, gold=10, turn=4, shop=(PetSlot(True, attack=2, health=3),), team=(PetSlot(False),) * 5)
    result = identifier.identify(board, parked_frame(), points={("shop", 0): (15, 155)},
                                 hover=lambda point: names.update(pointer="TIGER"),
                                 capture=tooltip_frame, park=lambda: None)
    assert result.shop[0].species == "tiger"


def test_failed_hovers_record_their_reason_and_raw_title():
    driver, game, identifier = make_runtime(shop_board, {(15, 25): "WOMBAT"})
    driver.observe(), driver.observe()
    assert identifier.failures["title_unread"] == 2 and identifier.unread_titles["WOMBAT"] == 1


def test_levels_follow_acknowledged_buys_merges_and_sales():
    identifier = SpeciesIdentifier(LAYOUT, ocr=lambda image: "", sleep=lambda seconds: None)
    unread = Board(Phase.SHOP, gold=10, turn=2, shop=(PetSlot(False),),
                   team=(PetSlot(True, attack=3, health=4),) + (PetSlot(False),) * 4)
    assert identifier.annotate(unread).team[0].level is None  # history unknown
    identifier.acknowledged(Action("buy", 0, 0))
    assert identifier.annotate(unread).team[0].level == 1
    identifier.acknowledged(Action("merge", 0, 0))
    identifier.acknowledged(Action("merge", 0, 0))
    assert identifier.annotate(unread).team[0].level == 2
    read = replace(unread, team=(PetSlot(True, attack=3, health=4, level=3),) + (PetSlot(False),) * 4)
    assert identifier.annotate(read).team[0].level == 3  # a visible level wins
    identifier.acknowledged(Action("sell", 0))
    assert identifier.annotate(unread).team[0].level is None


def test_unread_trophies_fall_back_to_the_whole_label(monkeypatch):
    import sap_ai.desktop.runtime as runtime_module
    calls = []
    monkeypatch.setattr(runtime_module, "read_trophy_label", lambda frame, region: calls.append(1) or 4)
    driver, game, _ = make_runtime(shop_board, {})
    driver.profile.hud = {"wins": Rect(5, 5, 10, 10)}  # gold, turn, lives already read
    first, second = driver.observe(), driver.observe()
    assert first.wins == second.wins == 4 and len(calls) == 1  # cached per label image


def test_glyph_reader_reads_italic_ones_and_other_digits_separately():
    from sap_ai.desktop.runtime import read_hud_glyphs
    crop = np.full((40, 60, 3), 255, dtype=np.uint8)
    crop[5:35, 10:18] = 0   # a bar: "1"
    crop[5:35, 30:52] = 0   # a wide block: read by OCR
    crop[12:28, 36:46] = 255
    assert read_hud_glyphs(crop, ocr_glyph=lambda image: "0") == 10
    ones = np.full((40, 60, 3), 255, dtype=np.uint8)
    ones[5:35, 10:18] = 0
    ones[5:35, 30:38] = 0
    assert read_hud_glyphs(ones, ocr_glyph=lambda image: "") == 11
    assert read_hud_glyphs(crop, ocr_glyph=lambda image: "") is None  # unread glyph: unknown
    assert read_hud_glyphs(np.full((40, 60, 3), 255, dtype=np.uint8)) is None
