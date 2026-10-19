"""Food schema foundation only: no native mechanics, inputs or policy support.

Apple below is an allowlisted protocol identity, with deliberately arbitrary
explicit prices/bonuses. These examples establish no current Apple balance,
stat cap, ability interaction, calibrated food geometry or native receipt.
"""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from sap_ai.desktop.state import Action, Board, DesktopPolicy, FoodSlot, PetSlot, Phase, legal_action
from tools.export_desktop_replay import export_session


EMPTY_PET = PetSlot(False)


def observed_food(**changes):
    return FoodSlot(**{"occupied": True, "food_id": "apple", "cost": 2,
                       "effect": "single_target_stats", "attack_bonus": 2,
                       "health_bonus": 0, **changes})


def shop_board(**changes):
    return Board(**{"phase": Phase.SHOP, "gold": 2, "turn": 4,
                    "shop": (PetSlot(False),),
                    "team": (PetSlot(True, None, 4, 5, None), *(EMPTY_PET,) * 4),
                    "food": (observed_food(),), **changes})


def test_food_readings_are_frozen_normalized_and_never_gain_inferred_balance():
    food = FoodSlot(True, "  APPLE  ", effect=" SINGLE_TARGET_STATS ")
    assert food.food_id == "apple" and food.effect == "single_target_stats"
    assert food.cost is food.attack_bonus is food.health_bonus is None
    assert not legal_action(shop_board(food=(food,)), Action("buy_food", 0, 0))
    with pytest.raises(FrozenInstanceError):
        food.cost = 3


@pytest.mark.parametrize("data", [
    {"occupied": 0}, {"occupied": "unknown"}, {"occupied": True, "food_id": " "},
    {"occupied": True, "food_id": 1}, {"occupied": True, "effect": " "},
    {"occupied": True, "effect": "random_stats"}, {"occupied": True, "effect": True},
    {"occupied": True, "cost": 0}, {"occupied": True, "cost": -1},
    {"occupied": True, "cost": True}, {"occupied": True, "cost": 1.5},
    {"occupied": True, "attack_bonus": -1}, {"occupied": True, "health_bonus": -1},
    {"occupied": True, "attack_bonus": True}, {"occupied": True, "health_bonus": 0.5},
])
def test_invalid_food_contract_types_and_unsupported_effects_fail_at_parse(data):
    with pytest.raises(ValueError):
        FoodSlot(**data)


@pytest.mark.parametrize("field,value", [("food_id", "apple"), ("cost", 2),
    ("effect", "single_target_stats"), ("attack_bonus", 0), ("health_bonus", 1)])
def test_empty_food_slot_cannot_retain_even_zero_valued_metadata(field, value):
    with pytest.raises(ValueError, match="empty slot"):
        FoodSlot(False, **{field: value})


def test_uncalibrated_unknown_and_empty_food_observations_are_distinct():
    board = shop_board(food=())
    unknown = replace(board, food=(FoodSlot(None),))
    empty = replace(board, food=(FoodSlot(False),))
    assert "food" not in board.to_dict()
    assert unknown.to_dict()["food"][0]["occupied"] is None
    assert empty.to_dict()["food"][0]["occupied"] is False
    assert len({board.fingerprint(), unknown.fingerprint(), empty.fingerprint()}) == 3
    for observation in (board, unknown, empty):
        assert not legal_action(observation, Action("buy_food", 0, 0))


def test_partial_readings_can_survive_unknown_occupancy_without_becoming_actionable():
    food = observed_food(occupied=None)
    board = shop_board(food=(food,))
    assert Board.from_dict(board.to_dict()) == board
    assert not legal_action(board, Action("buy_food", 0, 0))


def test_food_roundtrip_normalizes_lists_and_fingerprints_every_contract_field():
    board = shop_board(food=[observed_food(), FoodSlot(None)])
    assert isinstance(board.food, tuple)
    assert Board.from_dict(json.loads(json.dumps(board.to_dict()))) == board
    for changes in ({"cost": 3}, {"attack_bonus": 1}, {"health_bonus": 1},
                    {"food_id": "unknown item"}, {"effect": None}, {"occupied": None}):
        changed = replace(board, food=(replace(board.food[0], **changes), board.food[1]))
        assert changed.fingerprint() != board.fingerprint()


@pytest.mark.parametrize("food", [None, "apple", {}, [PetSlot(False)], [0], [FoodSlot(None)] * 6])
def test_board_rejects_untyped_or_unbounded_food_geometry(food):
    with pytest.raises(ValueError):
        shop_board(food=food)


@pytest.mark.parametrize("data", [
    {"food": None}, {"food": "apple"}, {"food": {}},
    {"food": [{"occupied": True, "unexpected": "metadata"}]},
    {"food": [{"occupied": True, "cost": "2"}]},
])
def test_food_deserialization_rejects_malformed_or_unknown_schema_fields(data):
    with pytest.raises((ValueError, TypeError)):
        Board.from_dict({"phase": "shop", **data})


def test_every_legacy_typed_fixture_board_keeps_exact_serialization_and_fingerprint():
    count = 0
    for path in sorted((Path(__file__).parent / "fixtures").glob("desktop_*.json")):
        for row in json.loads(path.read_text())["boards"]:
            assert "food" not in row
            board = Board.from_dict(row)
            # Some old fixtures omit optional pet fields. Legacy serialization
            # already supplied their None values; retain that exact shape.
            old_keys = ("phase", "gold", "turn", "wins", "lives", "shop", "team")
            pet_keys = ("occupied", "species", "attack", "health", "level")
            old_serialized = {key: row.get(key) for key in old_keys}
            for key in ("shop", "team"):
                old_serialized[key] = [{field: p.get(field) for field in pet_keys}
                                       for p in row.get(key, [])]
            assert board.to_dict() == old_serialized
            assert tuple(board.to_dict()) == old_keys
            expected = (row["phase"], row.get("gold"), row.get("turn"), row.get("wins"), row.get("lives"),
                        tuple(tuple(p.get(field) for field in pet_keys) for p in row.get("shop", [])),
                        tuple(tuple(p.get(field) for field in pet_keys) for p in row.get("team", [])))
            assert board.fingerprint() == expected
            count += 1
    assert count > 400


def test_explicit_empty_food_list_is_legacy_uncalibrated_and_serializes_without_key():
    board = Board.from_dict({"phase": "shop", "food": []})
    assert board == Board(Phase.SHOP)
    assert "food" not in board.to_dict()
    assert board.fingerprint() == Board(Phase.SHOP).fingerprint()


@pytest.mark.parametrize("cost,gold,attack,health,expected", [
    (1, 1, 7, 0, True), (2, 2, 0, 4, True), (4, 9, 2, 3, True),
    (4, 3, 2, 3, False), (None, 9, 2, 3, False), (2, None, 2, 3, False),
    (2, 9, None, 3, False), (2, 9, 2, None, False), (2, 9, 0, 0, False),
])
def test_food_legality_uses_explicit_cost_and_complete_positive_effect(cost, gold, attack, health, expected):
    food = observed_food(cost=cost, attack_bonus=attack, health_bonus=health)
    assert legal_action(shop_board(gold=gold, food=(food,)), Action("buy_food", 0, 0)) is expected


@pytest.mark.parametrize("food", [observed_food(food_id=None), observed_food(food_id="unknown item"),
    observed_food(food_id="chocolate"), observed_food(effect=None), FoodSlot(True),
    FoodSlot(False), observed_food(occupied=None)])
def test_known_positive_numbers_never_substitute_for_food_identity_and_effect(food):
    assert not legal_action(shop_board(food=(food,)), Action("buy_food", 0, 0))


@pytest.mark.parametrize("target", [PetSlot(None), EMPTY_PET,
    PetSlot(True, attack=None, health=5), PetSlot(True, attack=4, health=None)])
def test_food_requires_an_occupied_target_with_both_stats_readable(target):
    board = shop_board(team=(target, *(EMPTY_PET,) * 4))
    assert not legal_action(board, Action("buy_food", 0, 0))


@pytest.mark.parametrize("action", [Action("buy_food"), Action("buy_food", 0),
    Action("buy_food", target=0), Action("buy_food", 1, 0), Action("buy_food", 0, 4)])
def test_food_slot_and_target_selection_must_exist_and_match_food_geometry(action):
    assert not legal_action(shop_board(), action)


@pytest.mark.parametrize("board", [shop_board(phase=Phase.BATTLE), shop_board(shop=()),
    shop_board(shop=(PetSlot(None),)), shop_board(team=(PetSlot(True, attack=4, health=5),) * 4),
    shop_board(team=(PetSlot(True, attack=4, health=5), PetSlot(None), *(EMPTY_PET,) * 3))])
def test_food_never_bypasses_existing_known_shop_geometry_guard(board):
    assert not legal_action(board, Action("buy_food", 0, 0))


def test_food_index_is_independent_from_pet_shop_index():
    board = shop_board(food=(FoodSlot(False), observed_food()),
                       team=(*(EMPTY_PET,) * 4, PetSlot(True, None, 4, 5)))
    action = Action("buy_food", 1, 4)
    assert legal_action(board, action)
    assert Action.from_dict(json.loads(json.dumps(action.to_dict()))) == action
    assert not legal_action(board, Action("buy", 1, 0))


def test_food_legality_does_not_infer_species_level_or_stat_caps():
    board = shop_board(team=(PetSlot(True, None, 50, 50, None), *(EMPTY_PET,) * 4))
    assert legal_action(board, Action("buy_food", 0, 0))
    # Readable metadata makes the protocol legal, not a demonstrated effect at
    # a game cap. Future receipt/runtime support must validate those mechanics.


def test_default_policy_never_selects_food_even_if_cheaper_and_large_effect():
    food = observed_food(cost=1, attack_bonus=20, health_bonus=20)
    fixtures = Path(__file__).parent / "fixtures"
    count = 0
    for path in sorted(fixtures.glob("desktop_*.json")):
        for row in json.loads(path.read_text())["boards"]:
            legacy = Board.from_dict(row)
            observed = replace(legacy, food=(food,))
            assert DesktopPolicy().choose_action(observed) == DesktopPolicy().choose_action(legacy)
            count += 1
    assert count > 400


def test_optional_food_data_roundtrips_through_exporter_without_unknown_metadata():
    board = shop_board()
    rows = [{"event": "started"},
            {"event": "observed", "poll": 1, "board": board.to_dict()},
            {"event": "finished", "poll": 1, "result": {
                "actions": 0, "acknowledgments": 0, "polls": 1, "reason": "max_polls",
                "pending_action": None, "last_board": board.to_dict(), "error": None}}]
    replay = export_session("\n".join(json.dumps(row) for row in rows))
    assert replay["boards"] == [board.to_dict()]
    bad = json.loads(json.dumps(rows))
    bad[1]["board"]["food"][0]["path"] = "private image path"
    with pytest.raises(TypeError):
        export_session("\n".join(json.dumps(row) for row in bad))
