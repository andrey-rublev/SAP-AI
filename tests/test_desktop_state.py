"""Desktop decisions are validated without reading or changing the screen."""
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from desktop_state import (Action, Board, BoardChangedBeforeInput, DesktopPolicy,
                           PetSlot, Phase, legal_action, observed_sale_income)


EMPTY = PetSlot(False)


def pet(attack=2, health=3, species=None, level=None):
    return PetSlot(True, species, attack, health, level)


def board(*, shop=None, team=None, gold=10, **kwargs):
    return Board(Phase.SHOP, gold=gold,
                 shop=shop if shop is not None else (pet(), EMPTY, EMPTY),
                 team=team if team is not None else (EMPTY,) * 5, **kwargs)


def test_roundtrip_preserves_unknowns_and_normalizes_collections():
    value = Board("shop", gold=3, shop=[PetSlot(None)], team=[EMPTY] * 5)
    restored = Board.from_dict(json.loads(json.dumps(value.to_dict())))
    assert restored == value
    assert restored.shop[0].occupied is None
    assert isinstance(restored.team, tuple)
    assert hash(restored.fingerprint()) == hash(value.fingerprint())
    assert replace(value, gold=2).fingerprint() != value.fingerprint()
    action = Action("buy", 0, 4)
    assert Action.from_dict(json.loads(json.dumps(action.to_dict()))) == action


def test_preinput_exception_preserves_optional_typed_evidence_and_legacy_message():
    legacy = BoardChangedBeforeInput("board changed")
    assert str(legacy) == "board changed" and legacy.preflight_board is None
    current = board(gold=9)
    evidence = BoardChangedBeforeInput("board changed", preflight_board=current)
    assert isinstance(evidence, RuntimeError) and evidence.preflight_board is current


@pytest.mark.parametrize("value", [{"phase": "shop"}, 0, "board", object()])
def test_preinput_exception_rejects_untyped_preflight_evidence(value):
    with pytest.raises(TypeError, match="preflight_board must be a Board"):
        BoardChangedBeforeInput("board changed", preflight_board=value)


def test_frozen_state_and_pet_identity():
    value = pet(species="  Green   ANT  ")
    assert value.species == "green ant"
    with pytest.raises(FrozenInstanceError):
        value.attack = 99


@pytest.mark.parametrize("kwargs", [
    {"occupied": 1}, {"occupied": False, "attack": 0},
    {"occupied": True, "attack": True}, {"occupied": True, "health": 0},
    {"occupied": True, "attack": -1}, {"occupied": True, "level": 4},
    {"occupied": True, "level": 0}, {"occupied": True, "species": " "},
])
def test_invalid_pet_metadata_rejected(kwargs):
    with pytest.raises(ValueError):
        PetSlot(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"gold": True}, {"wins": -1}, {"turn": 1.5},
    {"team": [EMPTY] * 6}, {"shop": [0]}, {"phase": "menu"},
])
def test_invalid_board_rejected(kwargs):
    with pytest.raises(ValueError):
        Board(**{"phase": Phase.UNKNOWN, **kwargs})


@pytest.mark.parametrize("action", [
    ("buy", -1, 0), ("sell", True, None), ("buy", 5, 0),
    ("buy", 0, 5), ("freeze", None, None),
])
def test_invalid_action_rejected(action):
    with pytest.raises(ValueError):
        Action(*action)


def test_buy_uses_attack_and_health_and_actual_empty_destination():
    value = board(shop=(pet(8, 1), pet(3, 10), EMPTY),
                  team=(pet(), pet(), pet(), EMPTY, pet()))
    action = DesktopPolicy().choose_action(value)
    assert action == Action("buy", 1, 3)
    assert legal_action(value, action)
    assert not legal_action(value, Action("buy", 1, 1))


@pytest.mark.parametrize("value", [
    Board(Phase.UNKNOWN), Board(Phase.BATTLE), Board(Phase.RESULT),
    board(gold=None), board(shop=(PetSlot(None), EMPTY, EMPTY)),
    board(team=(PetSlot(None),) + (EMPTY,) * 4),
    board(shop=(PetSlot(True, attack=2), EMPTY, EMPTY)),
    board(team=(PetSlot(True, health=2),) + (EMPTY,) * 4),
    board(team=(EMPTY,) * 4), board(shop=()),
])
def test_uncertain_or_inactive_board_never_gets_policy_action(value):
    assert DesktopPolicy().choose_action(value) is None


@pytest.mark.parametrize("gold,legal", [(0, False), (2, False), (3, True)])
def test_buy_requires_three_gold(gold, legal):
    assert legal_action(board(gold=gold), Action("buy", 0, 4)) is legal


def test_empty_and_missing_shop_slots_are_not_purchase_sources():
    value = board()
    assert not legal_action(value, Action("buy", 1, 4))
    assert not legal_action(value, Action("buy", 4, 4))
    assert not legal_action(value, Action("buy", 0))


def test_merge_uses_known_species_and_level_and_respects_maximum_level():
    value = board(shop=(pet(species="FISH", level=1), EMPTY, EMPTY),
                  team=(pet(species="ant", level=1), pet(species="fish", level=3),
                        pet(species="fish", level=2), pet(), pet()))
    assert DesktopPolicy().choose_action(value) == Action("merge", 0, 2)
    assert not legal_action(value, Action("merge", 0, 0))
    assert not legal_action(value, Action("merge", 0, 1))
    assert legal_action(value, Action("merge", 0, 2))


@pytest.mark.parametrize("source,target", [
    (pet(level=1), pet(level=1)),
    (pet(species="fish"), pet(species="fish", level=1)),
    (pet(species="fish", level=1), pet(species="fish")),
    (pet(species="fish", level=2), pet(species="fish", level=1)),
    (pet(species="fish", level=1), EMPTY),
])
def test_equal_stats_never_establish_merge_compatibility(source, target):
    value = board(shop=(source, EMPTY, EMPTY), team=(target,) + (pet(),) * 4)
    assert not legal_action(value, Action("merge", 0, 0))


def test_fill_team_before_merging_duplicate():
    value = board(shop=(pet(species="fish", level=1), EMPTY, EMPTY),
                  team=(pet(species="fish", level=1), EMPTY, EMPTY, EMPTY, EMPTY))
    assert DesktopPolicy().choose_action(value) == Action("buy", 0, 1)


def test_replacement_is_a_sale_then_observed_purchase_and_never_sell_loop():
    policy = DesktopPolicy()
    offered = pet(5, 5, "fish", 1)
    before = board(gold=3, shop=(offered, EMPTY, EMPTY),
                   team=(pet(4, 4, "ant", 1), pet(1, 1, "beaver", 1),
                         pet(5, 5, "otter", 1), pet(4, 4, "pig", 1), pet(4, 4, "cricket", 1)))
    assert policy.choose_action(before) == Action("sell", 1)
    after_sale = replace(before, gold=4, team=(before.team[0], EMPTY, *before.team[2:]))
    assert policy.choose_action(after_sale) == Action("buy", 0, 1)
    after_buy = replace(after_sale, gold=1, shop=(EMPTY,) * 3,
                        team=(before.team[0], offered, *before.team[2:]))
    assert policy.choose_action(after_buy) == Action("end_turn")


@pytest.mark.parametrize("level", [None, 2, 3])
def test_policy_does_not_discard_unidentified_or_upgraded_pets(level):
    value = board(gold=3, shop=(pet(20, 20), EMPTY, EMPTY),
                  team=(pet(1, 1, level=level),) * 5)
    assert DesktopPolicy().choose_action(value) == Action("end_turn")


def test_replacement_margin_and_budget_keep_small_upgrades_from_churning():
    value = board(gold=3, shop=(pet(3, 4), EMPTY, EMPTY),
                  team=(pet(2, 3, level=1),) * 5)
    assert DesktopPolicy().choose_action(value) == Action("end_turn")
    assert DesktopPolicy(minimum_upgrade_gain=2).choose_action(value) == Action("sell", 0)
    assert DesktopPolicy(minimum_upgrade_gain=2).choose_action(replace(value, gold=2)) == Action("sell", 0)


@pytest.mark.parametrize("value,income", [
    (pet(species="pig", level=1), 2),
    (pet(species="pig"), None), (pet(species="pig", level=2), None),
    (pet(species="pig", level=3), None),
    (pet(species="ant", level=1), 1), (pet(level=1), 1),
    (pet(species="fish", level=2), 1), (pet(), 1),
    (EMPTY, None), (PetSlot(None, "pig", level=1), None),
])
def test_sale_income_preserves_exact_supported_receipts(value, income):
    assert observed_sale_income(value) == income


@pytest.mark.parametrize("species,gold,expected", [
    ("ant", 2, "sell"), (None, 2, "sell"), ("pig", 1, "sell"),
    ("ant", 1, "end_turn"), (None, 1, "end_turn"), ("pig", 0, "end_turn"),
])
def test_replacement_requires_sale_receipt_to_fund_purchase(species, gold, expected):
    value = board(gold=gold, shop=(pet(5, 5),),
                  team=(pet(1, 1, species, 1),) + (pet(8, 8, level=2),) * 4)
    action = DesktopPolicy().choose_action(value)
    assert action == (Action("sell", 0) if expected == "sell" else Action("end_turn"))
    assert legal_action(value, action)


def test_replacement_selects_weakest_affordable_pet_before_comparing_gain():
    value = board(gold=1, shop=(pet(4, 4),),
                  team=(pet(1, 1, "ant", 1), pet(2, 2, "pig", 1))
                  + (pet(8, 8, level=2),) * 3)
    assert DesktopPolicy().choose_action(value) == Action("sell", 1)
    assert DesktopPolicy().choose_action(replace(value, shop=(pet(4, 3),))) == Action("end_turn")


def test_sale_funding_never_sells_when_team_already_has_empty_space():
    value = board(gold=2, shop=(pet(8, 8),),
                  team=(pet(1, 1, "pig", 1), EMPTY) + (pet(8, 8, level=2),) * 3)
    assert DesktopPolicy().choose_action(value) == Action("end_turn")
    assert DesktopPolicy().choose_action(replace(value, gold=3)) == Action("buy", 0, 1)


def test_funded_merge_keeps_priority_over_replacement():
    value = board(gold=3, shop=(pet(8, 8, "fish", 1),),
                  team=(pet(1, 1, "fish", 1),) + (pet(8, 8, level=2),) * 4)
    assert DesktopPolicy().choose_action(value) == Action("merge", 0, 0)


def test_recorded_turn_six_upgrade_is_affordable_after_observed_sale():
    path = Path(__file__).parent / "fixtures" / "desktop_turns_4_to_terminal.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    boards = [Board.from_dict(value) for value in data["boards"]]
    frames = [boards[index] for index, count in data["observation_runs"] for _ in range(count)]
    before = frames[215]
    assert before.phase == Phase.SHOP and (before.turn, before.gold) == (6, 2)
    assert (before.team[0].strength, before.shop[1].strength) == (5, 9)
    assert before.team[0].level == 1 and before.team[0].species is None
    assert next(row["action"] for row in data["actions"] if row["poll"] == 216) == {"kind": "end_turn"}
    policy = DesktopPolicy()
    assert policy.choose_action(before) == Action("sell", 0)
    after = replace(before, gold=3, team=(EMPTY, *before.team[1:]))
    assert policy.choose_action(after) == Action("buy", 1, 0)


@pytest.mark.parametrize("gold,expected", [(0, "end_turn"), (1, "end_turn"), (3, "end_turn"), (4, "roll")])
def test_roll_preserves_purchase_budget(gold, expected):
    value = board(gold=gold, shop=(EMPTY,) * 3)
    assert DesktopPolicy().choose_action(value) == Action(expected)
    assert legal_action(value, Action("roll")) is (gold >= 1)


def test_sell_requires_occupied_team_slot_and_actions_require_exact_shape():
    value = board(team=(EMPTY, pet(), EMPTY, EMPTY, EMPTY))
    assert legal_action(value, Action("sell", 1))
    for action in (Action("sell", 0), Action("sell"), Action("sell", 1, 2),
                   Action("roll", 0), Action("end_turn", target=1), Action("continue")):
        assert not legal_action(value, action)


@pytest.mark.parametrize("phase", [Phase.UNKNOWN, Phase.BATTLE, Phase.RESULT])
def test_nonshop_phases_never_authorize_any_action(phase):
    value = replace(board(), phase=phase)
    for action in (Action("buy", 0, 0), Action("roll"), Action("sell", 0),
                   Action("merge", 0, 0), Action("end_turn"), Action("continue")):
        assert not legal_action(value, action)


def test_uncertain_occupancy_blocks_validator_even_for_unrelated_target():
    value = board(team=(EMPTY, PetSlot(None), EMPTY, EMPTY, EMPTY))
    assert not legal_action(value, Action("buy", 0, 0))
    assert not legal_action(value, Action("end_turn"))


def test_every_policy_decision_in_representative_boards_is_legal():
    policy = DesktopPolicy()
    for gold in range(11):
        for team in ((EMPTY,) * 5, (pet(level=1),) * 5, (pet(), EMPTY, EMPTY, EMPTY, EMPTY)):
            value = board(gold=gold, team=team, shop=(pet(8, 9, level=1), pet(), EMPTY))
            assert legal_action(value, policy.choose_action(value))


@pytest.mark.parametrize("phase,kind", [
    (Phase.MAIN_MENU, "open_play"), (Phase.PLAY_MENU, "open_arena"),
    (Phase.ARENA_SETUP, "start_arena"),
    (Phase.NAMING, "choose_name"), (Phase.NAMING_READY, "confirm_name"),
    (Phase.ROUND_RESULT, "continue_round"), (Phase.TIER_UNLOCK, "dismiss_tier"),
    (Phase.LIFE_REWARD, "dismiss_life_reward"),
    (Phase.END_TURN_CONFIRM, "confirm_end_turn"),
])
def test_transition_action_requires_exact_phase_and_no_slots(phase, kind):
    for candidate in Phase:
        assert legal_action(Board(candidate), Action(kind)) is (candidate == phase)
    assert not legal_action(Board(phase), Action(kind, slot=0))
    assert not legal_action(Board(phase), Action(kind, target=0))
    assert DesktopPolicy().choose_action(Board(phase)) is None


def test_life_reward_phase_and_action_roundtrip():
    value = Board(Phase.LIFE_REWARD, turn=3, lives=4)
    data = json.loads(json.dumps(value.to_dict()))
    assert data["phase"] == "life_reward"
    assert Board.from_dict(data) == value
    action = Action("dismiss_life_reward")
    assert Action.from_dict(json.loads(json.dumps(action.to_dict()))) == action
