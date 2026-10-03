"""Desktop decisions are validated without reading or changing the screen."""
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase, legal_action


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
    assert DesktopPolicy(minimum_upgrade_gain=2).choose_action(replace(value, gold=2)) == Action("end_turn")


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
