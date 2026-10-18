"""Opt-in real-Board replacement searches; no combat or desktop simulation."""
from dataclasses import replace
from itertools import chain, repeat

import pytest

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase, legal_action


EMPTY = PetSlot(False)
WEAK = PetSlot(True, "fish", 1, 1, 1)
OFFER = PetSlot(True, None, 0, 1, None)


def board(**changes):
    return Board(**{"phase": Phase.SHOP, "gold": 3, "turn": 4,
                    "shop": (OFFER,), "team": (WEAK,) * 5, **changes})


@pytest.mark.parametrize("gold,species,level,expected", [
    (3, None, 1, "roll"), (2, None, 1, "end_turn"), (1, None, 1, "end_turn"),
    (0, None, 1, "end_turn"), (2, "pig", 1, "roll"), (1, "pig", 1, "end_turn"),
    (3, "pig", None, "end_turn"), (3, "pig", 2, "end_turn"),
    (3, "fish", None, "end_turn"), (3, "fish", 2, "end_turn"), (3, "fish", 3, "end_turn"),
])
def test_only_supported_known_level_one_sale_income_can_fund_the_post_roll_purchase(gold, species, level, expected):
    team = (PetSlot(True, species, 1, 1, level), *(replace(WEAK, level=3),) * 4)
    observed = board(gold=gold, team=team)
    action = DesktopPolicy(sale_funded_rolls=True).choose_action(observed)
    assert action == Action(expected)
    assert legal_action(observed, action)
    assert DesktopPolicy().choose_action(observed) == Action("end_turn")


def test_default_and_explicit_disabled_policy_keep_the_old_roll_reserve():
    observed = board()
    assert DesktopPolicy().choose_action(observed) == Action("end_turn")
    assert DesktopPolicy(sale_funded_rolls=False).choose_action(observed) == Action("end_turn")
    assert DesktopPolicy(sale_funded_rolls=True).choose_action(observed) == Action("roll")


@pytest.mark.parametrize("bad", [None, 0, 1, "true", [], {}])
def test_opt_in_requires_an_explicit_boolean(bad):
    with pytest.raises(ValueError, match="boolean"):
        DesktopPolicy(sale_funded_rolls=bad)


def test_empty_team_space_does_not_plan_a_sale_to_fund_a_roll():
    observed = board(team=(EMPTY, *board().team[1:]))
    assert DesktopPolicy(sale_funded_rolls=True).choose_action(observed) == Action("buy", 0, 0)
    observed = replace(observed, gold=2)
    assert DesktopPolicy(sale_funded_rolls=True).choose_action(observed) == Action("end_turn")


@pytest.mark.parametrize("change", [{"phase": Phase.BATTLE}, {"gold": None},
    {"team": (PetSlot(None), *board().team[1:])},
    {"team": (replace(WEAK, health=None), *board().team[1:])},
    {"shop": (replace(OFFER, attack=None),)}])
def test_experimental_search_does_not_bypass_observation_guards(change):
    assert DesktopPolicy(sale_funded_rolls=True).choose_action(board(**change)) is None


def test_current_visible_upgrade_and_merge_take_priority_over_an_extra_search():
    strong = replace(OFFER, attack=5, health=5)
    assert DesktopPolicy(sale_funded_rolls=True).choose_action(board(shop=(strong,))) == Action("sell", 0)
    duplicate = replace(OFFER, species="fish", level=1)
    assert DesktopPolicy(sale_funded_rolls=True).choose_action(board(shop=(duplicate,))) == Action("merge", 0, 0)


def test_sale_funded_search_replans_roll_sale_and_purchase_with_real_receipts():
    before = board()
    strong = replace(OFFER, attack=4, health=3)
    rolled = replace(before, gold=2, shop=(strong,))
    sold = replace(rolled, gold=3, team=(EMPTY, *rolled.team[1:]))
    bought = replace(sold, gold=0, shop=(EMPTY,), team=(strong, *sold.team[1:]))
    frames = iter(chain([before] * 2, [rolled] * 3, [sold] * 3, repeat(bought)))
    actions = []
    result = DesktopSession(lambda: next(frames), actions.append, DesktopPolicy(sale_funded_rolls=True),
        max_actions=3, max_polls=15, sleep=lambda _: None, clock=lambda: 0.).run()
    assert actions == [Action("roll"), Action("sell", 0), Action("buy", 0, 0)]
    assert (result.actions, result.acknowledgments, result.reason) == (3, 3, "max_actions")
    assert result.pending_action is result.error is None
    assert result.last_board == bought


def test_a_disappointing_roll_never_queues_the_planned_sale():
    rolled = replace(board(), gold=2)
    assert DesktopPolicy(sale_funded_rolls=True).choose_action(rolled) == Action("end_turn")


def test_identified_level_one_pig_can_fund_two_fresh_searches_but_never_a_third():
    observed = board(team=(PetSlot(True, "pig", 1, 1, 1), *(replace(WEAK, level=3),) * 4))
    policy = DesktopPolicy(sale_funded_rolls=True)
    assert [policy.choose_action(replace(observed, gold=gold)) for gold in (3, 2, 1)] == [
        Action("roll"), Action("roll"), Action("end_turn"),
    ]
