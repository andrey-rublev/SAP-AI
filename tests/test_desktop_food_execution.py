"""Explicit food adapter contracts with mocked input; no native food evidence.

Prices and bonuses deliberately differ from any assumed game balance. This
suite does not establish stat caps, ability interactions or a native profile.
"""
from dataclasses import replace
from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from sap_ai.desktop.runtime import DesktopRuntime, DesktopUnavailable
from sap_ai.desktop.session import DesktopSession, action_acknowledged
from sap_ai.desktop.state import Action, Board, BoardChangedBeforeInput, FoodSlot, PetSlot, Phase, legal_action
from sap_ai.desktop.vision import Rect


EMPTY = PetSlot(False)
EMPTY_FOOD = FoodSlot(False)
APPLE = FoodSlot(True, "apple", 2, "single_target_stats", 2, 0)
OTHER = FoodSlot(True, "another food", 4)
BUY = Action("buy_food", 0, 1)


def before_board(**changes):
    return Board(**{"phase": Phase.SHOP, "gold": 5, "turn": 4,
                    "shop": (PetSlot(True, "fish", 2, 3, 1),),
                    "team": (EMPTY, PetSlot(True, "fish", 4, 5, 1),
                             PetSlot(True, "duck", 2, 2, 1), EMPTY, EMPTY),
                    "food": (APPLE, OTHER, EMPTY_FOOD), **changes})


def after_board(before, *, compacted=False):
    target = before.team[BUY.target]
    source = before.food[BUY.slot]
    target = replace(target, attack=target.attack + source.attack_bonus,
                     health=target.health + source.health_bonus)
    team = (*before.team[:BUY.target], target, *before.team[BUY.target + 1:])
    food = (OTHER, EMPTY_FOOD, EMPTY_FOOD) if compacted else (EMPTY_FOOD, *before.food[1:])
    return replace(before, gold=before.gold - source.cost, team=team, food=food)


@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("cost,attack,health", [(1, 0, 4), (2, 2, 0), (4, 3, 2)])
def test_food_receipt_requires_explicit_price_and_exact_gains(compacted, cost, attack, health):
    before = before_board(food=(replace(APPLE, cost=cost, attack_bonus=attack,
                                       health_bonus=health), OTHER, EMPTY_FOOD))
    assert action_acknowledged(before, after_board(before, compacted=compacted), BUY)


@pytest.mark.parametrize("change", [
    {"gold": 5}, {"gold": 2}, {"gold": None}, {"turn": 5}, {"turn": None},
    {"phase": Phase.BATTLE}, {"wins": 1}, {"lives": 1},
    {"shop": (EMPTY,)}, {"shop": (PetSlot(None),)},
    {"food": (APPLE, OTHER, EMPTY_FOOD)},  # No source removal.
    {"food": (EMPTY_FOOD, EMPTY_FOOD, EMPTY_FOOD)},  # Two removals.
    {"food": (EMPTY_FOOD, OTHER)}, {"food": ()},
    {"food": (EMPTY_FOOD, FoodSlot(None), EMPTY_FOOD)},
    {"food": (EMPTY_FOOD, FoodSlot(True), EMPTY_FOOD)},
    {"food": (EMPTY_FOOD, replace(OTHER, cost=5), EMPTY_FOOD)},
    {"food": (OTHER, APPLE, EMPTY_FOOD)},  # Reorder rather than consumption.
    {"food": (OTHER, EMPTY_FOOD, APPLE)},  # New replacement offer is unsupported.
])
def test_food_receipt_rejects_unrelated_or_incomplete_changes(change):
    before = before_board()
    assert not action_acknowledged(before, replace(after_board(before), **change), BUY)


@pytest.mark.parametrize("field,value", [("attack", 4), ("attack", 7), ("attack", None),
    ("health", 4), ("health", 6), ("health", None), ("species", None),
    ("species", "duck"), ("level", None), ("level", 2), ("occupied", None)])
def test_food_target_requires_exact_stats_and_unchanged_identity_metadata(field, value):
    before = before_board()
    after = after_board(before)
    team = list(after.team)
    team[BUY.target] = replace(team[BUY.target], **{field: value})
    assert not action_acknowledged(before, replace(after, team=team), BUY)


def test_food_cannot_acknowledge_buffs_to_another_teammate_or_an_unknown_survivor():
    before = before_board()
    after = after_board(before)
    team = list(after.team)
    team[2] = replace(team[2], attack=3)
    assert not action_acknowledged(before, replace(after, team=team), BUY)
    before = before_board(food=(APPLE, FoodSlot(True), EMPTY_FOOD))
    assert not action_acknowledged(before, after_board(before), BUY)


def test_food_does_not_infer_cap_behavior_or_a_turn():
    before = before_board(team=(EMPTY, PetSlot(True, None, 50, 50), EMPTY, EMPTY, EMPTY))
    capped = replace(after_board(before), team=before.team)
    assert not action_acknowledged(before, capped, BUY)
    before = before_board(turn=None)
    assert not action_acknowledged(before, after_board(before), BUY)


@pytest.mark.parametrize("change", [{"turn": None}, {"turn": 0},
    {"food": (APPLE, FoodSlot(None), EMPTY_FOOD)},
    {"food": (APPLE, FoodSlot(True), EMPTY_FOOD)}])
def test_food_that_cannot_produce_an_item_removal_receipt_is_rejected_before_input(change):
    before = before_board(**change)
    assert not legal_action(before, BUY)
    runtime, window = make_runtime(before)
    with pytest.raises(ValueError, match="not legal"):
        runtime.act(BUY)
    window.capture.assert_not_called()
    window.click.assert_not_called()


@pytest.mark.parametrize("food", [replace(APPLE, effect=None), replace(APPLE, cost=None),
    replace(APPLE, food_id="unknown"), replace(APPLE, attack_bonus=None), EMPTY_FOOD])
def test_incomplete_or_unsupported_contract_is_not_a_receipt(food):
    before = before_board()
    after = after_board(before)
    assert not action_acknowledged(replace(before, food=(food, OTHER, EMPTY_FOOD)), after, BUY)


def make_runtime(before=None):
    observed = before or before_board()
    profile = SimpleNamespace(validate=Mock(), image_size=(100, 100), buttons={},
        phase_templates=(), shop=[SimpleNamespace(portrait=Rect(5, 60, 10, 10))],
        food=[SimpleNamespace(portrait=Rect(60 + i * 12, 60, 10, 10)) for i in range(3)],
        team=[SimpleNamespace(portrait=Rect(5 + i * 15, 20, 10, 10)) for i in range(5)])
    window = SimpleNamespace(capture=Mock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
                             click=Mock(), drag=Mock())
    perceptor = SimpleNamespace(observe=Mock(return_value=observed))
    runtime = DesktopRuntime(profile, perceptor, window, execute=True)
    runtime.last_board = observed
    return runtime, window


def test_food_input_uses_food_geometry_then_the_selected_teammate():
    runtime, window = make_runtime()
    runtime.act(BUY)
    assert window.click.call_args_list == [call((65, 65), (100, 100)), call((25, 25), (100, 100))]
    window.drag.assert_not_called()


@pytest.mark.parametrize("missing", ["food", "target"])
def test_food_validates_both_calibrated_points_before_any_input(missing):
    runtime, window = make_runtime()
    if missing == "food":
        del runtime.profile.food
    else:
        runtime.profile.team = runtime.profile.team[:1]
    with pytest.raises(ValueError, match="calibrated food source and team target"):
        runtime.act(BUY)
    window.click.assert_not_called()


@pytest.mark.parametrize("row,index", [("food", 0), ("team", 1)])
@pytest.mark.parametrize("region", [Rect(200, 20, 10, 10), Rect(5, -1, 10, 10),
    Rect(5, 20, 0, 10), Rect(True, 20, 10, 10), SimpleNamespace(center=(25, 25))])
def test_mutated_or_malformed_food_sequence_geometry_rejects_before_source_selection(row, index, region):
    runtime, window = make_runtime()
    getattr(runtime.profile, row)[index].portrait = region
    with pytest.raises(ValueError):
        runtime.act(BUY)
    window.click.assert_not_called()


@pytest.mark.parametrize("failed_click", [1, 2])
def test_food_click_failure_propagates_without_a_retry(failed_click):
    runtime, window = make_runtime()
    window.click.side_effect = [None] * (failed_click - 1) + [DesktopUnavailable("focus lost")]
    with pytest.raises(DesktopUnavailable, match="focus lost"):
        runtime.act(BUY)
    assert window.click.call_count == failed_click


def test_stop_between_food_selection_and_target_prevents_second_click():
    runtime, window = make_runtime()
    stopped = [False]
    runtime.should_stop = lambda: stopped[0]
    window.click.side_effect = lambda *_: stopped.__setitem__(0, True)
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        runtime.act(BUY)
    window.click.assert_called_once_with((65, 65), (100, 100))


def test_food_preflight_rejects_changed_price_before_input():
    runtime, window = make_runtime()
    runtime.perceptor.observe.return_value = replace(runtime.last_board,
        food=(replace(APPLE, cost=3), OTHER, EMPTY_FOOD))
    with pytest.raises(BoardChangedBeforeInput):
        runtime.act(BUY)
    window.click.assert_not_called()


@pytest.mark.parametrize("valid", [False, True])
def test_full_food_session_acknowledges_only_the_receipt_and_never_reclicks(valid):
    before = before_board()
    after = after_board(before) if valid else replace(before, gold=3)
    runtime, window = make_runtime(before)
    # Includes the last-moment preflight capture, which must still see before.
    runtime.perceptor.observe.side_effect = chain([before, before, before], repeat(after))
    calls = []
    class Policy:
        def choose_action(self, _):
            calls.append(BUY)
            return BUY
    runner = DesktopSession(runtime.observe, runtime.act, Policy(), max_actions=1,
        max_polls=10, action_max_polls=3, sleep=lambda _: None, clock=lambda: 0.)
    result = runner.run()
    assert len(calls) == 1 and window.click.call_count == 2
    assert result.actions == 1 and result.acknowledgments == int(valid)
    assert result.reason == ("max_actions" if valid else "action_timeout")
    assert result.pending_action == (None if valid else BUY)


@pytest.mark.parametrize("failed_click", [1, 2])
def test_food_partial_input_failure_keeps_the_action_pending_in_the_session(failed_click):
    before = before_board()
    runtime, window = make_runtime(before)
    window.click.side_effect = [None] * (failed_click - 1) + [DesktopUnavailable("focus lost")]
    policy = SimpleNamespace(choose_action=Mock(return_value=BUY))
    result = DesktopSession(runtime.observe, runtime.act, policy, max_polls=10,
        sleep=lambda _: None, clock=lambda: 0.).run()
    assert result.reason == "action_error"
    assert result.actions == 1 and result.acknowledgments == 0
    assert result.pending_action == BUY and "focus lost" in result.error
    assert window.click.call_count == failed_click
    policy.choose_action.assert_called_once()
