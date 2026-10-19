"""Whole click-sequence validation with mocked frames and input only."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from sap_ai.desktop.runtime import DesktopRuntime, DesktopUnavailable
from sap_ai.desktop.state import Action, Board, BoardChangedBeforeInput, FoodSlot, PetSlot, Phase
from sap_ai.desktop.vision import Rect


PHASES = {"choose_name": Phase.NAMING, "confirm_name": Phase.NAMING_READY,
          "continue_round": Phase.ROUND_RESULT, "dismiss_tier": Phase.TIER_UNLOCK,
          "dismiss_life_reward": Phase.LIFE_REWARD, "confirm_end_turn": Phase.END_TURN_CONFIRM,
          "open_play": Phase.MAIN_MENU, "open_arena": Phase.PLAY_MENU, "start_arena": Phase.ARENA_SETUP}
ACTIONS = {"buy": Action("buy", 0, 0), "merge": Action("merge", 0, 1),
           "buy_food": Action("buy_food", 0, 1), "sell": Action("sell", 2),
           **{name: Action(name) for name in ("roll", "end_turn", *PHASES)}}


def make_runtime(kind):
    observed = Board(Phase.SHOP, gold=10, turn=1, shop=(PetSlot(True, "fish", 2, 3, 1),),
                     team=(PetSlot(False), PetSlot(True, "fish", 3, 4, 1),
                           PetSlot(True, "duck", 2, 2, 1), PetSlot(False), PetSlot(False)),
                     food=(FoodSlot(True, "apple", 2, "single_target_stats", 1, 1),))
    if kind in PHASES:
        observed = Board(PHASES[kind])
    buttons = {name: (90, 90) for name in ("roll", "end_turn", *PHASES)}
    buttons.update(sell=(55, 90), name_adjective=(20, 30), name_noun=(70, 30))
    profile = SimpleNamespace(validate=Mock(), image_size=(100, 100), buttons=buttons,
                              shop=[SimpleNamespace(portrait=Rect(10, 60, 10, 10))],
                              food=[SimpleNamespace(portrait=Rect(60, 60, 10, 10))],
                              team=[SimpleNamespace(portrait=Rect(10 + i * 15, 20, 10, 10)) for i in range(5)])
    window = SimpleNamespace(capture=Mock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
                             click=Mock(), drag=Mock())
    perceptor = SimpleNamespace(observe=Mock(return_value=observed))
    runtime = DesktopRuntime(profile, perceptor, window, execute=True)
    runtime.last_board = observed
    runtime._pointer_dirty = False
    return runtime, window


@pytest.mark.parametrize("kind,points", [("buy", [(15, 65), (15, 25)]),
                                         ("merge", [(15, 65), (30, 25)]),
                                         ("buy_food", [(65, 65), (30, 25)]),
                                         ("sell", [(45, 25), (55, 90)]),
                                         ("choose_name", [(20, 30), (70, 30)]),
                                         *[(name, [(90, 90)]) for name in ("roll", "end_turn", *PHASES) if name != "choose_name"]])
def test_valid_action_sequences_retain_order_and_exact_client_points(kind, points):
    runtime, window = make_runtime(kind)
    runtime.act(ACTIONS[kind])
    assert window.click.call_args_list == [call(point, (100, 100)) for point in points]
    assert runtime._pointer_dirty is True
    window.drag.assert_not_called()


@pytest.mark.parametrize("kind", ["buy", "merge", "buy_food"])
@pytest.mark.parametrize("problem", ["missing_slot", "missing_portrait", "untyped_region", "outside", "negative", "empty"])
def test_bad_later_team_destination_sends_zero_input(kind, problem):
    runtime, window = make_runtime(kind)
    index = ACTIONS[kind].target
    if problem == "missing_slot":
        runtime.profile.team = runtime.profile.team[:index]
    elif problem == "missing_portrait":
        del runtime.profile.team[index].portrait
    else:
        runtime.profile.team[index].portrait = {"untyped_region": SimpleNamespace(center=(25, 25)),
                                               "outside": Rect(100, 20, 10, 10),
                                               "negative": Rect(-1, 20, 10, 10),
                                               "empty": Rect(10, 20, 0, 10)}[problem]
    with pytest.raises(ValueError):
        runtime.act(ACTIONS[kind])
    window.click.assert_not_called()
    window.drag.assert_not_called()
    assert runtime._pointer_dirty is False


@pytest.mark.parametrize("kind,button", [("sell", "sell"), ("choose_name", "name_noun")])
@pytest.mark.parametrize("point", [None, (70,), (70, 30, 1), (True, 30), (70, False),
                                    (70.0, 30), (70, np.int64(30)), ("70", 30), {"x": 70, "y": 30},
                                    (-1, 30), (100, 30), (70, 100)])
def test_malformed_later_button_never_selects_the_first_item(kind, button, point):
    runtime, window = make_runtime(kind)
    runtime.profile.buttons[button] = point
    with pytest.raises(ValueError, match="input point"):
        runtime.act(ACTIONS[kind])
    window.click.assert_not_called()
    assert runtime._pointer_dirty is False


@pytest.mark.parametrize("kind,button,message", [("sell", "sell", "calibrated sell point"),
                                                ("choose_name", "name_noun", "both calibrated name options")])
def test_missing_later_button_preserves_useful_error_and_sends_zero_input(kind, button, message):
    runtime, window = make_runtime(kind)
    del runtime.profile.buttons[button]
    with pytest.raises(ValueError, match=message):
        runtime.act(ACTIONS[kind])
    window.click.assert_not_called()


@pytest.mark.parametrize("kind", ["buy", "merge", "buy_food", "sell"])
@pytest.mark.parametrize("region", [None, Rect(True, 20, 10, 10), Rect(10, 20, -1, 10), SimpleNamespace(center=(20, 20))])
def test_source_geometry_also_requires_valid_typed_rect(kind, region):
    runtime, window = make_runtime(kind)
    row = "food" if kind == "buy_food" else "team" if kind == "sell" else "shop"
    getattr(runtime.profile, row)[ACTIONS[kind].slot].portrait = region
    with pytest.raises(ValueError):
        runtime.act(ACTIONS[kind])
    window.click.assert_not_called()


def test_single_button_coordinates_are_validated_before_any_input():
    runtime, window = make_runtime("roll")
    runtime.profile.buttons["roll"] = (True, 50)
    with pytest.raises(ValueError, match="integer client coordinates"):
        runtime.act(ACTIONS["roll"])
    window.click.assert_not_called()


def test_validated_mutable_button_is_frozen_before_first_click():
    runtime, window = make_runtime("choose_name")
    noun = [70, 30]
    runtime.profile.buttons["name_noun"] = noun
    def mutate_later_destination(*_):
        noun[:] = [-1, -1]
        runtime.profile.buttons.pop("name_noun", None)
    window.click.side_effect = mutate_later_destination
    runtime.act(ACTIONS["choose_name"])
    assert window.click.call_args_list == [call((20, 30), (100, 100)), call((70, 30), (100, 100))]


@pytest.mark.parametrize("kind", ["sell", "choose_name"])
def test_stop_after_selection_prevents_second_click_without_retry(kind):
    runtime, window = make_runtime(kind)
    stopped = [False]
    runtime.should_stop = lambda: stopped[0]
    window.click.side_effect = lambda *_: stopped.__setitem__(0, True)
    with pytest.raises(DesktopUnavailable, match="stop requested"):
        runtime.act(ACTIONS[kind])
    assert window.click.call_count == 1 and runtime._pointer_dirty is True


def test_fresh_changed_board_still_defers_before_geometry_validation_and_input():
    runtime, window = make_runtime("buy")
    changed = replace(runtime.last_board, gold=9)
    runtime.perceptor.observe.return_value = changed
    runtime.profile.team = []
    with pytest.raises(BoardChangedBeforeInput) as caught:
        runtime.act(ACTIONS["buy"])
    assert caught.value.preflight_board is changed
    window.click.assert_not_called()
    assert runtime._pointer_dirty is False
