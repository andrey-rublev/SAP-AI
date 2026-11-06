"""Offline arena on the desktop Board/Action schema."""
import random

import pytest

from sap_ai.desktop.state import Action, DesktopPolicy, Phase
from sap_ai.sim.arena import (SpeciesBlind, Shop, _Pet, battle, compare, lives_lost, max_tier,
                              play_run, shop_size)


def test_combat_trades_front_to_back_and_detects_draws():
    assert battle([_Pet("a", 3, 3)], [_Pet("b", 1, 1), _Pet("c", 1, 1)]) == 1
    assert battle([_Pet("a", 2, 2)], [_Pet("b", 2, 2)]) == 0
    assert battle([None, None], [_Pet("b", 1, 1)]) == -1


def test_turn_schedule_unlocks_tiers_slots_and_harsher_losses():
    assert [max_tier(t) for t in (1, 2, 3, 11, 20)] == [1, 1, 2, 6, 6]
    assert [shop_size(t) for t in (1, 5, 9)] == [3, 4, 5]
    assert [lives_lost(t) for t in (1, 3, 5)] == [1, 2, 3]


def test_shop_board_matches_the_desktop_schema_and_merges_level_up():
    shop = Shop(random.Random(0))
    shop.start_turn(1)
    shop.offers = [_Pet("fish", 2, 2), _Pet("fish", 2, 2), _Pet("ant", 2, 1)]
    board = shop.board(wins=0, lives=5)
    assert board.phase == Phase.SHOP and len(board.shop) == 3 and len(board.team) == 5
    assert shop.apply(board, Action("buy", 0, 0))
    board = shop.board(0, 5)
    assert board.shop[0].species == "fish" and shop.apply(board, Action("merge", 0, 0))
    assert shop.team[0].attack == 3 and shop.team[0].experience == 1 and shop.gold == 4
    assert not shop.apply(shop.board(0, 5), Action("end_turn"))


def test_runs_are_seeded_and_bounded():
    policy = DesktopPolicy()
    first = play_run(policy, DesktopPolicy.recorded_baseline(), seed=3)
    assert first == play_run(policy, DesktopPolicy.recorded_baseline(), seed=3)
    assert 0 <= first["trophies"] <= 10 and first["turns"] >= 1


def test_species_identification_is_worth_trophies_offline():
    report = compare({"seen": DesktopPolicy.recorded_baseline(),
                      "blind": SpeciesBlind(DesktopPolicy.recorded_baseline())}, runs=60)
    assert report["seen"]["mean_trophies"] > report["blind"]["mean_trophies"]



def test_zero_attack_fronts_draw_even_with_stronger_pets_behind_them():
    # Both lead pets survive forever under the ability-free combat rules.
    left = [_Pet("a", 0, 4), _Pet("b", 10, 10)]
    right = [_Pet("c", 0, 8), _Pet("d", 3, 3)]
    assert battle(left, right) == 0
    assert left[0].health == 4 and right[0].health == 8
    assert battle([_Pet("a", 0, 4)], [_Pet("b", 1, 1)]) == -1
    assert battle([_Pet("a", 1, 1)], [_Pet("b", 0, 4)]) == 1


@pytest.mark.parametrize("runs,seed", [(0, 0), (-1, 0), (True, 0), (1.5, 0), (1, -1), (1, False)])
def test_invalid_benchmark_counts_fail_before_evaluating_a_policy(runs, seed):
    class NeverEvaluate:
        def choose_action(self, board):
            pytest.fail("invalid parameters must fail before policy evaluation")

    with pytest.raises(ValueError):
        compare({"never": NeverEvaluate()}, runs=runs, seed=seed)


@pytest.mark.parametrize("args", [["--runs", "0"], ["--seed", "-1"]])
def test_invalid_benchmark_cli_has_a_clear_usage_error(args):
    from sap_ai.sim.arena import main
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
