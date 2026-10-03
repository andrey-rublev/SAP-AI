"""Economy experiments remain pure, paired, reproducible and explicitly bounded."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase, legal_action
from tools import eval_desktop_economy as economy


def pet(species="ant", attack=2, health=2, level=1):
    return PetSlot(True, species, attack, health, level)


def full_board(gold=3, shop=None):
    return Board(Phase.SHOP, gold=gold, turn=5,
                 shop=shop or (pet("beaver", 9, 9),),
                 team=(pet("ant", 1, 1),) + (pet("pig", 8, 8, 3),) * 4)


def test_baselines_are_actual_production_policy_instances():
    assert type(economy.make_policy("baseline")) is DesktopPolicy
    assert type(economy.make_policy("threshold_2")) is DesktopPolicy
    assert economy.make_policy("threshold_6").minimum_upgrade_gain == 6
    assert isinstance(economy.make_policy("combined_4"), DesktopPolicy)


def test_offer_stream_is_keyed_by_roll_index_not_prior_rng_consumption():
    stream = economy.OfferStream(2718)
    expected = stream.offers(3)
    stream.offers(999)
    stream.levelup_offer(0)
    stream.rng("unrelated").random()
    assert stream.offers(3) == expected == economy.OfferStream(2718).offers(3)
    assert stream.offers(0) == stream.initial.shop
    assert stream.offers(2) != expected


def test_all_variants_see_same_immutable_initial_board_and_replay():
    first = economy.run_pair(50, trace=True)
    assert economy.run_pair(50, trace=True) == first
    starts = [value["trace"][0]["board"] for value in first.values()]
    assert all(board == starts[0] for board in starts)


def test_sale_funds_purchase_at_two_gold_in_baseline_and_candidate():
    board = full_board(gold=2)
    assert DesktopPolicy().choose_action(board) == Action("sell", 0)
    policy = economy.EconomyPolicy(sale_funded=True)
    assert policy.choose_action(board) == Action("sell", 0)
    after_sale = replace(board, gold=3, team=(economy.EMPTY, *board.team[1:]))
    assert policy.choose_action(after_sale) == Action("buy", 0, 0)
    assert legal_action(board, policy.choose_action(board))


@pytest.mark.parametrize("name", list(economy.VARIANTS))
def test_every_variant_chooses_affordable_pig_over_weaker_unaffordable_pet(name):
    board = replace(full_board(gold=1),
                    team=(pet("ant", 1, 1), pet("pig", 2, 2), *(pet("pig", 8, 8, 3),) * 3))
    policy = economy.make_policy(name)
    assert policy.choose_action(board) == Action("sell", 1)
    after_sale = replace(board, gold=3, team=(board.team[0], economy.EMPTY, *board.team[2:]))
    assert policy.choose_action(after_sale) == Action("buy", 0, 1)
    assert policy.choose_action(replace(board, gold=0)) == Action("end_turn")


@pytest.mark.parametrize("level", [None, 2, 3])
def test_custom_policy_abstains_from_unsupported_pig_sales(level):
    board = replace(full_board(), team=(pet("pig", 1, 1, level), *(pet("pig", 8, 8, 3),) * 4))
    assert DesktopPolicy().choose_action(board) == Action("end_turn")
    assert economy.EconomyPolicy(sale_funded=True, rank_gains=True).choose_action(board) == Action("end_turn")


def test_sale_funded_roll_can_leave_two_gold_but_requires_replaceable_pet():
    board = full_board(shop=(pet("beaver", 1, 1),))
    policy = economy.EconomyPolicy(sale_funded=True)
    assert DesktopPolicy().choose_action(board) == Action("end_turn")
    assert policy.choose_action(board) == Action("roll")
    assert policy.choose_action(replace(board, gold=2)) == Action("end_turn")
    protected = replace(board, team=tuple(replace(p, level=3) for p in board.team))
    assert policy.choose_action(protected) == Action("end_turn")


def test_ranked_candidate_compares_replacement_against_merge_net_gold_gain():
    fish = pet("fish", 2, 3)
    board = replace(full_board(), shop=(fish, pet("beaver", 9, 9)),
                    team=(fish, pet("ant", 1, 1), *(pet("pig", 8, 8, 3),) * 3))
    assert DesktopPolicy().choose_action(board) == Action("merge", 0, 0)
    assert economy.EconomyPolicy(rank_gains=True).choose_action(board) == Action("sell", 1)


def test_gain_ranking_picks_best_matching_merge_destination():
    fish = pet("fish", 8, 8)
    board = full_board(shop=(fish,))
    board = replace(board, team=(pet("fish", 12, 12, 2), pet("fish", 1, 1, 2), *(pet("pig", 8, 8, 3),) * 3))
    assert DesktopPolicy().choose_action(board) == Action("merge", 0, 0)
    assert economy.EconomyPolicy(rank_gains=True).choose_action(board) == Action("merge", 0, 1)


@pytest.mark.parametrize("board", [Board(Phase.UNKNOWN), replace(full_board(), gold=None),
                                  replace(full_board(), team=(PetSlot(None), *full_board().team[1:])),
                                  replace(full_board(), shop=(PetSlot(True, attack=2),))])
def test_candidate_keeps_parent_unknown_observation_guard(board):
    assert economy.EconomyPolicy(sale_funded=True, rank_gains=True).choose_action(board) is None


def test_fill_empty_slot_precedes_speculative_economy_changes():
    board = replace(full_board(), team=(economy.EMPTY, *full_board().team[1:]))
    assert economy.EconomyPolicy(sale_funded=True, rank_gains=True).choose_action(board) == Action("buy", 0, 0)


@pytest.mark.parametrize("species,gold,income,net_spent", [("ant", 2, 1, 2), ("pig", 1, 2, 1),
                                                         (None, 2, 1, 2)])
def test_shop_transition_accounts_for_sale_income_purchase_cost_and_stat_gain(species, gold, income, net_spent):
    initial = replace(full_board(gold=gold), team=(pet(species, 1, 1), *full_board().team[1:]))
    stream = SimpleNamespace(initial=initial, copies=(1, 6, 6, 6, 6), size=1)
    result = economy.run_shop(stream, DesktopPolicy(), trace=True)
    assert result["reason"] == "end_turn" and not result["violations"]
    assert [event["action"]["kind"] for event in result["trace"]] == ["sell", "buy", "end_turn"]
    assert (result["gross_spent"], result["sale_income"], result["net_spent"], result["leftover_gold"]) == (3, income, net_spent, 0)
    assert result["stat_gain"] == 16


@pytest.mark.parametrize("level", [None, 2, 3])
def test_simulator_rejects_unsupported_pig_sale_without_mutating_board(level):
    initial = replace(full_board(), team=(pet("pig", 1, 1, level), *full_board().team[1:]))
    stream = SimpleNamespace(initial=initial, copies=(1, 6, 6, 6, 6), size=1)
    forced_sale = SimpleNamespace(choose_action=lambda board: Action("sell", 0))
    result = economy.run_shop(stream, forced_sale, trace=True)
    assert result["reason"] == "unsupported_action" and result["violations"]
    assert (result["sells"], result["sale_income"], result["gross_spent"]) == (0, 0, 0)
    assert result["final_board"] == initial.to_dict()


@pytest.mark.parametrize("level", [None, 2, 3])
def test_simulator_rejects_non_level_one_ordinary_sale_without_mutating_board(level):
    initial = replace(full_board(), team=(pet("ant", 1, 1, level), *full_board().team[1:]))
    stream = SimpleNamespace(initial=initial, copies=(1, 6, 6, 6, 6), size=1)
    forced_sale = SimpleNamespace(choose_action=lambda board: Action("sell", 0))
    result = economy.run_shop(stream, forced_sale, trace=True)
    assert result["reason"] == "unsupported_action"
    assert result["violations"] == [{"unsupported_sale_level": level}]
    assert (result["sells"], result["sale_income"], result["gross_spent"]) == (0, 0, 0)
    assert result["final_board"] == initial.to_dict()


def test_merge_levelup_offer_uses_separate_stream():
    fish = pet("fish", 2, 3)
    initial = replace(full_board(shop=(fish,)), team=(fish, *(pet("pig", 8, 8, 3),) * 4))
    bonus = pet("beaver", 4, 5)
    requested = []
    stream = SimpleNamespace(initial=initial, copies=(2, 6, 6, 6, 6), size=1,
                             levelup_offer=lambda index: requested.append(index) or bonus)
    result = economy.run_shop(stream, DesktopPolicy(), trace=True)
    assert requested == [0]
    assert result["levelups"] == result["merges"] == 1
    assert result["final_board"]["team"][0]["level"] == 2
    assert result["final_board"]["shop"][0] == {
        "occupied": True, "species": "beaver", "attack": 4, "health": 5, "level": 1,
    }


def test_illegal_policy_is_reported_and_not_applied():
    policy = SimpleNamespace(choose_action=lambda board: Action("continue"))
    result = economy.run_shop(economy.OfferStream(1), policy)
    assert result["reason"] == "illegal_action" and result["violations"]
    assert result["actions"] == 0 and result["net_spent"] == 0


def test_pair_metrics_obey_economy_identity_and_bound():
    for seed in range(40):
        for result in economy.run_pair(seed, trace=True).values():
            assert not result["violations"] and result["reason"] == "end_turn"
            assert result["initial_gold"] - result["leftover_gold"] == result["net_spent"]
            assert result["gross_spent"] == 3 * (result["buys"] + result["merges"]) + result["rolls"]
            pig_bonuses = sum(event["action"]["kind"] == "sell"
                              and event["board"]["team"][event["action"]["slot"]]["species"] == "pig"
                              for event in result["trace"])
            assert result["sale_income"] == result["sells"] + pig_bonuses
            assert 0 <= result["unproductive_rolls"] <= result["rolls"]
            assert result["actions"] <= 50


def test_resumed_pairs_equal_continuous_evaluation(tmp_path):
    path = tmp_path / "resumed.json"
    economy.evaluate(output=path, pairs=7)
    resumed = economy.evaluate(output=path, pairs=13, resume=True)
    continuous = economy.evaluate(output=tmp_path / "continuous.json", pairs=20)
    for key in continuous.keys() - {"elapsed_seconds"}:
        assert resumed[key] == continuous[key], key
    assert resumed["pairs"] == resumed["next_seed"] == 20
    assert resumed["variants"]["baseline"]["summary"]["paired_mean_ci95"] == [0, 0]


def test_interrupted_pair_does_not_partially_advance_any_variant(tmp_path, monkeypatch):
    original = economy.run_shop
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == len(economy.VARIANTS) + 3:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(economy, "run_shop", interrupted)
    output = tmp_path / "interrupted.json"
    partial = economy.evaluate(output=output, pairs=3)
    assert partial["pairs"] == partial["next_seed"] == 1
    assert partial["stop_reason"] == "interrupted"
    monkeypatch.setattr(economy, "run_shop", original)
    resumed = economy.evaluate(output=output, pairs=2, resume=True)
    continuous = economy.evaluate(output=tmp_path / "continuous.json", pairs=3)
    assert resumed["variants"] == continuous["variants"]


def test_resume_rejects_changed_configuration_or_source(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    economy.evaluate(output=output, pairs=1)
    with pytest.raises(ValueError, match="output exists"):
        economy.evaluate(output=output, pairs=1)
    with pytest.raises(ValueError, match="same schema"):
        economy.evaluate(output=output, pairs=1, seed=1, resume=True)
    monkeypatch.setitem(economy.VARIANTS, "extra", {"minimum_upgrade_gain": 1})
    with pytest.raises(ValueError, match="configuration"):
        economy.evaluate(output=output, pairs=1, resume=True)


def test_stop_file_checkpoints_without_running_pairs(tmp_path):
    stop = tmp_path / "stop"
    stop.touch()
    report = economy.evaluate(output=tmp_path / "report.json", pairs=3, stop_file=stop)
    assert report["pairs"] == 0 and report["stop_reason"] == "stop_file"
    assert report["variants"]["baseline"]["summary"]["stat_gain_per_net_gold"] is None


@pytest.mark.parametrize("options", [{"pairs": 0}, {"seed": -1}, {"seconds": float("inf")},
                                    {"checkpoint_seconds": None}, {"seconds": 0}])
def test_invalid_limits_are_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        economy.evaluate(output=tmp_path / "report.json", **options)


def test_cli_replay_prints_all_variants_without_saving(tmp_path, capsys):
    output = tmp_path / "unused.json"
    assert economy.main(["--replay", "0", "--output", str(output)]) == 0
    assert set(json.loads(capsys.readouterr().out)) == set(economy.VARIANTS)
    assert not output.exists()
