"""Competing plans, native receipt contracts and a sealed stat-only evaluation."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_session import DesktopSession, action_acknowledged
from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase, legal_action
from tools.desktop_replacement_candidate import VisibleReplacementPolicy, visible_replacement_plan
from tools import eval_desktop_replacements as study


def pet(species=None, attack=2, health=2, level=1):
    return PetSlot(True, species, attack, health, level)


def competing_board():
    # Two gold buys one ordinary replacement or two supported Pig replacements.
    return Board(Phase.SHOP, gold=2, turn=6,
                 shop=(pet(None, 7, 7, None), pet(None, 6, 6, None)),
                 team=(pet(None, 1, 1), pet("pig", 2, 2), pet("pig", 2, 2),
                       pet(None, 20, 20, 2), pet(None, 20, 20, 2)))


def test_complete_budget_plan_beats_greedy_weakest_and_one_step_gain_ratio():
    board = competing_board()
    assert DesktopPolicy().choose_action(board) == Action("sell", 0)
    assert VisibleReplacementPolicy().choose_action(board) == Action("sell", 1)
    plan = visible_replacement_plan(board)
    assert (plan.stat_gain, plan.net_cost, plan.targets) == (18, 2, (1, 2))
    baseline = study.run_visible_shop(board, DesktopPolicy(), trace=True)
    candidate = study.run_visible_shop(board, VisibleReplacementPolicy(), trace=True)
    assert baseline["stat_gain"] == 12 and candidate["stat_gain"] == 18
    assert baseline["net_spent"] == candidate["net_spent"] == 2
    assert [row["action"]["kind"] for row in candidate["trace"]] == ["sell", "buy", "sell", "buy"]
    assert candidate["final_board"]["team"][0] == board.team[0].__dict__


def test_parent_choice_survives_when_an_immediate_pig_ratio_is_dominated():
    board = replace(competing_board(), gold=2, shop=(pet(None, 7, 7, None),))
    # One Pig gives ten stats per gold, but the ordinary sale gives twelve
    # total stats and there is no second visible offer to use the extra gold.
    assert VisibleReplacementPolicy().choose_action(board) == Action("sell", 0)
    assert study.run_visible_shop(board, VisibleReplacementPolicy())["stat_gain"] == 12


def test_equal_utility_keeps_baseline_slot_order():
    board = replace(competing_board(), gold=4,
                    team=tuple(replace(p, species=None) for p in competing_board().team))
    assert VisibleReplacementPolicy().choose_action(board) == DesktopPolicy().choose_action(board)


def test_actual_recorded_board_tie_does_not_claim_a_strategic_improvement():
    data = json.loads((Path(__file__).parent / "fixtures/desktop_turns_4_to_terminal.json").read_text())
    board = Board.from_dict(data["boards"][38])
    assert DesktopPolicy().choose_action(board) == Action("sell", 4)
    assert VisibleReplacementPolicy().choose_action(board) == Action("sell", 4)


@pytest.mark.parametrize("income_evidence", ["unknown_species", "level_unknown", "level_two", "level_three"])
def test_two_replacement_plan_cannot_invent_a_pig_receipt(income_evidence):
    board = competing_board()
    pigs = list(board.team)
    for index in (1, 2):
        pigs[index] = (replace(pigs[index], species=None) if income_evidence == "unknown_species"
                       else replace(pigs[index], level={"level_unknown": None, "level_two": 2,
                                                        "level_three": 3}[income_evidence]))
    board = replace(board, team=pigs)
    assert VisibleReplacementPolicy().choose_action(board) == Action("sell", 0)
    result = study.run_visible_shop(board, VisibleReplacementPolicy())
    assert result["stat_gain"] == 12 and result["sells"] == 1


@pytest.mark.parametrize("board", [Board(Phase.UNKNOWN),
    replace(competing_board(), gold=None),
    replace(competing_board(), team=(PetSlot(None), *competing_board().team[1:])),
    replace(competing_board(), shop=(PetSlot(True, attack=5),))])
def test_candidate_preserves_abstention_on_unknown_observations(board):
    assert VisibleReplacementPolicy().choose_action(board) is None


def test_candidate_keeps_parent_empty_fill_and_legal_identified_merge():
    board = replace(competing_board(), gold=3, team=(study.EMPTY, *competing_board().team[1:]))
    assert VisibleReplacementPolicy().choose_action(board) == Action("buy", 0, 0)
    board = replace(board, shop=(pet("fish", 3, 3),),
                    team=(pet("fish", 2, 2), *board.team[1:]))
    assert VisibleReplacementPolicy().choose_action(board) == Action("merge", 0, 0)
    # The candidate preserves a real legal proposal; this bounded evaluator
    # still refuses to invent its hidden experience or level-up outcome.
    with pytest.raises(study.UnsupportedTransition, match="unsupported action: merge"):
        study.apply_visible_action(board, Action("merge", 0, 0))


@pytest.mark.parametrize("source", [pet(None, 3, 3, 1), pet("fish", 3, 3, 2)])
def test_unidentified_or_non_level_one_shop_pet_cannot_manufacture_merge(source):
    board = replace(competing_board(), gold=3, shop=(source,),
                    team=(pet("fish", 10, 10), *competing_board().team[1:]))
    assert not legal_action(board, Action("merge", 0, 0))
    assert VisibleReplacementPolicy().choose_action(board).kind != "merge"


def test_modeled_pig_receipt_matches_independent_native_typed_fixture():
    data = json.loads((Path(__file__).parent / "fixtures/desktop_pig_sale.json").read_text())
    before, native_after = Board.from_dict(data["boards"][0]), Board.from_dict(data["boards"][-1])
    action = Action.from_dict(data["action"])
    assert study.apply_visible_action(before, action) == native_after
    assert action_acknowledged(before, native_after, action)
    assert not action_acknowledged(before, replace(native_after, gold=before.gold + 1), action)
    assert not action_acknowledged(before, replace(native_after, gold=before.gold + 3), action)


def test_native_opening_receipt_accepts_identity_dropout_and_rejects_wrong_survivor():
    data = json.loads((Path(__file__).parent / "fixtures/desktop_opening_purchases.json").read_text())
    before, after = (Board.from_dict(row) for row in data["boards"][:2])
    action = Action.from_dict(data["actions"][0]["action"])
    assert action_acknowledged(before, after, action)
    bad_shop = (replace(after.shop[0], health=4), *after.shop[1:])
    assert not action_acknowledged(before, replace(after, shop=bad_shop), action)
    assert not action_acknowledged(before, replace(after, gold=before.gold - 2), action)


@pytest.mark.parametrize("level", [None, 2, 3])
def test_legal_but_unsupported_sale_is_rejected_before_any_model_mutation(level):
    board = replace(competing_board(), team=(pet("pig", 1, 1, level), *competing_board().team[1:]))
    original = board.to_dict()
    assert legal_action(board, Action("sell", 0))
    with pytest.raises(study.UnsupportedTransition, match="unsupported sale"):
        study.apply_visible_action(board, Action("sell", 0))
    assert board.to_dict() == original


def test_sale_waits_for_receipt_before_fresh_buy_and_never_retries_wrong_bonus():
    before = competing_board()
    sale = Action("sell", 1)
    correct = study.apply_visible_action(before, sale)
    wrong = replace(correct, gold=before.gold + 1)
    frames, sent = iter([before, before, wrong, wrong, wrong, wrong]), []
    result = DesktopSession(lambda: next(frames), sent.append, VisibleReplacementPolicy(),
                            max_polls=6, action_max_polls=4, clock=lambda: 0.,
                            sleep=lambda _: None).run()
    assert sent == [sale]
    assert result.reason == "action_timeout" and result.pending_action == sale
    assert result.acknowledgments == 0


def test_simulator_stops_before_roll_without_assigning_unknown_offer_value():
    board = replace(competing_board(), gold=4, shop=(pet(None, 1, 1, None),))
    result = study.run_visible_shop(board, VisibleReplacementPolicy(), trace=True)
    assert result["reason"] == "visible_horizon"
    assert result["stat_gain"] == result["net_spent"] == result["receipts"] == 0
    assert result["final_board"] == board.to_dict()


def test_contract_failures_and_action_bound_are_visible(monkeypatch):
    board = competing_board()
    monkeypatch.setattr(study, "action_acknowledged", lambda *args: False)
    result = study.run_visible_shop(board, VisibleReplacementPolicy())
    assert result["reason"] == "unsupported_action" and result["receipts"] == 0
    assert "receipt rejected" in result["error"]
    illegal = SimpleNamespace(choose_action=lambda board: Action("buy", 0, 0))
    assert study.run_visible_shop(board, illegal)["reason"] == "invalid_action"


def test_keyed_population_is_reproducible_and_cannot_create_hidden_merges():
    for seed in range(30):
        board = study.sample_board(seed)
        assert board == study.sample_board(seed)
        assert all(p.species is None and p.level is None for p in board.shop)
        assert study.run_visible_shop(board, VisibleReplacementPolicy()) == study.run_visible_shop(board, VisibleReplacementPolicy())


def test_fixed_candidate_evaluation_uses_disjoint_splits_and_saves_source_provenance(tmp_path):
    output = tmp_path / "study.json"
    report = study.evaluate(output=output, train_pairs=50, heldout_pairs=60, train_seed=100, heldout_seed=1000)
    assert report == json.loads(output.read_text())
    assert report["training"]["pairs"] == 50 and report["heldout"]["pairs"] == 60
    assert report["candidate_frozen_before_evaluation"] is True
    assert "no parameters or model weights trained" in report["limitations"].casefold()
    assert set(report["source_sha256"]) == {"eval_desktop_replacements.py", "desktop_replacement_candidate.py",
                                              "desktop_state.py", "desktop_session.py"}
    assert not report["training"]["failure_examples"] and not report["heldout"]["failure_examples"]
    assert report["recorded_boards"]["unique_shop_boards"] > 100
    assert not report["recorded_boards"]["illegal_proposals"]


@pytest.mark.parametrize("options", [{"train_pairs": 0}, {"heldout_pairs": -1}, {"train_seed": -1},
                                    {"heldout_seed": 100010}, {"train_pairs": True}])
def test_evaluation_rejects_invalid_or_overlapping_seeds(tmp_path, options):
    with pytest.raises(ValueError):
        study.evaluate(output=tmp_path / "study.json", **options)
