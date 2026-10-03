"""Offline soak/replay contracts; no screenshots, models, or desktop inputs."""
import json
from dataclasses import replace

import pytest

import desktop_session
from desktop_session import action_acknowledged
from desktop_state import Action, DesktopPolicy, PetSlot, legal_action
from tools import eval_desktop as evaluation


@pytest.mark.parametrize("scenario", evaluation.SCENARIOS)
def test_each_scenario_exercises_its_expected_bounded_outcome(scenario):
    case = evaluation.run_case(31, scenario=scenario, turns=3)
    assert case["passed"], case
    assert case["reason"] == evaluation.EXPECTED[scenario]
    assert case["polls"] <= 700
    if scenario in evaluation.SCENARIOS[:5]:
        assert case["shop_rounds"] == 3
        assert case["actions"] == case["acknowledgments"]


def test_seed_replay_has_identical_actions_observations_and_trace_hash():
    first = evaluation.run_case(123, scenario="transient", turns=3, trace=True)
    assert evaluation.run_case(123, scenario="transient", turns=3, trace=True) == first
    assert evaluation.run_case(124, scenario="transient", turns=3)["trace_sha256"] != first["trace_sha256"]


def test_unknown_identity_never_authorizes_merge_or_stat_only_sale():
    case = evaluation.run_case(7, scenario="unidentified", turns=6)
    assert case["passed"]
    assert not set(case["action_kinds"]) & {"merge", "sell"}
    assert case["action_kinds"]["buy"] > 0


@pytest.mark.parametrize("scenario", ["ignored_input", "wrong_price", "wrong_target", "input_error", "ocr_timeout_after_input", "capped_merge",
                                      "wrong_stats", "contradictory_merge"])
def test_failed_or_unverifiable_effect_is_never_retried(scenario):
    case = evaluation.run_case(1, scenario=scenario)
    assert case["passed"] and case["actions"] == 1 and case["acknowledgments"] == 0


@pytest.mark.parametrize("scenario", ["ocr_timeout", "unknown_phase", "unknown_slot", "missing_stat", "missing_gold"])
def test_unknown_observation_never_reaches_input(scenario):
    case = evaluation.run_case(1, scenario=scenario)
    assert case["passed"] and case["actions"] == 0


def ready_fixture(scenario="nominal"):
    fixture = evaluation.SyntheticDesktop(2, scenario, 2)
    fixture.board = replace(fixture.board, team=(evaluation.EMPTY,) * 5)
    fixture.copies = [0] * 5
    fixture.observe()
    fixture.observe()
    return fixture


@pytest.mark.parametrize("scenario", ["nominal", "gaps"])
def test_purchase_transition_uses_price_explicit_destination_and_ordered_survivors(scenario):
    fixture = ready_fixture(scenario)
    before = fixture.board
    action = Action("buy", 1, 4)
    fixture.act(action)
    after = fixture.board
    assert after.gold == before.gold - 3
    assert after.team[4] == before.shop[1]
    assert after.team[:4] == (evaluation.EMPTY,) * 4
    remaining = list(before.shop)
    remaining[1] = evaluation.EMPTY
    if scenario != "gaps":
        remaining = [pet for pet in remaining if pet.occupied] + [evaluation.EMPTY]
    assert after.shop == tuple(remaining)
    assert action_acknowledged(before, after, action)


def test_merge_tracks_partial_experience_and_level_threshold():
    fixture = ready_fixture()
    fish = PetSlot(True, "fish", 5, 6, 1)
    fixture.board = replace(fixture.board, team=(fish,) + (evaluation.EMPTY,) * 4,
                            shop=(replace(fish, attack=2, health=3), *fixture.board.shop[1:]))
    fixture.copies[0] = 2
    fixture.observe()
    fixture.observe()
    before = fixture.board
    fixture.act(Action("merge", 0, 0))
    assert fixture.copies[0] == 3
    assert fixture.board.team[0] == PetSlot(True, "fish", 6, 7, 2)
    assert action_acknowledged(before, fixture.board, Action("merge", 0, 0))


@pytest.mark.parametrize("scenario,kind", [("wrong_stats", "buy"), ("contradictory_merge", "merge")])
@pytest.mark.parametrize("seed,field", [(0, "attack"), (1, "health")])
def test_stat_faults_preserve_purchase_evidence_but_contradict_the_target(scenario, kind, seed, field):
    fixture = evaluation.SyntheticDesktop(seed, scenario, 2)
    fixture.observe()
    fixture.observe()
    before = fixture.board
    action = DesktopPolicy().choose_action(before)
    assert action.kind == kind and legal_action(before, action)
    if kind == "merge":
        assert all(pet.occupied is True and pet.strength is not None for pet in before.team)
    baseline = before.shop[action.slot] if kind == "buy" else before.team[action.target]

    fixture.act(action)
    after, target = fixture.board, fixture.board.team[action.target]
    assert after.gold == before.gold - 3
    assert target.occupied is True and target.species == baseline.species and target.level == baseline.level
    remaining = [pet for index, pet in enumerate(before.shop) if index != action.slot and pet.occupied]
    assert after.shop == tuple(remaining + [evaluation.EMPTY] * (len(before.shop) - len(remaining)))
    assert getattr(target, field) == getattr(baseline, field) - 1
    other = "health" if field == "attack" else "attack"
    assert getattr(target, other) == getattr(baseline, other) + (kind == "merge")
    assert not action_acknowledged(before, after, action)

    case = evaluation.run_case(seed, scenario=scenario, turns=2, trace=True)
    assert case["passed"], case
    assert (case["reason"], case["actions"], case["acknowledgments"]) == ("action_timeout", 1, 0)
    assert case["action_kinds"] == {kind: 1} and case["polls"] <= 22
    assert case["result"]["pending_action"] == action.to_dict()
    assert not any(event["event"] == "acknowledged" for event in case["trace"])


@pytest.mark.parametrize("scenario,kind", [
    ("wrong_stats", "buy"), ("contradictory_merge", "merge"), ("capped_merge", "merge"),
])
def test_independent_oracle_detects_unverifiable_effect_if_acknowledgment_guard_is_bypassed(monkeypatch, scenario, kind):
    original = desktop_session.action_acknowledged

    def unchecked_stats(before, after, action, **kwargs):
        if action.kind == kind:
            return after.gold == before.gold - 3 and after.team[action.target].occupied is True
        return original(before, after, action, **kwargs)

    monkeypatch.setattr(desktop_session, "action_acknowledged", unchecked_stats)
    case = evaluation.run_case(1, scenario=scenario, turns=2, trace=True)
    assert not case["passed"]
    assert case["reason"] == "event_error" and case["actions"] == 1
    assert "acknowledged stale or incorrect action effect" in case["violations"]


@pytest.mark.parametrize("pet, income", [
    (PetSlot(True, "pig", 3, 1, 1), 2),
    (PetSlot(True, "ant", 2, 1, 1), 1),
    (PetSlot(True, "fish", 5, 6, 2), 1),
    (PetSlot(True, "beaver", 8, 8, 3), 1),
    (PetSlot(True, None, 3, 2, None), 1),
], ids=["level-one-pig", "ordinary-level-one", "ordinary-level-two",
        "ordinary-level-three", "unidentified"])
def test_sale_transition_credits_observed_income_and_acknowledges(pet, income):
    fixture = ready_fixture()
    fixture.board = replace(fixture.board, gold=1,
                            team=(pet,) + (evaluation.EMPTY,) * 4)
    fixture.copies[0] = 1
    fixture.observe()
    fixture.observe()
    before = fixture.board
    action = Action("sell", 0)

    fixture.act(action)
    # The fixture may emit stale frames before the completed sale becomes visible.
    observed = fixture.observe()
    while fixture.frames:
        observed = fixture.observe()
    if observed == before:
        observed = fixture.observe()

    assert observed.gold == before.gold + income
    assert observed.team == (evaluation.EMPTY,) * 5
    assert observed.shop == before.shop
    assert fixture.copies == [0] * 5
    assert action_acknowledged(before, observed, action)


@pytest.mark.parametrize("level", [None, 2, 3], ids=["missing-level", "level-two", "level-three"])
def test_unsupported_pig_sale_rejects_before_board_or_copy_mutation(level):
    fixture = ready_fixture()
    pig = PetSlot(True, "pig", 5, 3, level)
    fixture.board = replace(fixture.board, gold=1,
                            team=(pig,) + (evaluation.EMPTY,) * 4)
    fixture.copies[0] = 3
    fixture.observe()
    fixture.observe()
    before, copies = fixture.board, list(fixture.copies)

    with pytest.raises(AssertionError, match="^unsupported observed sale income$"):
        fixture.act(Action("sell", 0))

    assert fixture.board == before
    assert fixture.copies == copies
    assert not fixture.frames
    assert fixture.violations == ["unsupported observed sale income"]


def test_controller_oracle_accepts_pig_sale_that_funds_purchase(monkeypatch):
    class PigSaleDesktop(evaluation.SyntheticDesktop):
        def __init__(self, seed, scenario, turns):
            super().__init__(seed, scenario, turns)
            pig = PetSlot(True, "pig", 3, 1, 1)
            survivor = PetSlot(True, "ant", 10, 10, 3)
            offer = PetSlot(True, "fish", 8, 8, 1)
            self.board = replace(self.board, gold=1, shop=(offer,) + (evaluation.EMPTY,) * (self.shop_size - 1),
                                 team=(pig,) + (survivor,) * 4)
            self.copies = [1, 6, 6, 6, 6]
            self.initial = self.project(self.board)

    monkeypatch.setattr(evaluation, "SyntheticDesktop", PigSaleDesktop)
    case = evaluation.run_case(2, scenario="nominal", turns=1, trace=True)

    assert case["passed"], case
    assert case["reason"] == "result"
    assert case["actions"] == case["acknowledgments"] == 3
    assert case["action_kinds"] == {"sell": 1, "buy": 1, "end_turn": 1}
    sale = next(event for event in case["trace"]
                if event["event"] == "acknowledged" and event["action"]["kind"] == "sell")
    assert sale["board"]["gold"] == 3
    assert sale["board"]["team"][0]["occupied"] is False


def test_transition_oracle_rejects_illegal_purchase_independently():
    fixture = ready_fixture()
    fixture.board = replace(fixture.board, gold=2)
    fixture.observe()
    fixture.observe()
    with pytest.raises(AssertionError, match="budget"):
        fixture.act(Action("buy", 0, 0))
    assert fixture.violations == ["purchase without offer/budget"]
    assert fixture.board.gold == 2


def test_evaluator_detects_an_illegal_policy_instead_of_reporting_success():
    class BrokenPolicy:
        def choose_action(self, board):
            return Action("continue")

    case = evaluation.run_case(0, policy=BrokenPolicy())
    assert not case["passed"]
    assert case["reason"] == "illegal_action"
    assert case["actions"] == 0
    assert case["trace"] and case["initial_board"]


def test_independent_oracle_rejects_roll_acknowledgment_of_unchanged_gold(monkeypatch):
    original = desktop_session.action_acknowledged
    monkeypatch.setattr(desktop_session, "action_acknowledged",
                        lambda before, after, action, **kwargs: action.kind == "roll" or original(before, after, action, **kwargs))
    case = evaluation.run_case(4, scenario="nominal", turns=2, trace=True)
    assert not case["passed"]
    assert "acknowledged stale or incorrect action effect" in case["violations"]
    assert case["reason"] == "event_error"


def test_resume_matches_uninterrupted_seed_range_and_coverage(tmp_path):
    resumed_path, continuous_path = tmp_path / "resumed.json", tmp_path / "continuous.json"
    evaluation.evaluate(output=resumed_path, cases=13, turns=2)
    resumed = evaluation.evaluate(output=resumed_path, cases=35, turns=2, resume=True)
    continuous = evaluation.evaluate(output=continuous_path, cases=48, turns=2)
    for key in continuous.keys() - {"elapsed_seconds"}:
        assert resumed[key] == continuous[key], key
    assert resumed["cases"] == resumed["passed"] == resumed["next_seed"] == 48
    assert set(resumed["scenarios"]) == set(evaluation.SCENARIOS)
    assert set(resumed["action_kinds"]) == {"buy", "merge", "sell", "roll", "end_turn"}
    assert json.loads(resumed_path.read_text())["failed"] == 0


def test_existing_report_needs_explicit_resume_and_compatible_sources(tmp_path, monkeypatch):
    path = tmp_path / "soak.json"
    evaluation.evaluate(output=path, cases=1)
    with pytest.raises(ValueError, match="already exists"):
        evaluation.evaluate(output=path, cases=1)
    with pytest.raises(ValueError, match="matching"):
        evaluation.evaluate(output=path, cases=1, resume=True, turns=7)
    monkeypatch.setattr(evaluation, "source_signature", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="source"):
        evaluation.evaluate(output=path, cases=1, resume=True)


def test_stop_file_prevents_work_and_saves_resumable_report(tmp_path):
    stop, output = tmp_path / "stop", tmp_path / "report.json"
    stop.touch()
    report = evaluation.evaluate(output=output, seconds=10, stop_file=stop)
    assert report["cases"] == 0 and report["stop_reason"] == "stop_file"
    assert output.is_file()


def test_time_limit_is_bounded_without_real_sleep(tmp_path, monkeypatch):
    tick = iter(range(20))
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: next(tick))
    report = evaluation.evaluate(output=tmp_path / "report.json", seconds=0.5)
    assert report["cases"] == 0 and report["stop_reason"] == "time_limit"


def test_keyboard_interrupt_preserves_completed_seed_boundary(tmp_path, monkeypatch):
    original = evaluation.run_case

    def interrupted(seed, **kwargs):
        if seed == 1:
            raise KeyboardInterrupt()
        return original(seed, **kwargs)

    monkeypatch.setattr(evaluation, "run_case", interrupted)
    report = evaluation.evaluate(output=tmp_path / "report.json", cases=3)
    assert report["cases"] == report["next_seed"] == 1
    assert report["stop_reason"] == "interrupted"


@pytest.mark.parametrize("callback", ["observe", "act"])
def test_interrupt_inside_session_preserves_seed_and_resumes_same_case(tmp_path, monkeypatch, callback):
    output = tmp_path / "interrupted.json"
    original = getattr(evaluation.SyntheticDesktop, callback)

    def interrupt_second_case(fixture, *args):
        if fixture.scenario == "gaps":  # Seed one; seed zero must finish first.
            raise KeyboardInterrupt()
        return original(fixture, *args)

    monkeypatch.setattr(evaluation.SyntheticDesktop, callback, interrupt_second_case)
    partial = evaluation.evaluate(output=output, cases=3, turns=2)
    assert partial["stop_reason"] == "interrupted"
    assert partial["cases"] == partial["next_seed"] == partial["passed"] == 1
    assert partial["failed"] == 0 and not partial["failure_examples"]
    monkeypatch.setattr(evaluation.SyntheticDesktop, callback, original)
    resumed = evaluation.evaluate(output=output, cases=2, turns=2, resume=True)
    continuous = evaluation.evaluate(output=tmp_path / "continuous.json", cases=3, turns=2)
    for key in continuous.keys() - {"elapsed_seconds"}:
        assert resumed[key] == continuous[key], key


def test_failure_report_has_seed_trace_and_fail_fast(tmp_path, monkeypatch):
    original = evaluation.run_case

    class BrokenPolicy:
        def choose_action(self, board):
            return Action("continue")

    monkeypatch.setattr(evaluation, "run_case", lambda seed, **kwargs: original(seed, policy=BrokenPolicy(), **kwargs))
    report = evaluation.evaluate(output=tmp_path / "report.json", cases=10, fail_fast=True)
    assert report["cases"] == report["failed"] == 1
    assert report["stop_reason"] == "failure"
    assert report["failure_examples"][0]["seed"] == 0
    assert report["failure_examples"][0]["trace"]


@pytest.mark.parametrize("options", [{"seconds": float("inf")}, {"seconds": 0}, {"cases": 0},
                                    {"turns": 0}, {"seed": -1}, {"checkpoint_seconds": float("nan")}])
def test_invalid_soak_limits_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        evaluation.evaluate(output=tmp_path / "report.json", **options)


def test_cli_replay_is_json_and_does_not_write_report(tmp_path, capsys):
    output = tmp_path / "unused.json"
    assert evaluation.main(["--replay", "5", "--scenario", "wrong_target", "--output", str(output)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["seed"] == 5 and data["scenario"] == "wrong_target" and data["passed"]
    assert not output.exists()
