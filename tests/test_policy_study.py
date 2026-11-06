"""Reproducible, paired policy studies; no native or OCR dependencies."""
import json
from pathlib import Path

import pytest

from sap_ai.desktop.strategy import make_policy
from sap_ai.sim import policy_study


def test_named_presets_have_explicit_budgets_and_fresh_instances():
    baseline, candidate = make_policy("conservative"), make_policy("economy")
    assert (baseline.merge_first_team_size, baseline.sale_funded_rolls) == (3, False)
    assert (candidate.merge_first_team_size, candidate.sale_funded_rolls) == (5, True)
    assert make_policy("economy", sale_funded_rolls=False).sale_funded_rolls is False
    assert make_policy("economy") is not candidate
    with pytest.raises(ValueError):
        make_policy("typo")
    with pytest.raises(ValueError):
        make_policy("economy", sale_funded_rolls=1)


def test_identical_policies_have_zero_paired_difference_and_reproduce_exactly():
    kwargs = {"baseline": "economy", "candidate": "economy", "opponents": ("recorded",),
              "runs": 12, "seed": 123, "include_episodes": True}
    first = policy_study.study(**kwargs)
    assert first == policy_study.study(**kwargs)
    result = first["results"]["recorded"]
    assert result["mean_trophy_delta"] == 0 and result["paired_mean_ci95"] == [0, 0]
    assert result["better"] == result["worse"] == 0 and result["equal"] == 12
    assert [episode["seed"] for episode in result["episodes"]] == list(range(123, 135))
    assert all(len(value) == 64 for value in first["source_sha256"].values())


def test_paired_statistics_and_factory_isolation(monkeypatch):
    made, calls = [], []

    class Policy:
        def __init__(self, name):
            self.name, self.used = name, False
            made.append(self)

    def play(policy, opponent, seed):
        assert not policy.used and not opponent.used
        policy.used = opponent.used = True
        calls.append((policy.name, seed))
        trophies = seed - 20 if policy.name == "candidate" else 0
        return {"trophies": trophies, "turns": 3, "victory": False}

    monkeypatch.setattr(policy_study, "play_run", play)
    result = policy_study.evaluate_pair(lambda: Policy("baseline"), lambda: Policy("candidate"),
                                        lambda: Policy("opponent"), runs=3, seed=20)
    assert len(made) == 12 and len({id(policy) for policy in made}) == 12
    assert calls == [("baseline", 20), ("candidate", 20), ("baseline", 21),
                     ("candidate", 21), ("baseline", 22), ("candidate", 22)]
    assert result["mean_trophy_delta"] == 1 and result["better"] == 2 and result["equal"] == 1
    assert result["paired_mean_ci95"][0] == pytest.approx(1 - 1.96 / 3 ** .5)
    assert result["paired_mean_ci95"][1] == pytest.approx(1 + 1.96 / 3 ** .5)
    assert "episodes" not in result


def test_single_run_does_not_claim_a_confidence_interval():
    result = policy_study.study(runs=1, seed=8, opponents=("recorded",))
    assert result["results"]["recorded"]["paired_mean_ci95"] is None


@pytest.mark.parametrize("changes", [{"runs": 0}, {"runs": True}, {"seed": -1}, {"seed": 1.5},
                                    {"opponents": ()}, {"opponents": ("recorded", "recorded")}])
def test_invalid_studies_fail_before_running_games(changes, monkeypatch):
    monkeypatch.setattr(policy_study, "play_run", lambda *args: pytest.fail("no games on invalid inputs"))
    with pytest.raises(ValueError):
        policy_study.study(**changes)


def test_cli_writes_reproducible_json_without_overwriting_existing_reports(tmp_path, monkeypatch):
    output = tmp_path / "paired.json"
    report = policy_study.main(["--runs", "4", "--seed", "50", "--opponents", "recorded",
                               "--include-episodes", "--output", str(output)])
    assert json.loads(output.read_text()) == report
    old = output.read_bytes()
    monkeypatch.setattr(policy_study, "study", lambda **kwargs: pytest.fail("must reject existing output first"))
    with pytest.raises(SystemExit):
        policy_study.main(["--output", str(output)])
    assert output.read_bytes() == old



def test_economy_strategy_preserves_legality_and_abstention_on_recorded_boards():
    from sap_ai.desktop.state import Board, legal_action
    boards = {}
    for path in sorted((Path(__file__).parent / "fixtures").glob("desktop_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")).get("boards", []):
            board = Board.from_dict(row)
            boards.setdefault(board.fingerprint(), board)
    baseline, candidate = make_policy("conservative"), make_policy("economy")
    for board in boards.values():
        original, proposed = baseline.choose_action(board), candidate.choose_action(board)
        if original is None:
            assert proposed is None
        if proposed is not None:
            assert legal_action(board, proposed)
    assert len(boards) > 400
