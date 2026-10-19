"""Bounded development/held-out evaluation of a visible-shop policy candidate.

python tools/eval_desktop_replacements.py --train-pairs 10000 --heldout-pairs 10000

This is a fixed stat-only candidate, not reinforcement learning. Shops mask
species and level as commonly seen in native observations, so the experiment
cannot manufacture merges or level-up offers. It stops before any refresh or
combat, and checks every modeled sale/buy through production receipt logic.
Nothing imports this candidate into the live launcher or changes a checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

from sap_ai.desktop.session import action_acknowledged
from sap_ai.desktop.state import Action, Board, DesktopPolicy, PetSlot, Phase, legal_action, observed_sale_income
from tools.desktop_replacement_candidate import VisibleReplacementPolicy
from tools.eval_desktop import save_report


EMPTY = PetSlot(False)
LIMITATIONS = (
    "Synthetic attack+health utility within one visible shop; no win-rate claim. "
    "Species/levels are masked on shop pets; this population cannot test merges. "
    "Excludes food, equipment, experience, positioning, unmodeled buy/sell abilities, "
    "combat and future-roll opportunity cost. Synthetic production receipts are "
    "contract checks, not independent native-game receipts. Recorded-board audit "
    "checks proposals only and does not measure outcomes of changed actions. "
    "Candidate was fixed before both splits; no parameters or model weights trained."
)


class UnsupportedTransition(ValueError):
    """The bounded experiment cannot justify this proposed transition."""


def apply_visible_action(before: Board, action: Action) -> Board:
    """Apply only a legal supported sale/buy and validate its normal receipt.

    This intentionally has no fallback for merges, rolls, ability effects or
    leveled sales. An unsupported action leaves the immutable input untouched.
    """
    if not legal_action(before, action):
        raise UnsupportedTransition("illegal action")
    team, shop = list(before.team), list(before.shop)
    if action.kind == "sell":
        pet = team[action.slot]
        income = observed_sale_income(pet)
        if pet.level != 1 or income is None:
            raise UnsupportedTransition("unsupported sale level or income")
        team[action.slot] = EMPTY
        after = replace(before, gold=before.gold + income, team=team)
    elif action.kind == "buy":
        team[action.target] = shop.pop(action.slot)
        shop = [pet for pet in shop if pet.occupied]
        shop.extend([EMPTY] * (len(before.shop) - len(shop)))
        after = replace(before, gold=before.gold - 3, shop=shop, team=team)
    else:
        raise UnsupportedTransition(f"unsupported action: {action.kind}")
    if not action_acknowledged(before, after, action):
        raise UnsupportedTransition("production receipt rejected modeled transition")
    return after


def team_strength(board: Board) -> int:
    return sum(pet.strength for pet in board.team if pet.occupied)


def run_visible_shop(initial: Board, policy, *, max_actions: int = 20, trace: bool = False) -> dict:
    """Stop at first refresh/end; never assign outcomes to unseen offers."""
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    board, rows = initial, []
    buys = sells = receipts = 0
    error = None
    reason = "action_limit"
    for _ in range(max_actions):
        action = policy.choose_action(board)
        if action is None:
            reason = "abstained"
            break
        if not legal_action(board, action):
            reason, error = "invalid_action", "policy proposed an illegal action"
            break
        if action.kind in {"roll", "end_turn"}:
            reason = "visible_horizon" if action.kind == "roll" else "end_turn"
            break
        try:
            after = apply_visible_action(board, action)
        except UnsupportedTransition as exc:
            reason, error = "unsupported_action", str(exc)
            break
        receipts += 1
        buys += action.kind == "buy"
        sells += action.kind == "sell"
        if trace:
            rows.append({"before": board.to_dict(), "action": action.to_dict(), "after": after.to_dict()})
        board = after
    result = {"stat_gain": team_strength(board) - team_strength(initial),
              "net_spent": initial.gold - board.gold, "leftover_gold": board.gold,
              "buys": buys, "sells": sells, "receipts": receipts,
              "reason": reason, "error": error}
    if trace:
        result.update(trace=rows, final_board=board.to_dict())
    return result


def sample_board(seed: int) -> Board:
    """Seeded synthetic stats with incomplete identity, not official pack draws."""
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f"desktop-visible-replacements-v1|{seed}".encode()).digest()[:16], "big"))
    turn = rng.randint(1, 11)
    size = 3 if turn < 5 else 4 if turn < 9 else 5
    upper = 3 + min(6, (turn + 1) // 2)
    team = []
    empty_probability = 0.2 if rng.random() < 0.35 else 0
    for _ in range(5):
        if rng.random() < empty_probability:
            team.append(EMPTY)
            continue
        species = rng.choices((None, "pig", "ant"), weights=(7, 2, 1))[0]
        level = rng.choices((1, 2, None), weights=(7, 2, 1))[0]
        buff = rng.randint(0, min(6, turn))
        team.append(PetSlot(True, species, rng.randint(1, upper) + buff,
                            rng.randint(1, upper) + buff, level))
    shop = tuple(PetSlot(True, None, rng.randint(1, upper), rng.randint(1, upper), None)
                 for _ in range(size))
    return Board(Phase.SHOP, gold=rng.randint(0, 10), turn=turn, shop=shop, team=team)


def evaluate_split(start: int, count: int) -> dict:
    total = square = better = worse = equal = receipt_count = 0
    base_gain = candidate_gain = base_spent = candidate_spent = 0
    failures, changed_examples = [], []
    for seed in range(start, start + count):
        board = sample_board(seed)
        baseline = run_visible_shop(board, DesktopPolicy())
        candidate = run_visible_shop(board, VisibleReplacementPolicy())
        difference = candidate["stat_gain"] - baseline["stat_gain"]
        total += difference
        square += difference * difference
        better += difference > 0
        worse += difference < 0
        equal += difference == 0
        base_gain += baseline["stat_gain"]
        candidate_gain += candidate["stat_gain"]
        base_spent += baseline["net_spent"]
        candidate_spent += candidate["net_spent"]
        receipt_count += baseline["receipts"] + candidate["receipts"]
        if len(changed_examples) < 5 and difference:
            changed_examples.append({"seed": seed, "difference": difference,
                                     "baseline": baseline, "candidate": candidate})
        for name, result in (("baseline", baseline), ("candidate", candidate)):
            if result["reason"] not in {"visible_horizon", "end_turn"}:
                if len(failures) < 10:
                    failures.append({"seed": seed, "variant": name, "result": result})
    mean = total / count
    variance = max(0., (square - count * mean * mean) / (count - 1)) if count > 1 else 0.
    width = 1.96 * math.sqrt(variance / count)
    return {"seed_start": start, "seed_end_exclusive": start + count, "pairs": count,
            "mean_paired_stat_gain": mean, "paired_mean_ci95": [mean - width, mean + width],
            "better": better, "equal": equal, "worse": worse,
            "baseline_mean_stat_gain": base_gain / count, "candidate_mean_stat_gain": candidate_gain / count,
            "baseline_mean_net_spent": base_spent / count, "candidate_mean_net_spent": candidate_spent / count,
            "receipt_contract_checks": receipt_count, "failure_examples": failures,
            "changed_examples": changed_examples}


def audit_recorded_boards(fixtures: Path) -> dict:
    seen, changed, invalid, abstentions = set(), [], [], 0
    shop_boards = proposed = 0
    for path in sorted(fixtures.glob("desktop_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(data.get("boards", [])):
            board = Board.from_dict(row)
            if board.phase != Phase.SHOP or board.fingerprint() in seen:
                continue
            seen.add(board.fingerprint())
            shop_boards += 1
            baseline = DesktopPolicy().choose_action(board)
            candidate = VisibleReplacementPolicy().choose_action(board)
            if candidate is None:
                abstentions += 1
            else:
                proposed += 1
                if not legal_action(board, candidate):
                    invalid.append({"fixture": path.name, "board_index": index})
            if candidate != baseline:
                changed.append({"fixture": path.name, "board_index": index,
                                "baseline": baseline.to_dict() if baseline else None,
                                "candidate": candidate.to_dict() if candidate else None})
    return {"unique_shop_boards": shop_boards, "proposals": proposed,
            "abstentions": abstentions, "illegal_proposals": invalid, "changed_proposals": changed,
            "limitation": "Historical proposal audit only; no counterfactual native outcome is inferred."}


def evaluate(*, output: Path | str, train_pairs: int = 10000, heldout_pairs: int = 10000,
             train_seed: int = 100000, heldout_seed: int = 1000000) -> dict:
    for name, value in (("train_pairs", train_pairs), ("heldout_pairs", heldout_pairs),
                        ("train_seed", train_seed), ("heldout_seed", heldout_seed)):
        if type(value) is not int or value < (1 if name.endswith("pairs") else 0):
            raise ValueError(f"{name} has an invalid bound")
    if max(train_seed, heldout_seed) < min(train_seed + train_pairs, heldout_seed + heldout_pairs):
        raise ValueError("training and held-out seeds must not overlap")
    output = Path(output)
    if output.exists():
        raise ValueError("output exists; choose a new path")
    started = time.monotonic()
    report = {"schema_version": 1, "kind": "visible_replacement_candidate_evaluation",
              "candidate_frozen_before_evaluation": True, "minimum_upgrade_gain": 4,
              "limitations": LIMITATIONS,
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__), ROOT / "tools/desktop_replacement_candidate.py",
                               ROOT / "sap_ai/desktop/state.py", ROOT / "sap_ai/desktop/session.py")}}
    report["training"] = evaluate_split(train_seed, train_pairs)
    report["heldout"] = evaluate_split(heldout_seed, heldout_pairs)
    report["recorded_boards"] = audit_recorded_boards(ROOT / "tests/fixtures")
    report["elapsed_seconds"] = time.monotonic() - started
    save_report(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=".local/desktop/replacement-candidate-evaluation.json")
    parser.add_argument("--train-pairs", type=int, default=10000)
    parser.add_argument("--heldout-pairs", type=int, default=10000)
    parser.add_argument("--train-seed", type=int, default=100000)
    parser.add_argument("--heldout-seed", type=int, default=1000000)
    args = parser.parse_args(argv)
    try:
        report = evaluate(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report[key] for key in ("training", "heldout", "recorded_boards", "elapsed_seconds")}, indent=2))
    return int(any(report[split]["failure_examples"] for split in ("training", "heldout"))
               or bool(report["recorded_boards"]["illegal_proposals"]))


if __name__ == "__main__":
    raise SystemExit(main())
