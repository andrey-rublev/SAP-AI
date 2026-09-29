"""Paired CPU study of Board-native shopping heuristics, without desktop IO.

    python tools/eval_desktop_economy.py --pairs 50000 --seconds 1800 --output .local/desktop/economy.json
    python tools/eval_desktop_economy.py --replay 42

Each seed supplies the same initial Board and independently keyed future shops
to every policy. This is a synthetic shop-budget study, not a combat model:
abilities, food, equipment, frozen offers, and actual opponent strength are
excluded. Stat gain is an explicit proxy, never a win rate. No production policy
or model checkpoint is changed. Candidates use only observed Board fields.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase, legal_action
from tools.eval_desktop import save_report


EMPTY = PetSlot(False)
SPECIES = ("ant", "fish", "duck", "otter", "beaver", "cricket", "pig", "mosquito")
VARIANTS = {
    "baseline": {"minimum_upgrade_gain": 4},
    "threshold_2": {"minimum_upgrade_gain": 2},
    "threshold_6": {"minimum_upgrade_gain": 6},
    "sale_funded_4": {"minimum_upgrade_gain": 4, "sale_funded": True},
    "gain_ranked_4": {"minimum_upgrade_gain": 4, "rank_gains": True},
    **{f"combined_{gain}": {"minimum_upgrade_gain": gain, "sale_funded": True, "rank_gains": True}
       for gain in (2, 4, 6)},
}


class EconomyPolicy(DesktopPolicy):
    """Experimental subclass; every proposal remains a normal desktop Action.

    Sale funding uses the one-gold floor for a known level-one pet. Gain ranking
    compares immediate stat gains per net gold: three for a merge, two for a
    level-one replacement. Unknown observations still obey the parent policy.
    """

    def __init__(self, *, minimum_upgrade_gain=4, sale_funded=False, rank_gains=False):
        super().__init__(minimum_upgrade_gain=minimum_upgrade_gain)
        self.sale_funded, self.rank_gains = sale_funded, rank_gains

    def choose_action(self, board):
        original = super().choose_action(board)
        if original is None or original.kind == "buy" or not all(pet.occupied is True for pet in board.team):
            return original
        replaceable = [(i, pet) for i, pet in enumerate(board.team) if pet.level == 1]
        offers = [(i, pet) for i, pet in enumerate(board.shop) if pet.occupied]
        replacement = None
        if replaceable and offers and board.gold >= (2 if self.sale_funded else 3):
            weak_index, weak = min(replaceable, key=lambda item: (item[1].strength, item[0]))
            _, offered = max(offers, key=lambda item: (item[1].strength, -item[0]))
            gain = offered.strength - weak.strength
            if gain >= self.minimum_upgrade_gain:
                replacement = gain, Action("sell", weak_index)
        if self.rank_gains:
            merges = []
            for i, source in offers:
                for j, target in enumerate(board.team):
                    action = Action("merge", i, j)
                    if legal_action(board, action):
                        gain = (min(50, max(source.attack, target.attack) + 1)
                                + min(50, max(source.health, target.health) + 1) - target.strength)
                        merges.append((gain, action))
            merge = max(merges, key=lambda item: (item[0], -item[1].slot, -item[1].target), default=None)
            if replacement and (merge is None or replacement[0] * 3 > merge[0] * 2):
                return replacement[1]
            if merge:
                return merge[1]
        if self.sale_funded and original.kind in ("roll", "end_turn"):
            if replacement:
                return replacement[1]
            if board.gold == 3 and replaceable:
                # A roll leaves two; a subsequent observed level-one sale can
                # fund a purchase, but only if the new offer clears the margin.
                return Action("roll")
        return original


def make_policy(name):
    options = VARIANTS[name]
    return EconomyPolicy(**options) if "sale_funded" in options or "rank_gains" in options else DesktopPolicy(**options)


class OfferStream:
    """Exogenous draws are indexed by semantic event, never global RNG order."""

    def __init__(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = seed
        initial = self.rng("initial")
        self.turn = initial.randint(1, 11)
        self.tier = min(6, (self.turn + 1) // 2)
        self.size = 3 if self.turn < 5 else 4 if self.turn < 9 else 5
        team = []
        copies = []
        for _ in range(5):
            if initial.random() < 0.25:
                team.append(EMPTY)
                copies.append(0)
                continue
            level = initial.choices((1, 2, 3), weights=(8, 2, 1))[0]
            pet = self.pet(initial, level=level, buff=initial.randint(0, self.tier * 2))
            copies.append(initial.randint(1, 2) if level == 1 else initial.randint(3, 5) if level == 2 else 6)
            if initial.random() < 0.1:
                pet = replace(pet, species=None, level=None)
            team.append(pet)
        self.copies = tuple(copies)
        self.initial = Board(Phase.SHOP, gold=initial.randint(0, 10), turn=self.turn,
                             shop=self.offers(0), team=team)

    def rng(self, name, index=0):
        key = f"desktop-economy-v1|{self.seed}|{name}|{index}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16], "big"))

    def pet(self, rng, *, level=1, buff=0):
        # Bounded synthetic stats span early and buffed shops. These are not
        # claims about current species base stats or official shop probabilities.
        upper = 3 + self.tier
        return PetSlot(True, rng.choice(SPECIES), min(50, rng.randint(1, upper) + buff),
                       min(50, rng.randint(1, upper) + buff), level)

    def offers(self, roll_index):
        rng = self.rng("shop", roll_index)
        return tuple(self.pet(rng) for _ in range(self.size))

    def levelup_offer(self, index):
        return self.pet(self.rng("levelup", index), buff=1)


def strength(board):
    return sum(pet.strength for pet in board.team if pet.occupied)


def run_shop(stream, policy, *, max_actions=50, trace=False):
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    board = stream.initial
    copies = list(stream.copies)
    gross_spent = sale_income = rolls = buys = merges = sells = levelups = unproductive_rolls = actions = 0
    pending_roll = False
    history, violations = [], []
    reason = "action_limit"
    for step in range(max_actions):
        action = policy.choose_action(board)
        if action is None:
            reason = "policy_stopped"
            break
        if not legal_action(board, action):
            violations.append({"action": action.to_dict(), "board": board.to_dict()})
            reason = "illegal_action"
            break
        actions += 1
        if trace:
            history.append({"board": board.to_dict(), "action": action.to_dict()})
        if action.kind == "end_turn":
            reason = "end_turn"
            break
        team, offers, gold = list(board.team), list(board.shop), board.gold
        if action.kind == "roll":
            unproductive_rolls += int(pending_roll)
            pending_roll = True
            rolls += 1
            offers, gold = stream.offers(rolls), gold - 1
            gross_spent += 1
        elif action.kind == "sell":
            # Every tested policy restricts sales to observed level-one pets.
            if team[action.slot].level != 1:
                violations.append({"unsupported_sale_level": team[action.slot].level})
                reason = "unsupported_action"
                break
            team[action.slot], copies[action.slot] = EMPTY, 0
            gold += 1
            sale_income += 1
            sells += 1
        elif action.kind in ("buy", "merge"):
            source = offers.pop(action.slot)
            gold -= 3
            gross_spent += 3
            pending_roll = False
            if action.kind == "buy":
                team[action.target], copies[action.target] = source, 1
                buys += 1
            else:
                target = team[action.target]
                copies[action.target] += 1
                new_level = 3 if copies[action.target] >= 6 else 2 if copies[action.target] >= 3 else 1
                team[action.target] = replace(target, attack=min(50, max(source.attack, target.attack) + 1),
                                              health=min(50, max(source.health, target.health) + 1), level=new_level)
                merges += 1
                if new_level > target.level:
                    offers = [pet for pet in offers if pet.occupied]
                    offers.append(stream.levelup_offer(levelups))
                    levelups += 1
            offers = [pet for pet in offers if pet.occupied]
            offers += [EMPTY] * (stream.size - len(offers))
        else:
            violations.append({"unsupported_action": action.kind})
            reason = "unsupported_action"
            break
        board = replace(board, gold=gold, team=team, shop=offers)
    unproductive_rolls += int(pending_roll)
    metrics = {"initial_strength": strength(stream.initial), "final_strength": strength(board),
               "stat_gain": strength(board) - strength(stream.initial), "initial_gold": stream.initial.gold,
               "leftover_gold": board.gold, "gross_spent": gross_spent, "sale_income": sale_income,
               "net_spent": gross_spent - sale_income, "rolls": rolls, "unproductive_rolls": unproductive_rolls,
               "buys": buys, "merges": merges, "sells": sells, "levelups": levelups,
               "reason": reason, "violations": violations, "actions": actions}
    if trace:
        metrics.update(trace=history, final_board=board.to_dict())
    return metrics


def run_pair(seed, *, trace=False):
    stream = OfferStream(seed)
    return {name: run_shop(stream, make_policy(name), trace=trace) for name in VARIANTS}


METRICS = ("initial_strength", "final_strength", "stat_gain", "initial_gold", "leftover_gold", "gross_spent",
           "sale_income", "net_spent", "rolls", "unproductive_rolls", "buys", "merges", "sells", "levelups", "actions")


def summarize(report):
    count = report["pairs"]
    for totals in report["variants"].values():
        delta = totals["paired_gain_sum"] / count if count else 0.0
        variance = max(0.0, (totals["paired_gain_square_sum"] - count * delta * delta) / (count - 1)) if count > 1 else 0.0
        half_width = 1.96 * math.sqrt(variance / count) if count else 0.0
        totals["summary"] = {"mean_stat_gain": totals["stat_gain"] / count if count else 0.0,
                             "mean_leftover_gold": totals["leftover_gold"] / count if count else 0.0,
                             "mean_rolls": totals["rolls"] / count if count else 0.0,
                             "stat_gain_per_net_gold": totals["stat_gain"] / totals["net_spent"] if totals["net_spent"] else None,
                             "mean_paired_stat_gain": delta, "paired_mean_ci95": [delta - half_width, delta + half_width]}
    return report


def evaluate(*, output, pairs=50000, seconds=None, seed=0, checkpoint_seconds=30, resume=False, stop_file=None):
    if type(pairs) is not int or pairs < 1 or type(seed) is not int or seed < 0:
        raise ValueError("pairs must be positive and seed nonnegative integers")
    for name, value in (("seconds", seconds), ("checkpoint_seconds", checkpoint_seconds)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if checkpoint_seconds is None:
        raise ValueError("checkpoint_seconds is required")
    output = Path(output)
    signature = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (Path(__file__), ROOT / "desktop_state.py")}
    config = {"seed": seed, "variants": VARIANTS, "distribution": "synthetic_bounded_stats_v1", "max_actions": 50}
    if resume:
        report = json.loads(output.read_text())
        if report.get("schema_version") != 1 or report.get("config") != config or report.get("source_sha256") != signature:
            raise ValueError("resume requires the same schema, configuration, and policy/source files")
    else:
        if output.exists():
            raise ValueError("output exists; use --resume or another path")
        report = {"schema_version": 1, "kind": "paired_desktop_shop_economy", "config": config, "source_sha256": signature,
                  "limitations": "Synthetic stat utility only; excludes pet abilities, food, equipment, freezing, and combat. No win-rate claim.",
                  "pairs": 0, "next_seed": seed, "elapsed_seconds": 0.0, "failure_examples": [],
                  "variants": {name: {**dict.fromkeys(METRICS, 0), "legal_violations": 0, "truncations": 0,
                                       "unexpected_stops": 0, "paired_gain_sum": 0, "paired_gain_square_sum": 0,
                                       "better": 0, "equal": 0, "worse": 0} for name in VARIANTS}}
    started = last_saved = time.monotonic()
    previous_elapsed = report["elapsed_seconds"]
    reason = "pair_limit"

    def checkpoint():
        report["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        report["stop_reason"] = reason
        save_report(output, summarize(report))

    try:
        for _ in range(pairs):
            if stop_file and Path(stop_file).exists():
                reason = "stop_file"
                break
            if seconds is not None and time.monotonic() - started >= seconds:
                reason = "time_limit"
                break
            results = run_pair(report["next_seed"])
            # Build a whole completed pair before committing its counters. An
            # interrupt during a candidate leaves this seed available for replay.
            updated = {}
            failed = False
            for name, result in results.items():
                totals = dict(report["variants"][name])
                for metric in METRICS:
                    totals[metric] += result[metric]
                totals["legal_violations"] += len(result["violations"])
                totals["truncations"] += result["reason"] == "action_limit"
                totals["unexpected_stops"] += result["reason"] not in ("end_turn", "action_limit")
                difference = result["stat_gain"] - results["baseline"]["stat_gain"]
                totals["paired_gain_sum"] += difference
                totals["paired_gain_square_sum"] += difference * difference
                totals["better" if difference > 0 else "worse" if difference < 0 else "equal"] += 1
                failed |= bool(result["violations"]) or result["reason"] != "end_turn"
                updated[name] = totals
            failures = report["failure_examples"]
            if failed and len(failures) < 10:
                failures = failures + [{"seed": report["next_seed"], "results": results}]
            report = {**report, "variants": updated, "pairs": report["pairs"] + 1,
                      "next_seed": report["next_seed"] + 1, "failure_examples": failures}
            if time.monotonic() - last_saved >= checkpoint_seconds:
                reason = "running"
                checkpoint()
                last_saved = time.monotonic()
                print(json.dumps({"pairs": report["pairs"], "next_seed": report["next_seed"],
                                  "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        else:
            reason = "pair_limit"
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=".local/desktop/economy.json")
    parser.add_argument("--pairs", type=int, default=50000, help="additional paired seeds this invocation")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-seconds", type=float, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file")
    parser.add_argument("--replay", type=int, metavar="SEED")
    args = parser.parse_args(argv)
    try:
        if args.replay is not None:
            print(json.dumps(run_pair(args.replay, trace=True), indent=2))
            return 0
        report = evaluate(**{key: value for key, value in vars(args).items() if key != "replay"})
        print(json.dumps({"pairs": report["pairs"], "stop_reason": report["stop_reason"],
                          "variants": {name: totals["summary"] for name, totals in report["variants"].items()}}, indent=2))
        return int(any(totals["legal_violations"] or totals["truncations"] or totals["unexpected_stops"]
                       for totals in report["variants"].values()))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
