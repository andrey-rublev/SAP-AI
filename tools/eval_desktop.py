"""Seeded, resumable CPU soak for the observed desktop policy and controller.

Examples (run from the repository checkout)::

    python tools/eval_desktop.py --cases 1000 --output reports/desktop-soak.json
    python tools/eval_desktop.py --seconds 7200 --resume --output reports/desktop-soak.json
    python tools/eval_desktop.py --replay 17 --scenario wrong_price

This is a controller contract test, not a Super Auto Pets combat simulator or
training job. It models ordinary three-gold purchases, ordered shop compaction,
one-gold rolls, level-based sales, duplicate combines, and shop/battle phases.
Abilities, food, combat, freezes, OCR accuracy, and actual mouse IO are excluded.
No toy DQN is loaded: its scalar observations and six actions cannot represent
Board uncertainty, pet identities, or explicit source/destination actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

# Also support direct execution from another directory without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_session import DesktopSession
from desktop_state import Action, Board, DesktopPolicy, PetSlot, Phase


SCENARIOS = (
    "nominal", "gaps", "duplicates", "unidentified", "transient",
    "ignored_input", "wrong_price", "wrong_target", "input_error",
    "ocr_timeout", "ocr_timeout_after_input", "unknown_phase",
    "unknown_slot", "missing_stat", "missing_gold", "capped_merge",
)
EMPTY = PetSlot(False)
SPECIES = ("ant", "fish", "beaver", "duck", "pig", "otter")
EXPECTED = {
    **{name: "result" for name in SCENARIOS[:5]},
    **{name: "action_timeout" for name in ("ignored_input", "wrong_price", "wrong_target", "capped_merge")},
    "input_error": "action_error", "ocr_timeout": "observation_error",
    "ocr_timeout_after_input": "observation_error", "unknown_phase": "unknown_timeout",
    **{name: "policy_stopped" for name in ("unknown_slot", "missing_stat", "missing_gold")},
}


class SyntheticDesktop:
    """Independent transition fixture; never imports a desktop/vision backend.

    Synthetic pets have plausible buffed stats, not modeled species abilities.
    Hidden copy counts distinguish partial experience from an actual level-up.
    Fault scenarios deliberately break one observation/input contract, expecting
    a bounded stop rather than another attempted action.
    """

    def __init__(self, seed: int, scenario: str, turns: int):
        self.rng = random.Random(seed)
        self.scenario, self.turns = scenario, turns
        self.now = 0.0
        self.rounds = 0
        self.actions = []
        self.frames = deque()
        self.violations = []
        self.last_observed = None
        self.stable = 0
        self.shop_size = self.rng.randint(3, 5)
        team = [self.pet(level=self.rng.randint(1, 3)) if self.rng.random() < 0.65 else EMPTY for _ in range(5)]
        self.board = Board(Phase.SHOP, gold=10, turn=self.rng.randint(1, 12),
                           shop=self.offers(), team=team)
        if scenario not in SCENARIOS[:5]:
            # Faults all begin with a guaranteed purchase, independent of RNG.
            self.board = replace(self.board, team=(EMPTY,) * 5)
        if scenario == "capped_merge":
            fish = PetSlot(True, "fish", 50, 50, 1)
            self.board = replace(self.board, gold=3, shop=(replace(fish, attack=2, health=3),) + (EMPTY,) * (self.shop_size - 1),
                                 team=(fish,) + (PetSlot(True, "ant", 20, 20, 3),) * 4)
        self.copies = [0 if not pet.occupied else self.rng.randint(1, 2) if pet.level == 1
                       else self.rng.randint(3, 5) if pet.level == 2 else 6 for pet in self.board.team]
        if scenario == "capped_merge":
            self.copies[0] = 1  # No visible stat or level change on this combine.
        self.initial = self.project(self.board)

    def pet(self, level=1):
        return PetSlot(True, self.rng.choice(SPECIES), self.rng.randint(1, 20), self.rng.randint(1, 20), level)

    def offers(self):
        if self.scenario == "duplicates":
            return (self.pet(),) * self.shop_size
        return tuple(self.pet() for _ in range(self.shop_size))

    def project(self, board):
        if self.scenario == "unidentified" and board.phase == Phase.SHOP:
            hide = lambda pets: tuple(replace(pet, species=None, level=None) if pet.occupied else pet for pet in pets)
            return replace(board, shop=hide(board.shop), team=hide(board.team))
        return board

    def sleep(self, seconds):
        self.now += seconds  # Virtual time exercises deadlines without real waiting.

    def observe(self):
        self.now += 0.05
        if self.scenario == "ocr_timeout" or (self.scenario == "ocr_timeout_after_input" and self.actions):
            raise TimeoutError("injected OCR subprocess timeout")
        current = self.frames.popleft() if self.frames else self.board
        current = self.project(current)
        if self.scenario == "unknown_phase":
            current = Board(Phase.UNKNOWN)
        elif self.scenario == "unknown_slot":
            current = replace(current, team=(PetSlot(None), *current.team[1:]))
        elif self.scenario == "missing_stat":
            current = replace(current, shop=(replace(current.shop[0], health=None), *current.shop[1:]))
        elif self.scenario == "missing_gold":
            current = replace(current, gold=None)
        self.stable = self.stable + 1 if current == self.last_observed else 1
        self.last_observed = current
        return current

    def require(self, condition, message):
        if not condition:
            self.violations.append(message)
            raise AssertionError(message)

    def act(self, action):
        """Check rules independently of desktop_state.legal_action."""
        before = self.board
        self.actions.append(action)
        self.require(self.stable >= 2, "input before two identical observations")
        self.require(not self.frames and self.last_observed == self.project(before), "input against stale/queued observation")
        self.require(before.phase == Phase.SHOP and before.gold is not None, "input outside known shop")
        self.require(isinstance(action, Action), "policy did not produce Action")
        # Runtime preflight can be slow; acknowledgments must get their own budget.
        self.now += self.rng.uniform(0.1, 12.0)
        if self.scenario == "input_error":
            raise RuntimeError("injected selection failure")
        if self.scenario == "ignored_input":
            return
        team, offers = list(before.team), list(before.shop)
        gold = before.gold
        if action.kind in ("buy", "merge"):
            i, j = action.slot, action.target
            self.require(i is not None and 0 <= i < len(offers) and j is not None and 0 <= j < len(team), "invalid purchase coordinates")
            source, target = offers[i], team[j]
            self.require(source.occupied is True and gold >= 3, "purchase without offer/budget")
            if action.kind == "buy":
                self.require(target.occupied is False, "buy targeted occupied teammate")
                destination = (j + 1) % 5 if self.scenario == "wrong_target" else j
                self.require(team[destination].occupied is False, "fixture wrong-target slot occupied")
                team[destination], self.copies[destination] = source, 1
            else:
                self.require(target.occupied is True and source.species == target.species and source.level == 1
                             and target.level in (1, 2), "incompatible species/level combine")
                self.copies[j] += 1
                level = 3 if self.copies[j] >= 6 else 2 if self.copies[j] >= 3 else 1
                team[j] = replace(target, attack=min(50, max(source.attack, target.attack) + 1),
                                  health=min(50, max(source.health, target.health) + 1), level=level)
            offers[i] = EMPTY
            if self.scenario != "gaps":
                offers = [pet for pet in offers if pet.occupied]
                offers += [EMPTY] * (self.shop_size - len(offers))
            gold -= 2 if self.scenario == "wrong_price" else 3
        elif action.kind == "sell":
            i = action.slot
            self.require(i is not None and 0 <= i < len(team) and action.target is None and team[i].occupied is True, "invalid sell target")
            gold += team[i].level
            team[i], self.copies[i] = EMPTY, 0
        elif action.kind == "roll":
            self.require(action.slot is None and action.target is None and gold >= 1, "invalid/unaffordable roll")
            gold -= 1
            # A refresh can legitimately leave exactly the same offers.
            offers = list(before.shop if self.rng.random() < 0.1 else self.offers())
        elif action.kind == "end_turn":
            self.require(action.slot is None and action.target is None, "malformed end-turn action")
            self.rounds += 1
        else:
            self.require(False, "unsupported desktop action")
        self.require(gold >= 0, "negative gold")
        self.board = replace(before, gold=gold, shop=offers, team=team)
        self.frames.extend([before] * self.rng.randint(0, 2))
        if self.scenario == "transient":
            self.frames.append(Board(Phase.UNKNOWN))
        if action.kind == "end_turn":
            self.frames.append(Board(Phase.BATTLE))
            self.board = (Board(Phase.RESULT) if self.rounds >= self.turns else
                          replace(self.board, gold=10, turn=before.turn + 1, shop=self.offers()))


def run_case(seed: int, *, scenario: str | None = None, turns: int = 5,
             minimum_upgrade_gain: int = 4, trace: bool = False, policy=None) -> dict:
    """Replay one independent seed. All returned fields are deterministic."""
    if type(seed) is not int or seed < 0 or type(turns) is not int or turns < 1:
        raise ValueError("seed must be nonnegative and turns positive integers")
    scenario = scenario or SCENARIOS[seed % len(SCENARIOS)]
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    fixture = SyntheticDesktop(seed, scenario, turns)
    events = deque(maxlen=128)
    digest = hashlib.sha256()
    proposed_before = proposed_action = expected_effect = None

    def event(data):
        nonlocal proposed_before, proposed_action, expected_effect
        # Canonical event bytes make replays comparable without storing all frames.
        digest.update(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
        events.append(data)
        if data["event"] == "proposed":
            proposed_before = Board.from_dict(data["board"])
            proposed_action = Action.from_dict(data["action"])
        elif data["event"] == "acted":
            # This is the independent fixture's applied transition, not an
            # expected state reconstructed by the production acknowledgment code.
            expected_effect = fixture.project(fixture.board)
        elif data["event"] == "acknowledged":
            observed = Board.from_dict(data["board"])
            fixture.require(proposed_action is not None and expected_effect is not None
                            and Action.from_dict(data["action"]) == proposed_action,
                            "acknowledgment has no matching applied action")
            if proposed_action.kind == "end_turn":
                verified = observed.phase == Phase.BATTLE or (
                    observed == expected_effect and observed.phase == Phase.SHOP
                    and observed.turn > proposed_before.turn
                )
            else:
                verified = observed == expected_effect
                cost = {"buy": -3, "merge": -3, "roll": -1}.get(proposed_action.kind)
                if proposed_action.kind == "sell":
                    cost = proposed_before.team[proposed_action.slot].level
                verified = verified and cost is not None and observed.gold == proposed_before.gold + cost
            fixture.require(verified, "acknowledged stale or incorrect action effect")
            proposed_before = proposed_action = expected_effect = None

    result = DesktopSession(fixture.observe, fixture.act, policy or DesktopPolicy(minimum_upgrade_gain=minimum_upgrade_gain),
                            max_actions=turns * 20 + 10, max_polls=turns * 200 + 100,
                            action_max_polls=20, max_unknown_polls=8,
                            clock=lambda: fixture.now, sleep=fixture.sleep, event_callback=event).run()
    # The controller catches Ctrl+C to release a real session cleanly. This
    # harness has no stop callback, so "stopped" means its case was interrupted,
    # not completed. Let evaluate retain that seed for the next invocation.
    if result.reason == "stopped":
        raise KeyboardInterrupt()
    violations = list(fixture.violations)
    if result.reason != EXPECTED[scenario]:
        violations.append(f"expected {EXPECTED[scenario]}, got {result.reason}")
    if scenario in SCENARIOS[:5]:
        if result.actions != result.acknowledgments:
            violations.append("healthy run has unacknowledged actions")
        if fixture.rounds != turns:
            violations.append("healthy run did not finish the requested shop rounds")
    else:
        expected_actions = 0 if scenario in ("ocr_timeout", "unknown_phase", "unknown_slot", "missing_stat", "missing_gold") else 1
        if result.actions != expected_actions or result.acknowledgments != 0:
            violations.append("fault scenario retried input or acknowledged an unverified effect")
    outcome = {"seed": seed, "scenario": scenario, "passed": not violations, "violations": violations,
               "expected_reason": EXPECTED[scenario], "reason": result.reason, "actions": result.actions,
               "acknowledgments": result.acknowledgments, "polls": result.polls,
               "shop_rounds": fixture.rounds, "action_kinds": dict(Counter(action.kind for action in fixture.actions)),
               "trace_sha256": digest.hexdigest(), "error": result.error}
    if trace or violations:
        outcome.update(initial_board=fixture.initial.to_dict(), result=result.to_dict(), trace=list(events))
    return outcome


def source_signature():
    files = (Path(__file__), ROOT / "desktop_state.py", ROOT / "desktop_session.py")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def save_report(path, report):
    """Atomic snapshots retain the last completed case after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def evaluate(*, output, cases=None, seconds=None, seed=0, turns=5, minimum_upgrade_gain=4,
             resume=False, checkpoint_seconds=30.0, stop_file=None, fail_fast=False):
    """Run additional cases until the count/time limit or stop-file is reached."""
    if cases is None and seconds is None:
        cases = 1000
    for name, value in (("cases", cases), ("turns", turns), ("minimum_upgrade_gain", minimum_upgrade_gain)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (("seconds", seconds), ("checkpoint_seconds", checkpoint_seconds)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    output = Path(output)
    config = {"seed": seed, "turns": turns, "minimum_upgrade_gain": minimum_upgrade_gain,
              "scenarios": list(SCENARIOS)}
    signature = source_signature()
    if resume:
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("schema_version") != 1 or report.get("config") != config or report.get("source_sha256") != signature:
            raise ValueError("resume requires matching report schema, seed/configuration, and source files")
    else:
        if output.exists():
            raise ValueError("output already exists; use --resume or a new output path")
        report = {"schema_version": 1, "kind": "synthetic_desktop_controller_soak", "config": config,
                  "limitations": "No real UI, OCR, abilities, food, combat, win rates, or model training.",
                  "source_sha256": signature, "next_seed": seed, "cases": 0, "passed": 0, "failed": 0,
                  "actions": 0, "acknowledgments": 0, "polls": 0, "max_case_polls": 0,
                  "scenarios": {}, "reasons": {}, "action_kinds": {}, "failure_examples": [], "elapsed_seconds": 0.0}
    started = checkpointed = time.monotonic()
    previous_elapsed = report["elapsed_seconds"]
    completed = 0
    stop_reason = "case_limit"

    def checkpoint():
        report["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        report["stop_reason"] = stop_reason
        save_report(output, report)

    try:
        while cases is None or completed < cases:
            if stop_file is not None and Path(stop_file).exists():
                stop_reason = "stop_file"
                break
            if seconds is not None and time.monotonic() - started >= seconds:
                stop_reason = "time_limit"
                break
            outcome = run_case(report["next_seed"], turns=turns, minimum_upgrade_gain=minimum_upgrade_gain)
            report["next_seed"] += 1
            completed += 1
            report["cases"] += 1
            report["passed" if outcome["passed"] else "failed"] += 1
            for name in ("actions", "acknowledgments", "polls"):
                report[name] += outcome[name]
            report["max_case_polls"] = max(report["max_case_polls"], outcome["polls"])
            for name, counts in (("scenarios", {outcome["scenario"]: 1}), ("reasons", {outcome["reason"]: 1}),
                                 ("action_kinds", outcome["action_kinds"])):
                for key, value in counts.items():
                    report[name][key] = report[name].get(key, 0) + value
            if not outcome["passed"]:
                if len(report["failure_examples"]) < 20:
                    report["failure_examples"].append(outcome)
                if fail_fast:
                    stop_reason = "failure"
                    break
            if time.monotonic() - checkpointed >= checkpoint_seconds:
                stop_reason = "running"
                checkpoint()
                checkpointed = time.monotonic()
                print(json.dumps({key: report[key] for key in ("cases", "failed", "next_seed", "elapsed_seconds")}), flush=True)
        else:
            stop_reason = "case_limit"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="reports/desktop-soak.json")
    parser.add_argument("--cases", type=int, help="additional cases this invocation; default 1000 without --seconds")
    parser.add_argument("--seconds", type=float, help="maximum wall-clock runtime; cases use virtual time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--turns", type=int, default=5, help="shop rounds per healthy scenario; no combat is simulated")
    parser.add_argument("--minimum-upgrade-gain", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-seconds", type=float, default=30)
    parser.add_argument("--stop-file", help="stop at a case boundary when this local path exists")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--replay", type=int, metavar="SEED", help="print one detailed deterministic case without writing a report")
    parser.add_argument("--scenario", choices=SCENARIOS, help="override scenario for --replay")
    args = parser.parse_args(argv)
    try:
        if args.replay is not None:
            result = run_case(args.replay, scenario=args.scenario, turns=args.turns,
                              minimum_upgrade_gain=args.minimum_upgrade_gain, trace=True)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        if args.scenario is not None:
            parser.error("--scenario requires --replay")
        report = evaluate(**{key: value for key, value in vars(args).items() if key not in ("replay", "scenario")})
        print(json.dumps({key: report[key] for key in ("cases", "passed", "failed", "next_seed", "stop_reason")}, indent=2))
        return 0 if report["failed"] == 0 else 1
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
