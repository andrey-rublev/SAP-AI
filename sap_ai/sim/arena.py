"""Offline arena on the real bot's Board/Action schema, scored in trophies.

Unlike the scalar toy env, this uses the desktop bot's own types, so the exact
``DesktopPolicy`` that plays the Steam client can be compared offline. It is an
approximation: base stats are close to the Turtle pack, shops unlock tiers and
slots by turn, merges add +1/+1 and levels, and combat is ability-free
front-to-back trading. Each run fights ghost teams built by an opponent policy
in its own shop, as Arena mode matches recorded teams of the same turn.

    python -m sap_ai.sim.arena --runs 300
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import random
from statistics import mean

from sap_ai.desktop.state import Action, Board, DesktopPolicy, PetSlot, Phase, legal_action, observed_sale_income


# (attack, health) by tier; abilities are not modelled.
PETS = {
    1: {"ant": (2, 1), "beaver": (3, 2), "cricket": (1, 2), "duck": (2, 3), "fish": (2, 2),
        "horse": (2, 1), "mosquito": (2, 2), "otter": (1, 3), "pig": (4, 1)},
    2: {"crab": (4, 1), "dodo": (2, 3), "elephant": (3, 5), "flamingo": (3, 2), "hedgehog": (3, 2),
        "peacock": (2, 5), "rat": (4, 5), "shrimp": (2, 3), "spider": (2, 2), "swan": (1, 2)},
    3: {"badger": (5, 4), "blowfish": (3, 5), "camel": (2, 6), "dog": (3, 3), "giraffe": (2, 5),
        "kangaroo": (2, 3), "ox": (1, 3), "rabbit": (1, 2), "sheep": (2, 2), "snail": (2, 2)},
    4: {"bison": (4, 4), "deer": (1, 1), "dolphin": (4, 3), "hippo": (4, 5), "parrot": (4, 2),
        "penguin": (1, 3), "rooster": (5, 3), "skunk": (3, 5), "squirrel": (2, 5), "worm": (3, 3)},
    5: {"cow": (4, 6), "crocodile": (8, 4), "monkey": (1, 2), "rhino": (5, 8), "scorpion": (1, 1),
        "seal": (3, 8), "shark": (4, 4), "turkey": (3, 4)},
    6: {"boar": (10, 6), "cat": (4, 5), "dragon": (6, 8), "fly": (5, 5), "gorilla": (6, 9),
        "leopard": (10, 4), "mammoth": (3, 10), "snake": (6, 6), "tiger": (4, 3)},
}
EMPTY = PetSlot(False)
START_LIVES, TROPHIES_TO_WIN, TURN_GOLD, MAX_TURNS, MAX_SHOP_ACTIONS = 5, 10, 10, 30, 60


def max_tier(turn: int) -> int:
    return min(6, (turn + 1) // 2)


def shop_size(turn: int) -> int:
    return 3 if turn <= 4 else 4 if turn <= 8 else 5


def lives_lost(turn: int) -> int:
    return 1 if turn <= 2 else 2 if turn <= 4 else 3


@dataclass
class _Pet:
    species: str
    attack: int
    health: int
    experience: int = 0

    @property
    def level(self) -> int:
        return 1 + (self.experience >= 2) + (self.experience >= 5)

    def slot(self) -> PetSlot:
        return PetSlot(True, self.species, self.attack, self.health, self.level)


class Shop:
    """One player's shop phase; the board it exposes is what the bot would see."""

    def __init__(self, rng: random.Random):
        self.rng, self.team, self.offers = rng, [None] * 5, []
        self.turn = self.gold = 0

    def _roll(self):
        pool = [(name, stats) for tier in range(1, max_tier(self.turn) + 1) for name, stats in PETS[tier].items()]
        self.offers = [_Pet(name, *stats) for name, stats in self.rng.choices(pool, k=shop_size(self.turn))]

    def start_turn(self, turn: int):
        self.turn, self.gold = turn, TURN_GOLD
        self._roll()

    def board(self, wins: int, lives: int) -> Board:
        shop = tuple(pet.slot() if pet else EMPTY for pet in self.offers)
        shop += (EMPTY,) * (shop_size(self.turn) - len(shop))
        return Board(Phase.SHOP, gold=self.gold, turn=self.turn, wins=wins, lives=lives, shop=shop,
                     team=tuple(pet.slot() if pet else EMPTY for pet in self.team))

    def apply(self, board: Board, action: Action) -> bool:
        """Apply a legal action; return False when the shop phase is over."""
        if action.kind == "end_turn" or not legal_action(board, action):
            return False
        if action.kind == "roll":
            self.gold -= 1
            self._roll()
        elif action.kind == "sell":
            self.gold = min(self.gold + observed_sale_income(board.team[action.slot]), 20)
            self.team[action.slot] = None
        elif action.kind in ("buy", "merge"):
            self.gold -= 3
            pet = self.offers.pop(action.slot)
            if action.kind == "buy":
                self.team[action.target] = pet
            else:
                target = self.team[action.target]
                target.attack, target.health = target.attack + 1, target.health + 1
                target.experience += 1
        else:
            return False
        return True

    def play_turn(self, policy, wins: int, lives: int) -> None:
        for _ in range(MAX_SHOP_ACTIONS):
            board = self.board(wins, lives)
            action = policy.choose_action(board)
            if action is None or not self.apply(board, action):
                return


def battle(left, right) -> int:
    """+1 if ``left`` wins, -1 if it loses, 0 for a draw; slot 0 fights first."""
    a = [[p.attack, p.health] for p in left if p]
    b = [[p.attack, p.health] for p in right if p]
    while a and b:
        a[0][1] -= b[0][0]
        b[0][1] -= a[0][0]
        if a[0][1] <= 0:
            a.pop(0)
        if b[0][1] <= 0:
            b.pop(0)
    return (len(a) > 0) - (len(b) > 0)


def play_run(policy, opponent, seed: int) -> dict:
    """One arena run against per-turn ghost teams; returns trophies and turns."""
    player, ghost = Shop(random.Random(seed)), Shop(random.Random(seed + 1_000_003))
    wins, lives, turn = 0, START_LIVES, 0
    while wins < TROPHIES_TO_WIN and lives > 0 and turn < MAX_TURNS:
        turn += 1
        for shop, chooser in ((player, policy), (ghost, opponent)):
            shop.start_turn(turn)
            shop.play_turn(chooser, wins, lives)
        outcome = battle(player.team, ghost.team)
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            lives -= lives_lost(turn)
    return {"trophies": wins, "turns": turn, "victory": wins >= TROPHIES_TO_WIN}


class SpeciesBlind:
    """Hide species from a policy, as before hover identification (7.4% read)."""

    def __init__(self, policy):
        self.policy = policy

    def choose_action(self, board: Board):
        def blind(slots):
            return tuple(replace(pet, species=None) if pet.occupied else pet for pet in slots)
        return self.policy.choose_action(replace(board, shop=blind(board.shop), team=blind(board.team)))


def compare(policies: dict, runs: int, seed: int = 0, opponent=None) -> dict:
    opponent = opponent or DesktopPolicy.recorded_baseline()
    report = {}
    for name, policy in policies.items():
        results = [play_run(policy, opponent, seed + index) for index in range(runs)]
        report[name] = {"mean_trophies": round(mean(r["trophies"] for r in results), 2),
                        "victories": sum(r["victory"] for r in results), "runs": runs,
                        "mean_turns": round(mean(r["turns"] for r in results), 2)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare desktop policies offline, in trophies.")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = compare({"current": DesktopPolicy(), "recorded_baseline": DesktopPolicy.recorded_baseline(),
                      "baseline_no_species": SpeciesBlind(DesktopPolicy.recorded_baseline())},
                     args.runs, args.seed)
    for name, row in report.items():
        print(f"{name:20} mean trophies {row['mean_trophies']:5.2f}  victories {row['victories']}/{row['runs']}"
              f"  mean turns {row['mean_turns']:5.2f}")
    return report


if __name__ == "__main__":
    main()
