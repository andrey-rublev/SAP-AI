"""Offline candidate for allocating a visible shop's replacement budget.

This module is deliberately outside the live launcher and policy defaults.
It consumes the real Board and returns ordinary desktop Actions. Its objective
is attack + health only; food, abilities, experience and combat are unmodeled.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sap_ai.desktop.state import Action, Board, DesktopPolicy, observed_sale_income


@dataclass(frozen=True)
class ReplacementPlan:
    """A sequence of distinct level-one replacements using visible offers."""

    stat_gain: int = 0
    net_cost: int = 0
    targets: tuple[int, ...] = ()


def visible_replacement_plan(board: Board, *, minimum_upgrade_gain: int = 4) -> ReplacementPlan:
    """Optimize a full team's supported sale/purchase sequences in a small DP.

    Offers are consumed in descending observed strength, exactly as the parent
    policy buys after a sale. Higher-strength offers have the same price and
    dominate weaker offers for this stat-only objective. Every original target
    can be replaced once; selling a newly bought pet cannot improve this model.
    No merge or sale of an unknown-level/leveled pet is simulated.
    """
    guard = DesktopPolicy(minimum_upgrade_gain=minimum_upgrade_gain).choose_action(board)
    if guard is None or not all(pet.occupied is True for pet in board.team):
        return ReplacementPlan()
    offers = sorted((pet for pet in board.shop if pet.occupied),
                    key=lambda pet: -pet.strength)
    targets = tuple((index, pet.strength, 3 - income)
                    for index, pet in enumerate(board.team)
                    if pet.level == 1 and (income := observed_sale_income(pet)) is not None)

    @lru_cache(maxsize=None)
    def search(offer_index: int, available: int, gold: int) -> ReplacementPlan:
        best = ReplacementPlan()
        if offer_index == len(offers):
            return best
        for target_index, (slot, old_strength, cost) in enumerate(targets):
            if not available & (1 << target_index) or cost > gold:
                continue
            gain = offers[offer_index].strength - old_strength
            if gain < minimum_upgrade_gain:
                continue
            tail = search(offer_index + 1, available ^ (1 << target_index), gold - cost)
            candidate = ReplacementPlan(gain + tail.stat_gain, cost + tail.net_cost,
                                        (slot, *tail.targets))
            # Keep gold on equal utility, then use the lowest slot sequence.
            if ((candidate.stat_gain, -candidate.net_cost) > (best.stat_gain, -best.net_cost)
                    or ((candidate.stat_gain, candidate.net_cost) == (best.stat_gain, best.net_cost)
                        and candidate.targets < best.targets)):
                best = candidate
        return best

    return search(0, (1 << len(targets)) - 1, board.gold)


class VisibleReplacementPolicy(DesktopPolicy):
    """Sandbox policy; preserve production guards, empty fills and merge choices.

    The first sell of a planned replacement is sent through the normal session.
    An observed empty slot then makes the parent buy the strongest offer. The
    plan is recomputed from fresh observations; it is never an input macro.
    """

    def choose_action(self, board: Board) -> Action | None:
        original = super().choose_action(board)
        if original is None or original.kind in {"buy", "merge"}:
            return original
        plan = visible_replacement_plan(board, minimum_upgrade_gain=self.minimum_upgrade_gain)
        if not plan.targets:
            return original
        # Keep the observed policy's ordering when planning adds no utility.
        # This reference needs no merge/XP prediction: these are only full-team
        # replacements with offers consumed in the parent's purchase order.
        offers = sorted((pet for pet in board.shop if pet.occupied), key=lambda pet: -pet.strength)
        remaining = {index: pet for index, pet in enumerate(board.team)
                     if pet.level == 1 and observed_sale_income(pet) is not None}
        gold, gain, cost = board.gold, 0, 0
        for offered in offers:
            affordable = [(index, pet) for index, pet in remaining.items()
                          if gold + observed_sale_income(pet) >= 3]
            if not affordable:
                break
            target, pet = min(affordable, key=lambda item: (item[1].strength, item[0]))
            improvement = offered.strength - pet.strength
            if improvement < self.minimum_upgrade_gain:
                break
            paid = 3 - observed_sale_income(pet)
            gain, cost, gold = gain + improvement, cost + paid, gold - paid
            del remaining[target]
        if (plan.stat_gain, -plan.net_cost) <= (gain, -cost):
            return original
        return Action("sell", plan.targets[0])
