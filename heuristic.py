"""A deterministic baseline for the built-in scalar-strength toy arena.

The policy optimizes strength gained per purchase, respects action masks, and
uses spare gold to look for useful purchases. Its assumptions do not model
real Super Auto Pets abilities or positional combat.
"""
from __future__ import annotations

import numpy as np

from game import SuperAutoPetsEnv


class HeuristicAgent:
    def __init__(self, buy_cost: int = SuperAutoPetsEnv.BUY_COST, n_shop: int = SuperAutoPetsEnv.SHOP_SLOTS):
        if n_shop != SuperAutoPetsEnv.SHOP_SLOTS:
            raise ValueError("this policy requires the arena's three shop slots")
        if buy_cost != SuperAutoPetsEnv.BUY_COST:
            raise ValueError("buy_cost must match the arena's purchase cost")
        self.buy_cost = buy_cost
        self.n_shop = n_shop

    def choose_action(self, obs, greedy: bool = True, mask=None) -> int:
        """Choose a legal strength improvement; use stable slot-order tie breaks."""
        obs = np.asarray(obs, dtype=float)
        legal = SuperAutoPetsEnv.mask_from_obs(obs)
        if mask is not None:
            supplied = np.asarray(mask)
            if supplied.shape != (6,) or not np.isin(supplied, [False, True]).all():
                raise ValueError("mask must contain six boolean values")
            legal &= supplied.astype(bool)
        if not legal.any():
            raise ValueError("no legal actions are available")
        gold, shop, team = obs[0], obs[2:5], obs[5:10]
        pets = team[team > 0]
        if obs.size >= 12 and legal[5]:
            # Once this board beats even the strongest possible opponents for
            # every remaining win, extra shop actions only spend reward/time.
            final_winning_turn = obs[1] + SuperAutoPetsEnv.WINS_TO_WIN - obs[10] - 1
            strongest_remaining_enemy = 6 + 2 * final_winning_turn
            if team.sum() > strongest_remaining_enemy:
                return 5
        gains = [
            (1.0 if pet in pets else float(pet)) if legal[slot] else -np.inf
            for slot, pet in enumerate(shop)
        ]
        best_buy = int(np.argmax(gains))
        best_gain = gains[best_buy]

        # Selling is useful only if the subsequent purchase actually replaces
        # the lost strength. A matching remaining pet would merely gain +1.
        upgrade_gain = -np.inf
        if legal[4] and gold + SuperAutoPetsEnv.SELL_VALUE >= self.buy_cost:
            weakest_index = int(np.flatnonzero(team == pets.min())[0])
            remaining = np.delete(team, weakest_index)
            for pet in shop[shop > 0]:
                added = 1.0 if pet in remaining else float(pet)
                upgrade_gain = max(upgrade_gain, added - float(pets.min()))
        if upgrade_gain > max(0, best_gain):
            return 4
        if best_gain > 0:
            return best_buy

        # A refresh is worthwhile only if a subsequent purchase is affordable
        # and at least one possible shop pet could improve this team.
        can_improve = len(pets) < SuperAutoPetsEnv.TEAM_SLOTS or bool(
            (pets <= SuperAutoPetsEnv.MAX_TIER).any()
        )
        if legal[3] and gold >= self.buy_cost + SuperAutoPetsEnv.ROLL_COST and can_improve:
            return 3
        if legal[5]:
            return 5
        # External masks may disable the preferred end-turn action. Honor them
        # even when the only allowed action offers no strategic improvement.
        return int(np.flatnonzero(legal)[0])


if __name__ == "__main__":
    from main import evaluate

    env = SuperAutoPetsEnv()
    rate, wins = evaluate(env, agent=HeuristicAgent(), episodes=500)
    print(f"Heuristic agent : run-win rate {rate:5.1%} | avg wins/run {wins:4.2f}")
