"""A rule-based Super Auto Pets bot.

``HeuristicAgent`` plays sensible, hand-written strategy against
:class:`game.SuperAutoPetsEnv` without any training.  It is useful as a strong
baseline to compare the learned agent against, and as a safe default policy for
driving the live client.

Its :meth:`choose_action` signature matches :class:`agent.QLearningAgent`, so it
is a drop-in wherever an agent is expected.
"""
from __future__ import annotations

import numpy as np

from game import SuperAutoPetsEnv


class HeuristicAgent:
    def __init__(self, buy_cost: int = SuperAutoPetsEnv.BUY_COST, n_shop: int = SuperAutoPetsEnv.SHOP_SLOTS):
        self.buy_cost = buy_cost
        self.n_shop = n_shop

    def choose_action(self, obs, greedy: bool = True) -> int:
        obs = np.asarray(obs)
        gold = int(round(obs[0]))
        shop = [int(round(x)) for x in obs[2 : 2 + self.n_shop]]
        team = [int(round(x)) for x in obs[2 + self.n_shop :]]
        team_pets = [t for t in team if t > 0]

        if gold >= self.buy_cost:
            # 1) Combine: buy a shop pet that matches one already on the team.
            for slot, pet in enumerate(shop):
                if pet > 0 and pet in team_pets:
                    return slot
            # 2) Fill an empty slot with the strongest available shop pet.
            if 0 in team and any(p > 0 for p in shop):
                return int(max(range(self.n_shop), key=lambda i: shop[i]))
            # 3) Team full: sell the weakest pet if the shop offers an upgrade.
            if team_pets and any(p > min(team_pets) for p in shop):
                return 4  # sell weakest
        # 4) Nothing worth buying -> fight.
        return 5


if __name__ == "__main__":
    from main import evaluate

    env = SuperAutoPetsEnv()
    rate, wins = evaluate(env, agent=HeuristicAgent(), episodes=500)
    print(f"Heuristic agent : run-win rate {rate:5.1%} | avg wins/run {wins:4.2f}")
