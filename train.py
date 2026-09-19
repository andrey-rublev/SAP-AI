"""Train a Q-learning agent against the real Super Auto Pets rules.

This uses the ``sapai`` package (the community Super Auto Pets simulator) so the
agent learns against authentic pets and battle resolution.  Install it with
``pip install sapai`` (it is listed in requirements.txt).

For a dependency-free demo that trains against the built-in surrogate
environment instead, run ``python main.py``.
"""
from __future__ import annotations

import sys

try:
    from sapai.pets import Pet
    from sapai.teams import Team
    from sapai.battle import Battle
except ModuleNotFoundError:
    sys.exit(
        "This trainer needs the 'sapai' package.\n"
        "  Install it with:  pip install sapai\n"
        "  Or run 'python main.py' for the dependency-free demo."
    )

from agent import QLearningAgent

ACTIONS = ["buy", "roll", "battle"]


class SAPEnvironment:
    """A minimal sapai-backed environment exposing a classic RL step API."""

    def __init__(self):
        self.team = Team([])
        self.enemy_team = Team(["sheep", "tiger"])

    def reset(self):
        """Reset to an empty team and a fixed enemy; state is the team string."""
        self.team = Team([])
        self.enemy_team = Team(["sheep", "tiger"])
        return str(self.team)

    def step(self, action):
        """Apply an action and return ``(state, reward, done, info)``."""
        reward, done = 0.0, False

        if action == "buy":
            pet = Pet("ant")
            for i in range(5):  # place the pet in the first empty slot
                if self.team.get_slot(i).is_empty():
                    self.team.move(pet, i)
                    reward = 1.0
                    break

        elif action == "roll":
            reward = 0.0  # refreshing the shop has no immediate reward

        elif action == "battle":
            battle = Battle(self.team, self.enemy_team)
            winner = battle.battle()
            reward = {0: 10.0, 1: -10.0}.get(winner, 5.0)  # win / lose / draw
            done = True

        return str(self.team), reward, done, {}


def main(num_episodes: int = 1000):
    env = SAPEnvironment()
    agent = QLearningAgent(ACTIONS)

    print(f"Training for {num_episodes} episodes against the sapai rules ...")
    agent.train(env, num_episodes=num_episodes, log_every=max(1, num_episodes // 10))

    # Demonstrate the learned greedy policy on a fresh episode.
    print("\nGreedy rollout with the learned policy:")
    state, done = env.reset(), False
    while not done:
        action = agent.choose_action(state, greedy=True)
        state, reward, done, _ = env.step(action)
        print(f"  action={action:<6s} reward={reward:+.1f}")

    agent.save("qtable_sapai.json")
    print("\nSaved learned Q-table to qtable_sapai.json")


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(episodes)
