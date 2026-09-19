"""Dependency-free demo of the SAP-AI stack.

Trains a :class:`~agent.QLearningAgent` against the built-in
:class:`~game.SuperAutoPetsEnv` surrogate and compares a random baseline with the
learned greedy policy.  Needs only ``gymnasium`` and ``numpy`` (no Steam game and
no ``sapai``), so it always runs::

    python main.py
"""
from __future__ import annotations

from agent import QLearningAgent
from game import SuperAutoPetsEnv


def compact_state(obs):
    """Compress the 10-dim observation into a small, learnable key.

    The raw observation lists every shop and team slot, which is far too large a
    state space for tabular Q-learning.  We summarise it as
    ``(gold, turn, team_strength, num_pets, best_shop_pet)``.
    """
    gold, turn = int(obs[0]), int(obs[1])
    shop, team = obs[2:5], obs[5:10]
    return (
        min(gold, 10),
        min(turn, 12),
        min(int(team.sum()), 40),
        int((team > 0).sum()),
        int(shop.max()),
    )


def evaluate(env, agent=None, episodes=200, max_steps=500):
    """Return ``(run_win_rate, avg_wins_per_run)`` over some episodes."""
    victories, total_wins = 0, 0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        for _ in range(max_steps):
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(obs, greedy=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        victories += info["wins"] >= env.WINS_TO_WIN
        total_wins += info["wins"]
    return victories / episodes, total_wins / episodes


def main():
    env = SuperAutoPetsEnv()
    actions = list(range(env.action_space.n))

    print("=" * 62)
    print(" SAP-AI - training a Q-learning agent on the built-in arena")
    print("=" * 62)

    # 1) Random baseline (also serves as an environment smoke test).
    base_rate, base_wins = evaluate(env, agent=None, episodes=200)
    print(f"\nRandom baseline : run-win rate {base_rate:5.1%} | avg wins/run {base_wins:4.2f}")

    # 2) Train.
    agent = QLearningAgent(actions, epsilon_decay=0.999, state_fn=compact_state, seed=0)
    print("\nTraining ...")
    agent.train(env, num_episodes=4000, log_every=1000)

    # 3) Evaluate the learned greedy policy.
    rate, wins = evaluate(env, agent=agent, episodes=200)
    print(f"\nLearned policy  : run-win rate {rate:5.1%} | avg wins/run {wins:4.2f}")
    print(
        f"Improvement     : {rate - base_rate:+.1%} run-win rate, "
        f"{wins - base_wins:+.2f} wins/run"
    )

    agent.save("qtable.json")
    print("\nSaved learned Q-table to qtable.json")


if __name__ == "__main__":
    main()
