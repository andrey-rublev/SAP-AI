"""Dependency-free demo of the SAP-AI stack.

Trains a :class:`~agent.QLearningAgent` against the built-in
:class:`~game.SuperAutoPetsEnv` surrogate and compares a random baseline with the
learned greedy policy.  Needs only ``gymnasium`` and ``numpy`` (no Steam game and
no ``sapai``), so it always runs::

    python main.py
"""
from __future__ import annotations

import argparse

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
        for _ in range(max_steps):
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(obs, greedy=True, mask=info.get("action_mask"))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        victories += info["wins"] >= env.WINS_TO_WIN
        total_wins += info["wins"]
    return victories / episodes, total_wins / episodes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a Q-learning agent on the built-in SAP arena.")
    p.add_argument("--episodes", type=int, default=4000, help="training episodes (default: 4000)")
    p.add_argument("--eval-episodes", type=int, default=200, help="episodes per evaluation (default: 200)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--epsilon-decay", type=float, default=0.999, help="per-episode epsilon decay (default: 0.999)")
    p.add_argument("--output", default="qtable.json", help="where to save the Q-table (default: qtable.json)")
    p.add_argument("--no-save", action="store_true", help="don't write the Q-table to disk")
    p.add_argument("--plot", metavar="PATH", help="save a training-reward curve PNG to PATH")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    env = SuperAutoPetsEnv()
    actions = list(range(env.action_space.n))

    print("=" * 62)
    print(" SAP-AI - training a Q-learning agent on the built-in arena")
    print("=" * 62)

    # 1) Random baseline (also serves as an environment smoke test).
    base_rate, base_wins = evaluate(env, agent=None, episodes=args.eval_episodes)
    print(f"\nRandom baseline : run-win rate {base_rate:5.1%} | avg wins/run {base_wins:4.2f}")

    # 2) Train.
    agent = QLearningAgent(
        actions, epsilon_decay=args.epsilon_decay, state_fn=compact_state, seed=args.seed
    )
    print(f"\nTraining for {args.episodes} episodes ...")
    history = agent.train(env, num_episodes=args.episodes, log_every=max(1, args.episodes // 4))

    if args.plot:
        from metrics import plot_training_curve

        plot_training_curve(history, args.plot, title="Q-learning training reward")
        print(f"Saved training curve to {args.plot}")

    # 3) Evaluate the learned greedy policy.
    rate, wins = evaluate(env, agent=agent, episodes=args.eval_episodes)
    print(f"\nLearned policy  : run-win rate {rate:5.1%} | avg wins/run {wins:4.2f}")
    print(
        f"Improvement     : {rate - base_rate:+.1%} run-win rate, "
        f"{wins - base_wins:+.2f} wins/run"
    )

    if not args.no_save:
        agent.save(args.output)
        print(f"\nSaved learned Q-table to {args.output}")


if __name__ == "__main__":
    main()
