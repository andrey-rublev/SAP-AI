"""Train tabular Q-learning on the small, built-in SAP-inspired arena."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent import QLearningAgent
from evaluation import evaluate_policy, positive_int
from game import SuperAutoPetsEnv


def compact_state(obs):
    """Compact v2 features preserving which shop slot each buy action refers to.

    The old encoder used only max(shop), giving different shopping decisions the
    same key. Each slot now retains its strength and whether it combines. Team
    strength and battle pressure are bucketed to keep the table manageable.
    This is still a lossy abstraction; the DQN receives the full observation.
    """
    obs = np.asarray(obs)
    gold, turn = int(obs[0]), int(obs[1])
    shop, team = obs[2:5].astype(int), obs[5:10].astype(int)
    occupied = team[team > 0]
    strength = int(team.sum())
    pressure = max(-3, min(3, (strength - (3 + 2 * turn)) // 3))
    slots = tuple(int(pet) + (7 if pet > 0 and pet in occupied else 0) for pet in shop)
    return (
        min(gold, 10), pressure, len(occupied),
        int(occupied.min()) if len(occupied) else 0,
        *slots,
    )


compact_state.version = 2


def evaluate(env, agent=None, episodes=200, max_steps=500, seed=10_000):
    """Backward-compatible pair; use evaluate_policy for diagnostics and CIs."""
    result = evaluate_policy(env, agent, episodes, max_steps, seed)
    return result.win_rate, result.avg_wins


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=positive_int, default=4000)
    p.add_argument("--eval-episodes", type=positive_int, default=200)
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--eval-seed", type=int, default=10_000, help="first held-out evaluation seed")
    p.add_argument("--max-steps", type=positive_int, default=500)
    p.add_argument("--epsilon-decay", type=float, default=0.999)
    p.add_argument("--output", default="qtable.json")
    p.add_argument("--resume", metavar="PATH", help="continue from a v2 Q-table")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--plot", metavar="PATH", help="save training-reward plot")
    p.add_argument("--report", metavar="PATH", help="write evaluation details and rewards as JSON")
    args = p.parse_args(argv)
    if args.seed < 0 or args.eval_seed < 0:
        p.error("seeds must be non-negative")
    if not 0 < args.epsilon_decay <= 1:
        p.error("--epsilon-decay must be in (0, 1]")
    return args


def _describe(label, result):
    low, high = result.win_rate_ci95
    print(f"{label:<16}: win rate {result.win_rate:5.1%} "
          f"(95% CI {low:.1%}-{high:.1%}) | avg wins {result.avg_wins:.2f} "
          f"| truncated {result.truncations}")


def main(argv=None):
    args = parse_args(argv)
    env, eval_env = SuperAutoPetsEnv(), SuperAutoPetsEnv()
    try:
        print("SAP-AI | tabular Q-learning | simplified arena")
        baseline = evaluate_policy(eval_env, episodes=args.eval_episodes,
                                   max_steps=args.max_steps, seed=args.eval_seed)
        _describe("Legal random", baseline)
        agent = QLearningAgent(range(env.action_space.n), epsilon_decay=args.epsilon_decay,
                               state_fn=compact_state, seed=args.seed)
        if args.resume:
            agent.load(args.resume)
            if agent.actions != list(range(env.action_space.n)):
                raise ValueError("Q-table must use the arena's action order [0, 1, 2, 3, 4, 5]")
        print(f"Training {args.episodes} episodes (seed {args.seed}) ...")
        history = agent.train(env, num_episodes=args.episodes, max_steps=args.max_steps,
                              log_every=max(1, args.episodes // 4), seed=args.seed)
        result = evaluate_policy(eval_env, agent, episodes=args.eval_episodes,
                                 max_steps=args.max_steps, seed=args.eval_seed)
        _describe("Learned policy", result)
        if args.plot:
            from metrics import plot_training_curve
            Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
            plot_training_curve(history, args.plot, title="Q-learning training reward")
            print(f"Saved training curve to {args.plot}")
        if not args.no_save:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            agent.save(args.output)
            print(f"Saved Q-table to {args.output}")
        if args.report:
            report = {"schema_version": 1, "environment": "simplified_arena_v2",
                      "config": vars(args), "training_rewards": history,
                      "random": baseline.to_dict(), "tabular": result.to_dict()}
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved report to {args.report}")
        return result
    finally:
        env.close()
        eval_env.close()


def cli():
    main()


if __name__ == "__main__":
    main()
