"""Compare policies on identical reset seeds and export auditable JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent import QLearningAgent
from evaluation import evaluate_policy, positive_int
from game import SuperAutoPetsEnv
from heuristic import HeuristicAgent
from main import compact_state


def train_tabular(env, episodes: int, seed: int, max_steps: int = 500) -> QLearningAgent:
    agent = QLearningAgent(range(env.action_space.n), epsilon_decay=0.999,
                           state_fn=compact_state, seed=seed)
    agent.train(env, num_episodes=episodes, max_steps=max_steps, seed=seed)
    return agent


def run_report(episodes=4000, eval_episodes=300, seed=0, include_dqn=False,
               dqn_episodes=1500, eval_seed=10_000, max_steps=500):
    if episodes <= 0 or eval_episodes <= 0 or max_steps <= 0:
        raise ValueError("episode and step counts must be positive")
    if include_dqn and dqn_episodes <= 0:
        raise ValueError("dqn_episodes must be positive when DQN is enabled")
    if seed < 0 or eval_seed < 0:
        raise ValueError("seeds must be non-negative")
    env, eval_env = SuperAutoPetsEnv(), SuperAutoPetsEnv()
    try:
        policies = [("Random", None), ("Heuristic", HeuristicAgent()),
                    ("Tabular Q", train_tabular(env, episodes, seed, max_steps))]
        if include_dqn:
            from dqn import DQNAgent
            dqn = DQNAgent(obs_dim=env.observation_space.shape[0],
                           n_actions=env.action_space.n,
                           obs_scale=env.observation_space.high, seed=seed)
            dqn.train(env, num_episodes=dqn_episodes, max_steps=max_steps, seed=seed)
            policies.append(("DQN", dqn))
        results = []
        for name, policy in policies:
            result = evaluate_policy(eval_env, policy, eval_episodes, max_steps, eval_seed)
            results.append({"agent": name, **result.to_dict()})
        return {
            "schema_version": 1,
            "environment": "simplified_arena_v2",
            "config": {"training_episodes": episodes, "evaluation_episodes": eval_episodes,
                       "training_seed": seed, "evaluation_seed": eval_seed,
                       "max_steps": max_steps, "include_dqn": include_dqn,
                       "dqn_episodes": dqn_episodes if include_dqn else None},
            "results": results,
        }
    finally:
        env.close()
        eval_env.close()


def run(episodes: int, eval_episodes: int, seed: int, include_dqn: bool, dqn_episodes: int,
        *, eval_seed=10_000, max_steps=500):
    """Return legacy (name, win_rate, avg_wins) rows."""
    report = run_report(episodes, eval_episodes, seed, include_dqn, dqn_episodes,
                        eval_seed, max_steps)
    return [(row["agent"], row["win_rate"], row["avg_wins"]) for row in report["results"]]


def print_table(rows):
    print(f"{'agent':<12}{'run-win':>10}{'avg wins':>11}")
    print("-" * 33)
    for name, rate, wins in rows:
        print(f"{name:<12}{rate:>9.1%}{wins:>11.2f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=positive_int, default=4000)
    p.add_argument("--eval-episodes", type=positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=10_000)
    p.add_argument("--max-steps", type=positive_int, default=500)
    p.add_argument("--dqn", action="store_true", help="also train and evaluate Double DQN")
    p.add_argument("--dqn-episodes", type=positive_int, default=1500)
    p.add_argument("--output", metavar="PATH", help="write metrics and individual episodes as JSON")
    args = p.parse_args(argv)
    if args.seed < 0 or args.eval_seed < 0:
        p.error("seeds must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = run_report(args.episodes, args.eval_episodes, args.seed, args.dqn,
                        args.dqn_episodes, args.eval_seed, args.max_steps)
    rows = [(r["agent"], r["win_rate"], r["avg_wins"]) for r in report["results"]]
    print_table(rows)
    print(f"\nEvaluation reset seeds: {args.eval_seed}..{args.eval_seed + args.eval_episodes - 1}")
    for row in report["results"]:
        lo, hi = row["win_rate_ci95"]
        print(f"{row['agent']:<12} 95% CI {lo:.1%}-{hi:.1%}; "
              f"avg reward {row['avg_reward']:.2f}; truncated {row['truncations']}")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved benchmark to {path}")
    return rows


def cli():
    main()


if __name__ == "__main__":
    main()
