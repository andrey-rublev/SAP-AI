"""Compare agents on the built-in Super Auto Pets arena, side by side.

    python benchmark.py                 # random, heuristic, tabular Q-learning
    python benchmark.py --dqn           # also train + benchmark the DQN (slow)
"""
from __future__ import annotations

import argparse

from agent import QLearningAgent
from game import SuperAutoPetsEnv
from heuristic import HeuristicAgent
from main import compact_state, evaluate


def train_tabular(env, episodes: int, seed: int) -> QLearningAgent:
    agent = QLearningAgent(
        list(range(env.action_space.n)),
        epsilon_decay=0.999,
        state_fn=compact_state,
        seed=seed,
    )
    agent.train(env, num_episodes=episodes)
    return agent


def run(episodes: int, eval_episodes: int, seed: int, include_dqn: bool, dqn_episodes: int):
    env = SuperAutoPetsEnv()
    rows = []
    rows.append(("Random", *evaluate(env, None, eval_episodes)))
    rows.append(("Heuristic", *evaluate(env, HeuristicAgent(), eval_episodes)))
    rows.append(("Tabular Q", *evaluate(env, train_tabular(env, episodes, seed), eval_episodes)))

    if include_dqn:
        from dqn import DQNAgent

        dqn = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            obs_scale=env.observation_space.high,
            seed=seed,
        )
        dqn.train(env, num_episodes=dqn_episodes)
        rows.append(("DQN", *evaluate(env, dqn, eval_episodes)))
    return rows


def print_table(rows):
    print(f"{'agent':<12}{'run-win':>10}{'avg wins':>11}")
    print("-" * 33)
    for name, rate, wins in rows:
        print(f"{name:<12}{rate:>9.1%}{wins:>11.2f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark SAP-AI agents on the built-in env.")
    p.add_argument("--episodes", type=int, default=4000, help="tabular training episodes")
    p.add_argument("--eval-episodes", type=int, default=300, help="episodes per evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dqn", action="store_true", help="also train + benchmark the DQN (slow)")
    p.add_argument("--dqn-episodes", type=int, default=1500)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = run(args.episodes, args.eval_episodes, args.seed, args.dqn, args.dqn_episodes)
    print_table(rows)
    return rows


if __name__ == "__main__":
    main()
