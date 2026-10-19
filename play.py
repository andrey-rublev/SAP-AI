"""Run heuristic, Q-table or DQN policies in the offline toy arena.

The real game is played by the desktop controller (``desktop.py`` / ``play.cmd``).
"""
from __future__ import annotations

import argparse

from agent import QLearningAgent
from game import SuperAutoPetsEnv
from heuristic import HeuristicAgent
from main import compact_state


def load_policy(args):
    if args.policy == "heuristic":
        return HeuristicAgent()
    if args.policy == "qtable":
        if not args.qtable:
            raise SystemExit("--policy qtable requires --qtable PATH")
        policy = QLearningAgent([], state_fn=compact_state, seed=args.seed).load(args.qtable)
        if policy.actions != list(range(6)):
            raise ValueError("Q-table must use the toy arena's six actions [0, 1, 2, 3, 4, 5]")
        return policy
    if not args.checkpoint:
        raise SystemExit("--policy dqn requires --checkpoint PATH")
    try:
        from dqn import DQNAgent
    except ImportError as exc:
        raise SystemExit("DQN playback requires PyTorch; install the project's DQN dependencies") from exc
    policy = DQNAgent.from_checkpoint(args.checkpoint, device=args.device)
    if policy.n_actions != 6:
        raise ValueError("DQN checkpoint must use the toy arena's six actions")
    return policy


def _policy_observation(policy, obs):
    """Support old ten-value DQN checkpoints without fabricating metadata."""
    dimension = getattr(policy, "obs_dim", len(obs))
    if dimension == len(obs):
        return obs
    if dimension == 10 and len(obs) > 10:
        return obs[:10]
    raise ValueError(f"policy expects {dimension} observation values, but this mode provides {len(obs)}")


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def run_dry(policy, episodes: int, max_steps: int, seed: int = 0, verbose: bool = True):
    """Play seeded simulator episodes; report external cutoffs explicitly."""
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    env = SuperAutoPetsEnv()
    results = []
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            total_reward = 0.0
            terminated = truncated = False
            for t in range(max_steps):
                action = policy.choose_action(_policy_observation(policy, obs), greedy=True, mask=info.get("action_mask"))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if verbose:
                    print(f"ep {ep} step {t:3d} | action {action} | reward {reward:+.2f} | wins {info['wins']} lives {info['lives']}")
                if terminated or truncated:
                    break
            cutoff = not (terminated or truncated)
            outcome = "won" if info["wins"] >= env.WINS_TO_WIN else "lost" if terminated else "truncated"
            info = {**info, "terminated": bool(terminated), "truncated": bool(truncated or cutoff),
                    "cutoff": cutoff, "steps": t + 1, "total_reward": total_reward, "outcome": outcome}
            if verbose:
                print(f"ep {ep} finished [{outcome}]: wins={info['wins']} lives={info['lives']}\n")
            results.append(info)
    finally:
        env.close()
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a policy in the offline SAP-inspired toy arena.")
    p.add_argument("--dry-run", action="store_true", help="accepted for compatibility; the simulator is the only mode")
    p.add_argument("--policy", choices=["heuristic", "qtable", "dqn"], default="heuristic")
    p.add_argument("--qtable", help="saved Q-table for --policy qtable")
    p.add_argument("--checkpoint", help="saved PyTorch model for --policy dqn")
    p.add_argument("--device", default="cpu", help="DQN device (default: cpu)")
    p.add_argument("--episodes", type=_positive_int, default=1)
    p.add_argument("--max-steps", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="omit per-step simulator output")
    args = p.parse_args(argv)
    if args.seed < 0:
        p.error("--seed must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    policy = load_policy(args)
    return run_dry(policy, args.episodes, args.max_steps, seed=args.seed, verbose=not args.quiet)


def cli():
    """Console entry point; result lists must not become an error exit status."""
    main()


if __name__ == "__main__":
    cli()
