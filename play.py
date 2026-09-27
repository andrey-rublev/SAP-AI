"""Run heuristic, Q-table or DQN policies in the offline toy arena by default.

--live --layout calibrated.json explicitly enables the experimental OCR bridge.
It does not reproduce real SAP rules or identify pet species. A live session
stops after ending one turn because battle/loading detection is absent.
--preview reads the live board and prints one proposal without clicking.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

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


def _nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
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


def run_live(policy, max_steps: int, delay: float, *, layout_path, preview=False, turn=0):
    """Run only the current shop phase; never continue blindly into a battle."""
    if max_steps <= 0 or not math.isfinite(delay) or delay < 0 or turn < 0:
        raise ValueError("max_steps must be positive; delay and turn must be nonnegative")
    if getattr(policy, "obs_dim", 10) != 10:
        raise ValueError("live OCR provides only 10 legacy scalars; this DQN requires simulator metadata unavailable on screen")
    import autogui

    autogui.configure_layout(layout_path, enable_control=not preview)
    print("Experimental live bridge: scalar OCR cannot identify pet species or reproduce real SAP rules.")
    print("Only empty-slot purchases are supported. Unreadable OCR stops the session.")
    print("Keep the calibrated shop screen focused. Move the mouse to a screen corner to abort.")
    actions = []
    try:
        for _ in range(max_steps):
            obs = autogui.read_board(turn)
            mask = autogui.live_action_mask(obs)
            action = policy.choose_action(_policy_observation(policy, obs), greedy=True, mask=mask)
            if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or action not in range(6) or not mask[action]:
                raise ValueError(f"policy proposed unsupported live action {action}")
            print(f"{'Preview' if preview else 'Live'} action {action} | board {obs.tolist()}")
            actions.append(int(action))
            if preview:
                break
            autogui.perform_action(action, obs=obs)
            if action == 5:
                print("Turn ended. Session stopped; resume manually after the next shop is visible.")
                break
            time.sleep(delay)
    finally:
        autogui.disable_control()
    return actions


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a policy in the offline SAP-inspired toy arena (default).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="explicitly enable experimental screen/OCR control")
    mode.add_argument("--dry-run", action="store_true", help="use the simulator (already the default)")
    p.add_argument("--layout", help="validated calibration JSON required for --live")
    p.add_argument("--preview", action="store_true", help="with --live, read one board and print an action without clicking")
    p.add_argument("--turn", type=int, default=0, help="current turn index for live OCR (default: 0)")
    p.add_argument("--policy", choices=["heuristic", "qtable", "dqn"], default="heuristic")
    p.add_argument("--qtable", help="saved Q-table for --policy qtable")
    p.add_argument("--checkpoint", help="saved PyTorch model for --policy dqn")
    p.add_argument("--device", default="cpu", help="DQN device (default: cpu)")
    p.add_argument("--episodes", type=_positive_int, default=1)
    p.add_argument("--max-steps", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=_nonnegative_float, default=0.4, help="seconds between live actions")
    p.add_argument("--quiet", action="store_true", help="omit per-step simulator output")
    args = p.parse_args(argv)
    if args.live and not args.layout:
        p.error("--live requires a calibrated --layout PATH")
    if args.preview and not args.live:
        p.error("--preview requires --live")
    if args.turn < 0:
        p.error("--turn must be nonnegative")
    if args.seed < 0:
        p.error("--seed must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    policy = load_policy(args)
    if not args.live:
        return run_dry(policy, args.episodes, args.max_steps, seed=args.seed, verbose=not args.quiet)
    return run_live(policy, args.max_steps, args.delay, layout_path=args.layout, preview=args.preview, turn=args.turn)


def cli():
    """Console entry point; result lists must not become an error exit status."""
    main()


if __name__ == "__main__":
    cli()
