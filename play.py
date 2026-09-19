"""Run a policy through a Super Auto Pets game loop.

Two modes:

* ``--dry-run`` drives the built-in simulator (:class:`game.SuperAutoPetsEnv`) and
  prints the action stream.  Fully offline - used to exercise/verify the control
  loop without the game.
* live (default) drives the real Steam client via :mod:`autogui`.  Calibrate
  ``autogui.LAYOUT`` for your resolution first.  NOTE: full board perception
  (reading every shop/team pet from the screen) is not implemented yet, so live
  play currently sees only gold/hearts - it will act, but not well, until the
  perception layer lands.

Policies::

    python play.py --dry-run                       # rule-based heuristic
    python play.py --dry-run --policy qtable --qtable qtable.json
    python play.py                                 # live, needs the game focused
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from agent import QLearningAgent
from game import SuperAutoPetsEnv
from heuristic import HeuristicAgent
from main import compact_state


def load_policy(args):
    if args.policy == "heuristic":
        return HeuristicAgent()
    if not args.qtable:
        raise SystemExit("--policy qtable requires --qtable PATH")
    return QLearningAgent([], state_fn=compact_state).load(args.qtable)


def run_dry(policy, episodes: int, max_steps: int, seed: int = 0, verbose: bool = True):
    """Play whole episodes against the simulator; return each episode's final info."""
    env = SuperAutoPetsEnv()
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        for t in range(max_steps):
            action = policy.choose_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if verbose:
                print(
                    f"ep {ep} step {t:3d} | action {action} | reward {reward:+.2f} "
                    f"| wins {info['wins']} lives {info['lives']}"
                )
            if terminated or truncated:
                break
        outcome = "WON" if info["wins"] >= env.WINS_TO_WIN else "out"
        if verbose:
            print(f"ep {ep} finished [{outcome}]: wins={info['wins']} lives={info['lives']}\n")
        results.append(info)
    return results


def run_live(policy, max_steps: int, delay: float):
    import autogui  # imported lazily; needs pyautogui/pytesseract

    print("Live mode - focus Super Auto Pets. Slam the mouse into a screen corner to abort.")
    print("WARNING: board perception is a stub; the bot only reads gold/hearts for now.")
    for t in range(max_steps):
        state = autogui.read_state()
        # Best-effort observation: gold is real, the rest is unknown (zeros).
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = state.get("gold", 0)
        action = policy.choose_action(obs, greedy=True)
        autogui.perform_action(action, obs=obs)
        time.sleep(delay)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a policy through a SAP game loop.")
    p.add_argument("--dry-run", action="store_true", help="use the simulator instead of the live client")
    p.add_argument("--policy", choices=["heuristic", "qtable"], default="heuristic")
    p.add_argument("--qtable", help="path to a saved Q-table (for --policy qtable)")
    p.add_argument("--episodes", type=int, default=1, help="dry-run episodes (default: 1)")
    p.add_argument("--max-steps", type=int, default=300, help="max steps per episode/session")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.4, help="seconds between live actions")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    policy = load_policy(args)
    if args.dry_run:
        return run_dry(policy, args.episodes, args.max_steps, seed=args.seed)
    run_live(policy, args.max_steps, args.delay)


if __name__ == "__main__":
    main()
