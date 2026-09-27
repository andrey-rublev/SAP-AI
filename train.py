"""Optional sapai battle-engine drill with a small custom shop.

This is not a complete SAP rules environment: shopping uses fixed costs and
three low-tier pet choices; only combat is delegated to sapai. Install sapai
from its upstream repository (see README). Importing this module needs no sapai.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from agent import QLearningAgent
from evaluation import positive_int

ACTIONS = ["buy_0", "buy_1", "buy_2", "roll", "battle"]
PETS = ("ant", "fish", "otter", "beaver", "pig")


def _load_engine():
    try:
        from sapai.teams import Team
        from sapai.battle import Battle
    except ImportError as exc:
        raise RuntimeError(
            "The optional sapai battle drill needs the upstream sapai package. "
            "See README for installation, or run python main.py for the built-in arena."
        ) from exc
    return Team, Battle


@contextmanager
def _battle_rng(seed):
    """Isolate sapai's legacy global NumPy randomness in this serial drill."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


class SAPEnvironment:
    """Bounded team-building drill using real sapai combat and a custom shop."""

    def __init__(self, seed=None, max_steps=50, engine=None):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._Team, self._Battle = engine or _load_engine()
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._done = True

    def _roll_shop(self):
        self.shop = self._rng.integers(1, len(PETS) + 1, size=3).tolist()

    def _obs(self):
        return (self.gold, *self.shop, *self.pets, *([0] * (5 - len(self.pets))))

    def _info(self):
        mask = [self.gold >= 3 and len(self.pets) < 5 and pet > 0 for pet in self.shop]
        mask += [self.gold >= 1 and len(self.pets) < 5, True]
        if self._done and not self._truncated:
            mask = [False] * len(ACTIONS)
        return {"action_mask": np.array(mask, dtype=bool), "gold": self.gold,
                "steps": self.steps, "battle_winner": self.winner}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.gold, self.steps, self.winner = 15, 0, None
        self.pets = []
        self.team = self._Team([])
        self.enemy_team = self._Team(["sheep", "tiger"])
        self._roll_shop()
        self._done = self._truncated = False
        return self._obs(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("call reset() before stepping a finished drill")
        if action not in ACTIONS:
            raise ValueError(f"invalid action {action!r}")
        legal = bool(self._info()["action_mask"][ACTIONS.index(action)])
        self.steps += 1
        reward, terminated = -0.01, False
        if not legal:
            reward = -0.1
        elif action.startswith("buy_"):
            slot = int(action[-1])
            self.pets.append(self.shop[slot])
            self.shop[slot] = 0
            self.gold -= 3
            # Team.move takes two slot indices; it cannot insert a new Pet.
            self.team = self._Team([PETS[pet - 1] for pet in self.pets])
        elif action == "roll":
            self.gold -= 1
            self._roll_shop()
        else:
            with _battle_rng(int(self._rng.integers(0, 2**32))):
                self.winner = int(self._Battle(self.team, self.enemy_team).battle())
            if self.winner not in (0, 1, 2):
                raise RuntimeError(f"unexpected sapai winner code: {self.winner}")
            reward = {0: 1.0, 1: -1.0, 2: 0.0}[self.winner]
            terminated = True
        self._truncated = not terminated and self.steps >= self.max_steps
        self._done = terminated or self._truncated
        info = self._info()
        if self._truncated:
            info["truncation_reason"] = "step_limit"
        return self._obs(), reward, terminated, self._truncated, info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=positive_int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=positive_int, default=50)
    p.add_argument("--output", default="qtable_sapai.json")
    args = p.parse_args(argv)
    if args.seed < 0:
        p.error("--seed must be non-negative")
    try:
        env = SAPEnvironment(seed=args.seed, max_steps=args.max_steps)
    except RuntimeError as exc:
        p.error(str(exc))
    agent = QLearningAgent(ACTIONS, seed=args.seed)
    print(f"Training {args.episodes} episodes on the custom-shop sapai battle drill ...")
    agent.train(env, num_episodes=args.episodes, max_steps=args.max_steps,
                seed=args.seed, log_every=max(1, args.episodes // 10))
    print("\nGreedy rollout:")
    state, info = env.reset(seed=args.seed + 10_000)
    for _ in range(args.max_steps):
        action = agent.choose_action(state, greedy=True, mask=info["action_mask"])
        state, reward, terminated, truncated, info = env.step(action)
        print(f"  action={action:<8s} reward={reward:+.2f}")
        if terminated or truncated:
            break
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.output)
    print(f"Saved battle-drill Q-table to {args.output}")
    return agent


if __name__ == "__main__":
    main()
