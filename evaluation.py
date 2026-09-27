"""Seeded policy evaluation with legal random actions and episode diagnostics."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import sqrt
import random

import numpy as np


def positive_int(value):
    """argparse type for counts that cannot be zero or negative."""
    import argparse

    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    reward: float
    wins: int
    lives: int
    turns: int
    steps: int
    victory: bool
    terminated: bool
    truncated: bool
    stop_reason: str


@dataclass(frozen=True)
class EvaluationResult:
    episodes: tuple[EpisodeResult, ...]

    @property
    def win_rate(self):
        return sum(ep.victory for ep in self.episodes) / len(self.episodes)

    @property
    def avg_wins(self):
        return sum(ep.wins for ep in self.episodes) / len(self.episodes)

    @property
    def avg_reward(self):
        return sum(ep.reward for ep in self.episodes) / len(self.episodes)

    @property
    def avg_steps(self):
        return sum(ep.steps for ep in self.episodes) / len(self.episodes)

    @property
    def truncations(self):
        return sum(ep.truncated for ep in self.episodes)

    @property
    def win_rate_ci95(self):
        """Wilson interval, including sensible bounds for all wins/all losses."""
        n, p, z = len(self.episodes), self.win_rate, 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)

    def to_dict(self, *, include_episodes=True):
        result = {
            "episode_count": len(self.episodes),
            "win_rate": self.win_rate,
            "win_rate_ci95": list(self.win_rate_ci95),
            "avg_wins": self.avg_wins,
            "avg_reward": self.avg_reward,
            "avg_steps": self.avg_steps,
            "truncations": self.truncations,
        }
        if include_episodes:
            result["episodes"] = [asdict(ep) for ep in self.episodes]
        return result


@contextmanager
def _evaluation_rng(agent, seed):
    """Repeat tie breaks without advancing a trained policy's exploration RNG."""
    rng = getattr(agent, "_rng", None)
    if not isinstance(rng, random.Random):
        yield
        return
    state = rng.getstate()
    try:
        rng.seed(seed)
        yield
    finally:
        rng.setstate(state)


def evaluate_policy(env, agent=None, episodes=200, max_steps=500, seed=10_000):
    """Evaluate on ``seed .. seed + episodes - 1`` without training the policy.

    The random baseline samples only legal actions, just like learned policies.
    A step budget expiry is recorded as a truncation, never as a finished loss.
    Use a dedicated evaluation environment: its state is reset here.
    """
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    outcomes = []
    with _evaluation_rng(agent, seed):
        for ep in range(episodes):
            episode_seed = seed + ep
            rng = np.random.default_rng(episode_seed)
            obs, info = env.reset(seed=episode_seed)
            total = 0.0
            terminated = truncated = False
            for step in range(1, max_steps + 1):
                mask = info.get("action_mask")
                if agent is None:
                    legal = np.arange(env.action_space.n) if mask is None else np.flatnonzero(mask)
                    if not len(legal):
                        raise ValueError("nonterminal observation has no legal actions")
                    action = int(rng.choice(legal))
                else:
                    action = agent.choose_action(obs, greedy=True, mask=mask)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            if terminated:
                reason = "victory" if info["wins"] >= env.WINS_TO_WIN else "defeat"
            elif truncated:
                reason = info.get("truncation_reason") or info.get("episode_reason") or "environment_limit"
            else:
                truncated = True
                reason = "evaluation_step_limit"
            outcomes.append(EpisodeResult(
                seed=episode_seed, reward=total, wins=int(info["wins"]),
                lives=int(info["lives"]), turns=int(info["turn"]), steps=step,
                victory=bool(terminated and info["wins"] >= env.WINS_TO_WIN),
                terminated=bool(terminated), truncated=bool(truncated), stop_reason=reason,
            ))
    return EvaluationResult(tuple(outcomes))
