"""Tabular Q-learning agent.

A small, dependency-light Q-learning implementation shared by ``train.py`` (which
trains against the real ``sapai`` battle simulator) and ``main.py`` (which trains
against the built-in :class:`game.SuperAutoPetsEnv`).

It is deliberately environment-agnostic:

* States may be strings, ints, tuples or numpy arrays.  An optional ``state_fn``
  can compress a raw observation into a compact, learnable key.
* The :meth:`train` loop accepts either the classic 4-tuple
  ``(obs, reward, done, info)`` or the Gymnasium 5-tuple
  ``(obs, reward, terminated, truncated, info)`` step API.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        actions,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        state_fn=None,
        seed: int | None = None,
    ):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_fn = state_fn
        self._rng = random.Random(seed)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    # ------------------------------------------------------------- helpers
    def _key(self, state):
        """Turn any state into a hashable, table-friendly key."""
        if self.state_fn is not None:
            state = self.state_fn(state)
        if isinstance(state, (str, int, float, bool, bytes)):
            return state
        try:  # numpy arrays / lists / tuples of numbers
            return tuple(int(round(float(x))) for x in state)
        except TypeError:
            return str(state)

    # -------------------------------------------------------------- policy
    def choose_action(self, state, greedy: bool = False):
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        q_row = self.q[self._key(state)]
        best = max(q_row.values())
        # Random tie-break so the agent doesn't fixate on the first-listed action.
        best_actions = [a for a, v in q_row.items() if v == best]
        return self._rng.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        k, nk = self._key(state), self._key(next_state)
        current = self.q[k][action]
        future = 0.0 if done else max(self.q[nk].values())
        self.q[k][action] = current + self.alpha * (
            reward + self.gamma * future - current
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------ training
    @staticmethod
    def _unpack_reset(result):
        # Gymnasium returns (obs, info); classic envs return just obs.
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0]
        return result

    @staticmethod
    def _unpack_step(result):
        if len(result) == 5:  # gymnasium
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated or truncated, info
        obs, reward, done, info = result  # classic gym
        return obs, reward, done, info

    def train(self, env, num_episodes: int = 1000, max_steps: int = 200, log_every: int = 0):
        """Run Q-learning and return the per-episode total reward history."""
        history = []
        for ep in range(1, num_episodes + 1):
            state = self._unpack_reset(env.reset())
            total = 0.0
            for _ in range(max_steps):
                action = self.choose_action(state)
                nxt, reward, done, _ = self._unpack_step(env.step(action))
                self.learn(state, action, reward, nxt, done)
                state = nxt
                total += reward
                if done:
                    break
            self.decay_epsilon()
            history.append(total)
            if log_every and ep % log_every == 0:
                recent = sum(history[-log_every:]) / log_every
                print(
                    f"  episode {ep:5d} | avg reward {recent:7.3f} "
                    f"| epsilon {self.epsilon:.3f}"
                )
        return history

    # --------------------------------------------------------- persistence
    def save(self, path: str):
        data = {
            "actions": self.actions,
            "params": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "q": {json.dumps(k): v for k, v in self.q.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.actions = data["actions"]
        for name, value in data.get("params", {}).items():
            setattr(self, name, value)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        for encoded, row in data["q"].items():
            decoded = json.loads(encoded)
            key = tuple(decoded) if isinstance(decoded, list) else decoded
            self.q[key] = row
        return self
