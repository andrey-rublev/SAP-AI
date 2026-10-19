"""Environment-agnostic, masked tabular Q-learning with resumable checkpoints."""
from __future__ import annotations

import inspect
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path


def _reset_env(env, seed=None):
    """Seed a Gymnasium environment without breaking classic reset() APIs."""
    if seed is None:
        return env.reset()
    try:
        inspect.signature(env.reset).bind(seed=seed)
    except (TypeError, ValueError):
        if callable(getattr(env, "seed", None)):
            env.seed(seed)
        return env.reset()
    return env.reset(seed=seed)


def _transition(result):
    """Keep time limits distinct from absorbing terminal states in both APIs."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        truncated = bool(done and isinstance(info, dict) and info.get("TimeLimit.truncated"))
        terminated = bool(done and not truncated)
    else:
        raise ValueError("env.step must return a 4- or 5-element transition")
    return obs, reward, bool(terminated), bool(truncated), info


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value


def _encode_key(value):
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_encode_key(x) for x in value]}
    return value


def _decode_key(value):
    if isinstance(value, dict):
        if set(value) == {"bytes"}:
            return bytes.fromhex(value["bytes"])
        if set(value) == {"tuple"}:
            return tuple(_decode_key(x) for x in value["tuple"])
        raise ValueError("Unsupported checkpoint key")
    return _tuple_tree(value)


class QLearningAgent:
    def __init__(
        self, actions, alpha: float = 0.1, gamma: float = 0.95,
        epsilon: float = 1.0, epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995, state_fn=None, seed: int | None = None,
    ):
        self.actions = list(actions)
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        for name, value in {"alpha": alpha, "gamma": gamma, "epsilon": epsilon,
                            "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay}.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
            setattr(self, name, float(value))
        self.state_fn = state_fn
        self.seed = None if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self.episodes_trained = 0
        self.q = defaultdict(lambda: dict.fromkeys(self.actions, 0.0))

    def _key(self, state):
        """Round numerical observations; leave explicit hashable keys intact."""
        if self.state_fn is not None:
            state = self.state_fn(state)
        if isinstance(state, (str, int, float, bool, bytes)) or state is None:
            return state
        try:
            return tuple(int(round(float(x))) for x in state)
        except (TypeError, ValueError):
            try:
                hash(state)
                return state
            except TypeError:
                return str(state)

    def _legal_actions(self, mask):
        if mask is not None and len(mask) != len(self.actions):
            raise ValueError("action mask length must match the action count")
        return [a for i, a in enumerate(self.actions) if mask is None or mask[i]]

    def choose_action(self, state, greedy: bool = False, mask=None):
        """Select a legal action, breaking equal values with the local RNG.

        Policy lookups do not insert unseen states into the learned table.
        An empty mask is a caller error: terminal states should not be acted on.
        """
        legal = self._legal_actions(mask)
        if not legal:
            raise ValueError("Cannot choose an action: no legal actions")
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(legal)
        row = self.q.get(self._key(state), {})
        best = max(row.get(a, 0.0) for a in legal)
        return self._rng.choice([a for a in legal if row.get(a, 0.0) == best])

    def learn(self, state, action, reward, next_state, done, next_mask=None):
        """Update Q; ``done`` means terminated, not a time-limit truncation."""
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action!r}")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        key = self._key(state)
        current = self.q[key][action]
        future = 0.0
        if not done:
            legal = self._legal_actions(next_mask)
            row = self.q.get(self._key(next_state), {})
            future = max((row.get(a, 0.0) for a in legal), default=0.0)
        self.q[key][action] = current + self.alpha * (reward + self.gamma * future - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @staticmethod
    def _unpack_reset(result):
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0]
        return result

    @staticmethod
    def _mask_from_reset(result):
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[1].get("action_mask")
        return None

    @staticmethod
    def _unpack_step(result):
        obs, reward, terminated, truncated, info = _transition(result)
        return obs, reward, terminated or truncated, info

    def train(self, env, num_episodes: int = 1000, max_steps: int = 200,
              log_every: int = 0, seed: int | None = None):
        """Train with legal targets; optionally seed the first environment reset.

        Later episodes continue the environment's RNG stream. The constructor
        seed is used on the first training call when no reset seed is supplied.
        """
        if num_episodes < 0 or max_steps < 1 or log_every < 0:
            raise ValueError("episodes/log_every must be nonnegative and max_steps positive")
        reset_seed = seed if seed is not None else (self.seed if not self.episodes_trained else None)
        history = []
        for ep in range(1, num_episodes + 1):
            reset_out = _reset_env(env, reset_seed if ep == 1 else None)
            state = self._unpack_reset(reset_out)
            mask = self._mask_from_reset(reset_out)
            total = 0.0
            for _ in range(max_steps):
                action = self.choose_action(state, mask=mask)
                nxt, reward, terminated, truncated, info = _transition(env.step(action))
                next_mask = info.get("action_mask") if isinstance(info, dict) else None
                self.learn(state, action, reward, nxt, terminated, next_mask)
                state, mask = nxt, next_mask
                total += reward
                if terminated or truncated:
                    break
            self.decay_epsilon()
            self.episodes_trained += 1
            history.append(total)
            if log_every and ep % log_every == 0:
                recent = sum(history[-log_every:]) / log_every
                print(f"  episode {ep:5d} | avg reward {recent:7.3f} | epsilon {self.epsilon:.3f}")
        return history

    def save(self, path: str):
        """Atomically save values, hyperparameters, encoder identity and RNG state."""
        data = {
            "format_version": 2,
            "actions": [_encode_key(a) for a in self.actions],
            "params": {name: getattr(self, name) for name in
                       ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay")},
            "state_encoder": getattr(self.state_fn, "__name__", None),
            "state_encoder_version": getattr(self.state_fn, "version", None),
            "seed": self.seed, "episodes_trained": self.episodes_trained,
            "rng_state": self._rng.getstate(),
            "q": {json.dumps(_encode_key(k)): [row[a] for a in self.actions]
                  for k, row in self.q.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as file:
                temporary = file.name
                json.dump(data, file, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, path: str):
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        if data.get("format_version", 1) not in (1, 2):
            raise ValueError("Unsupported Q-table checkpoint version")
        encoder = (getattr(self.state_fn, "__name__", None), getattr(self.state_fn, "version", None))
        saved_encoder = (data.get("state_encoder"), data.get("state_encoder_version"))
        if ("state_encoder" in data and encoder != saved_encoder) or (
            "state_encoder" not in data and encoder[1] is not None
        ):
            raise ValueError("Checkpoint state encoder does not match; supply its original encoder or retrain the policy")
        restored = QLearningAgent([_decode_key(a) for a in data["actions"]],
                                  state_fn=self.state_fn, seed=data.get("seed"), **data.get("params", {}))
        for encoded, values in data["q"].items():
            if len(values) != len(restored.actions) or not all(math.isfinite(v) for v in values):
                raise ValueError("Checkpoint Q rows must contain one finite value per action")
            key = _decode_key(json.loads(encoded))
            restored.q[key] = dict(zip(restored.actions, values))
        restored.episodes_trained = int(data.get("episodes_trained", 0))
        if "rng_state" in data:
            restored._rng.setstate(_tuple_tree(data["rng_state"]))
        self.__dict__.update(restored.__dict__)
        self.q.default_factory = lambda: dict.fromkeys(self.actions, 0.0)
        return self
