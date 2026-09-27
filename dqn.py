"""Masked Double DQN with replay, stable updates and portable checkpoints.

PyTorch is an optional dependency, imported only by this module. All observation
sizes are inferred from the environment, including legacy ten-feature models.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import QLearningAgent, _reset_env, _transition


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.buffer = deque(maxlen=int(capacity))
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done, next_mask):
        # Environments frequently reuse arrays. Own every transition's memory.
        self.buffer.append((np.array(state, dtype=np.float32, copy=True), int(action),
                            float(reward), np.array(next_state, dtype=np.float32, copy=True),
                            bool(done), np.array(next_mask, dtype=bool, copy=True)))

    def sample(self, batch_size: int):
        return tuple(zip(*self._rng.sample(self.buffer, batch_size)))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self, obs_dim: int, n_actions: int, obs_scale=None, lr: float = 1e-3,
        gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995, buffer_size: int = 50_000,
        batch_size: int = 64, target_update: int = 20, hidden: int = 128,
        device: str | None = None, seed: int | None = None,
        double_dqn: bool = True, max_grad_norm: float = 10.0,
    ):
        for name, value in {"obs_dim": obs_dim, "n_actions": n_actions, "hidden": hidden,
                            "buffer_size": buffer_size, "batch_size": batch_size,
                            "target_update": target_update}.items():
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if batch_size > buffer_size:
            raise ValueError("batch_size must not exceed buffer_size")
        for name, value in {"gamma": gamma, "epsilon": epsilon, "epsilon_min": epsilon_min,
                            "epsilon_decay": epsilon_decay}.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
            setattr(self, name, float(value))
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("lr and max_grad_norm must be finite and positive")
        self.lr = float(lr)
        self.max_grad_norm = float(max_grad_norm)
        self.double_dqn = bool(double_dqn)
        self.actions = list(range(self.n_actions))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = None if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self.episodes_trained = 0
        self.learning_steps = 0
        scale = np.ones(self.obs_dim, dtype=np.float32) if obs_scale is None else np.array(obs_scale, dtype=np.float32, copy=True)
        if scale.shape != (self.obs_dim,) or not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("obs_scale must have obs_dim finite, nonnegative values")
        scale[scale == 0] = 1.0
        self.obs_scale = scale
        # Seed initialization without resetting unrelated NumPy/Torch RNG streams.
        with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext():
            if seed is not None:
                torch.random.default_generator.manual_seed(self.seed)
            self.q = QNetwork(self.obs_dim, self.n_actions, self.hidden)
        self.q.to(self.device)
        self.target = copy.deepcopy(self.q).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(self.buffer_size, seed=self.seed)

    def _observation(self, obs):
        value = np.asarray(obs, dtype=np.float32)
        if value.shape != (self.obs_dim,) or not np.isfinite(value).all():
            raise ValueError(f"observation must contain {self.obs_dim} finite values")
        return value

    def _tensor(self, obs):
        return torch.as_tensor(self._observation(obs) / self.obs_scale, device=self.device)

    def _mask(self, mask):
        result = np.ones(self.n_actions, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if result.shape != (self.n_actions,):
            raise ValueError("action mask length must match the action count")
        return result

    def choose_action(self, obs, greedy: bool = False, mask=None) -> int:
        legal_mask = self._mask(mask)
        legal = np.flatnonzero(legal_mask).tolist()
        if not legal:
            raise ValueError("Cannot choose an action: no legal actions")
        observation = self._observation(obs)
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(legal)
        with torch.no_grad():
            q_values = self.q(self._tensor(observation).unsqueeze(0)).squeeze(0)
            q_values = q_values.masked_fill(~torch.as_tensor(legal_mask, device=self.device), -torch.inf)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done, next_mask=None):
        """Record a transition; ``done`` denotes termination, not truncation."""
        if isinstance(action, (bool, np.bool_)) or int(action) != action or not 0 <= action < self.n_actions:
            raise ValueError("action must be an integer in the action space")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        self.buffer.push(self._observation(state), action, reward, self._observation(next_state),
                         done, self._mask(next_mask))

    def learn(self):
        """Run one masked Double DQN update with Huber loss and clipped gradients."""
        if len(self.buffer) < self.batch_size:
            return None
        state, action, reward, next_state, done, next_mask = self.buffer.sample(self.batch_size)
        state = torch.as_tensor(np.stack(state) / self.obs_scale, device=self.device)
        next_state = torch.as_tensor(np.stack(next_state) / self.obs_scale, device=self.device)
        action = torch.as_tensor(action, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32).unsqueeze(1)
        done = torch.as_tensor(done, device=self.device, dtype=torch.bool).unsqueeze(1)
        next_mask = torch.as_tensor(np.stack(next_mask), device=self.device, dtype=torch.bool)
        values = self.q(state).gather(1, action)
        with torch.no_grad():
            target_values = self.target(next_state)
            if self.double_dqn:
                selected = self.q(next_state).masked_fill(~next_mask, -torch.inf).argmax(1, keepdim=True)
                future = target_values.gather(1, selected)
            else:
                future = target_values.masked_fill(~next_mask, -torch.inf).max(1, keepdim=True).values
            # Multiplying -inf by zero produces NaN. Select zero before arithmetic.
            future = torch.where(~done & next_mask.any(1, keepdim=True), future, 0.0)
            targets = reward + self.gamma * future
        loss = F.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.learning_steps += 1
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def train(self, env, num_episodes: int = 500, max_steps: int = 300,
              log_every: int = 0, seed: int | None = None):
        """Train with both Gym APIs; target_update is measured in episodes."""
        if num_episodes < 0 or max_steps < 1 or log_every < 0:
            raise ValueError("episodes/log_every must be nonnegative and max_steps positive")
        reset_seed = seed if seed is not None else (self.seed if not self.episodes_trained else None)
        history = []
        for ep in range(1, num_episodes + 1):
            result = _reset_env(env, reset_seed if ep == 1 else None)
            obs, mask = QLearningAgent._unpack_reset(result), QLearningAgent._mask_from_reset(result)
            total = 0.0
            for _ in range(max_steps):
                action = self.choose_action(obs, mask=mask)
                nxt, reward, terminated, truncated, info = _transition(env.step(action))
                next_mask = info.get("action_mask") if isinstance(info, dict) else None
                self.remember(obs, action, reward, nxt, terminated, next_mask)
                self.learn()
                obs, mask = nxt, next_mask
                total += reward
                if terminated or truncated:
                    break
            self.decay_epsilon()
            self.episodes_trained += 1
            if self.episodes_trained % self.target_update == 0:
                self.update_target()
            history.append(total)
            if log_every and ep % log_every == 0:
                recent = sum(history[-log_every:]) / log_every
                print(f"  episode {ep:5d} | avg reward {recent:7.3f} | epsilon {self.epsilon:.3f}")
        return history

    def _config(self):
        return {name: getattr(self, name) for name in (
            "obs_dim", "n_actions", "lr", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
            "buffer_size", "batch_size", "target_update", "hidden", "seed", "double_dqn", "max_grad_norm",
        )}

    def save(self, path: str, include_replay: bool = True):
        """Atomically save a safely loadable checkpoint, including training state.

        Set include_replay=False for a smaller inference-only artifact. Environment
        state is not captured; resume training at an episode boundary with a seed.
        """
        data = {
            "format_version": 2, "config": self._config(),
            "model": self.q.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "obs_scale": self.obs_scale.tolist(),
            "obs_dim": self.obs_dim, "n_actions": self.n_actions,
            "episodes_trained": self.episodes_trained, "learning_steps": self.learning_steps,
            "rng_state": self._rng.getstate(), "replay_rng_state": self.buffer._rng.getstate(),
        }
        if include_replay and len(self.buffer):
            columns = list(zip(*self.buffer.buffer))
            data["replay"] = [torch.as_tensor(np.stack(column)) for column in columns]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
                temporary = file.name
            torch.save(data, temporary)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None):
        """Infer model dimensions/config without requiring unsafe pickle loading."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version", 1) not in (1, 2):
            raise ValueError("Unsupported DQN checkpoint version")
        if "config" in data:
            config = dict(data["config"])
        else:
            # Original checkpoints omitted hidden width and all training settings.
            config = {"obs_dim": int(data["obs_dim"]), "n_actions": int(data["n_actions"]),
                      "hidden": int(data["model"]["net.0.weight"].shape[0])}
        restored = cls(**config, obs_scale=data["obs_scale"], device=device)
        restored.q.load_state_dict(data["model"])
        restored.target.load_state_dict(data.get("target", data["model"]))
        if "optimizer" in data:
            restored.optimizer.load_state_dict(data["optimizer"])
        restored.episodes_trained = int(data.get("episodes_trained", 0))
        restored.learning_steps = int(data.get("learning_steps", 0))
        if "rng_state" in data:
            restored._rng.setstate(data["rng_state"])
        if "replay_rng_state" in data:
            restored.buffer._rng.setstate(data["replay_rng_state"])
        if "replay" in data:
            columns = [column.cpu().numpy() for column in data["replay"]]
            if len(columns) != 6 or len({len(column) for column in columns}) != 1:
                raise ValueError("Invalid checkpoint replay columns")
            for transition in zip(*columns):
                restored.remember(*transition)
        return restored

    def load(self, path: str):
        """Restore the saved architecture and training state into this agent."""
        restored = self.from_checkpoint(path, device=str(self.device))
        self.__dict__.update(restored.__dict__)
        return self


def main(argv=None):
    from game import SuperAutoPetsEnv
    from evaluation import evaluate_policy

    parser = argparse.ArgumentParser(description="Train a masked Double DQN on the built-in SAP arena.")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for the small network (default: 1)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="dqn.pt")
    parser.add_argument("--resume", metavar="PATH", help="resume a DQN checkpoint")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--plot", metavar="PATH")
    parser.add_argument("--report", metavar="PATH", help="save rewards and per-episode evaluation metrics as JSON")
    args = parser.parse_args(argv)
    if args.episodes < 0 or args.eval_episodes < 1 or args.max_steps < 1 or args.threads < 1:
        parser.error("episodes must be nonnegative; eval-episodes, max-steps and threads must be positive")
    if args.seed < 0 or args.eval_seed < 0:
        parser.error("seeds must be nonnegative")
    torch.set_num_threads(args.threads)
    env = SuperAutoPetsEnv()
    try:
        agent = (DQNAgent.from_checkpoint(args.resume, device=args.device) if args.resume else DQNAgent(
            obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n,
            obs_scale=env.observation_space.high, epsilon_decay=0.997, seed=args.seed, device=args.device))
        if agent.obs_dim != env.observation_space.shape[0] or agent.n_actions != env.action_space.n:
            parser.error("checkpoint dimensions do not match this environment; retrain for the current observation schema")
        baseline = evaluate_policy(env, episodes=args.eval_episodes,
                                   max_steps=args.max_steps, seed=args.eval_seed)
        print(f"Random baseline : run-win rate {baseline.win_rate:5.1%} | avg wins/run {baseline.avg_wins:4.2f}")
        history = agent.train(env, num_episodes=args.episodes, max_steps=args.max_steps,
                              log_every=max(1, args.episodes // 4), seed=args.seed)
        result = evaluate_policy(env, agent, episodes=args.eval_episodes,
                                 max_steps=args.max_steps, seed=args.eval_seed)
        print(f"DQN policy      : run-win rate {result.win_rate:5.1%} | avg wins/run {result.avg_wins:4.2f}")
        low, high = result.win_rate_ci95
        print(f"95% interval    : {low:.1%}-{high:.1%} | truncated {result.truncations}")
        if args.plot:
            from metrics import plot_training_curve
            Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
            plot_training_curve(history, args.plot, title="Double DQN training reward")
        if not args.no_save:
            agent.save(args.output)
            print(f"Saved model to {args.output}")
        if args.report:
            report = {"schema_version": 1, "environment": "simplified_arena_v2",
                      "config": vars(args), "training_rewards": history,
                      "episodes_trained": agent.episodes_trained,
                      "random": baseline.to_dict(), "dqn": result.to_dict()}
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved report to {args.report}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
