"""Deep Q-Network agent for the full Super Auto Pets observation.

Tabular Q-learning (see :mod:`agent`) has to compress the observation into a tiny
key to stay learnable.  This DQN instead learns directly from the raw 10-dim
observation with a small MLP, experience replay and a target network.

``choose_action`` matches the other agents' signature, so a trained DQN drops
straight into :func:`main.evaluate` and :mod:`play`.

Requires PyTorch (``pip install torch``); it is only imported here, so the rest
of the project stays torch-free.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        self.buffer = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = self._rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        obs_scale=None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update: int = 20,
        hidden: int = 128,
        device: str | None = None,
        seed: int | None = None,
    ):
        # Coerce to native ints: gymnasium's action_space.n is a numpy scalar,
        # which would otherwise be pickled into checkpoints and rejected by
        # torch.load's safe weights_only default.
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.actions = list(range(self.n_actions))
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._rng = random.Random(seed)

        # Input normalisation keeps the different observation scales comparable.
        scale = np.ones(obs_dim, dtype=np.float32) if obs_scale is None else np.asarray(obs_scale, dtype=np.float32)
        scale[scale == 0] = 1.0
        self.obs_scale = scale

        self.q = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, seed=seed)

    # ---------------------------------------------------------------- policy
    def _tensor(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32) / self.obs_scale, device=self.device)

    def choose_action(self, obs, greedy: bool = False, mask=None) -> int:
        legal = list(range(self.n_actions)) if mask is None else [i for i in range(self.n_actions) if mask[i]]
        if not legal:
            legal = list(range(self.n_actions))
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(legal)
        with torch.no_grad():
            q_values = self.q(self._tensor(obs).unsqueeze(0)).squeeze(0)
        if mask is not None:
            legal_mask = torch.as_tensor(np.asarray(mask, dtype=bool), device=q_values.device)
            q_values = torch.where(legal_mask, q_values, torch.full_like(q_values, float("-inf")))
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(
            np.asarray(state, dtype=np.float32), action, reward,
            np.asarray(next_state, dtype=np.float32), done,
        )

    def learn(self):
        """One gradient step on a replay minibatch. Returns the loss, or None."""
        if len(self.buffer) < self.batch_size:
            return None
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.as_tensor(np.array(state) / self.obs_scale, device=self.device, dtype=torch.float32)
        next_state = torch.as_tensor(np.array(next_state) / self.obs_scale, device=self.device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32).unsqueeze(1)
        done = torch.as_tensor(done, device=self.device, dtype=torch.float32).unsqueeze(1)

        q = self.q(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            target = reward + self.gamma * next_q * (1.0 - done)
        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    # -------------------------------------------------------------- training
    def train(self, env, num_episodes: int = 500, max_steps: int = 300, log_every: int = 0):
        history = []
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            mask = info.get("action_mask")
            total = 0.0
            for _ in range(max_steps):
                action = self.choose_action(obs, mask=mask)
                nxt, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.remember(obs, action, reward, nxt, done)
                self.learn()
                obs = nxt
                mask = info.get("action_mask")
                total += reward
                if done:
                    break
            self.decay_epsilon()
            if ep % self.target_update == 0:
                self.update_target()
            history.append(total)
            if log_every and ep % log_every == 0:
                recent = sum(history[-log_every:]) / log_every
                print(f"  episode {ep:5d} | avg reward {recent:7.3f} | epsilon {self.epsilon:.3f}")
        return history

    # ------------------------------------------------------------- persistence
    def save(self, path: str):
        torch.save(
            {
                "model": self.q.state_dict(),
                # a plain list (not a numpy array) so torch.load's safe
                # weights_only=True default can restore the checkpoint
                "obs_scale": self.obs_scale.tolist(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["model"])
        self.target.load_state_dict(ckpt["model"])
        self.obs_scale = np.asarray(ckpt["obs_scale"], dtype=np.float32)
        return self


if __name__ == "__main__":
    from game import SuperAutoPetsEnv
    from main import evaluate

    env = SuperAutoPetsEnv()
    base_rate, base_wins = evaluate(env, agent=None, episodes=200)
    print(f"Random baseline : run-win rate {base_rate:5.1%} | avg wins/run {base_wins:4.2f}")

    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        obs_scale=env.observation_space.high,
        epsilon_decay=0.997,
        seed=0,
    )
    print("\nTraining DQN ...")
    agent.train(env, num_episodes=1500, log_every=250)

    rate, wins = evaluate(env, agent=agent, episodes=200)
    print(f"\nDQN policy      : run-win rate {rate:5.1%} | avg wins/run {wins:4.2f}")
    agent.save("dqn.pt")
    print("Saved model to dqn.pt")
