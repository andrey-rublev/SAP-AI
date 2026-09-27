"""A small, deterministic-seed toy arena inspired by Super Auto Pets.

This is a scalar-strength shop/battle simulator, not a model of the real game's
pet abilities, positioning, combat, or economy. It is useful for exercising RL
algorithms locally; performance here does not establish real-client skill.

The float32 observation is ``[gold, turn, shop(3), team(5), wins, lives,
shop_actions]``. The original first ten values retain their meaning. Including
run progress and the shop-action counter makes episode boundaries observable.

Actions 0..2 buy a shop slot, 3 rolls, 4 sells the weakest pet, and 5 fights.
In-range but illegal actions are penalized no-ops. Runs end at ten wins or zero
lives, and truncate at the turn or per-turn shop-action limit.

Shop reward is 0.1 times the actual change in team strength, minus the action
cost (and an extra 0.02 for rolling). Thus buying then selling cannot farm
positive reward. Battle rewards are +1/0/-1, with +5 for victory or -2 for defeat.
The strength shaping is a practical training signal, not a policy-invariance
guarantee under discounted returns.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SuperAutoPetsEnv(gym.Env):
    """A bounded, dependency-light toy environment using the Gymnasium API."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    SHOP_SLOTS = 3
    TEAM_SLOTS = 5
    MAX_TIER = 6
    MAX_PET_STRENGTH = 50
    MAX_GOLD = 10  # turn allowance; selling may raise the balance above this
    START_LIVES = 5
    WINS_TO_WIN = 10
    MAX_TURNS = 30
    MAX_SHOP_ACTIONS = 40

    BUY_COST = 3
    ROLL_COST = 1
    SELL_VALUE = 1
    STEP_PENALTY = 0.05
    STRENGTH_REWARD = 0.1
    ILLEGAL_ACTION_PENALTY = -0.1
    OBSERVATION_VERSION = 2

    def __init__(self, render_mode: str | None = None, *, max_shop_actions: int = MAX_SHOP_ACTIONS):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render mode: {render_mode!r}")
        if (isinstance(max_shop_actions, (bool, np.bool_))
                or not isinstance(max_shop_actions, (int, np.integer))
                or max_shop_actions < 1):
            raise ValueError("max_shop_actions must be a positive integer")
        self.render_mode = render_mode
        self.max_shop_actions = int(max_shop_actions)
        self.action_space = spaces.Discrete(6)
        high = np.array(
            [self.MAX_GOLD + self.TEAM_SLOTS * self.SELL_VALUE, self.MAX_TURNS]
            + [self.MAX_TIER] * self.SHOP_SLOTS
            + [self.MAX_PET_STRENGTH] * self.TEAM_SLOTS
            + [self.WINS_TO_WIN, self.START_LIVES, self.max_shop_actions],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        self.gold = self.turn = self.wins = self.lives = self.shop_actions = 0
        self.shop_pets = np.zeros(self.SHOP_SLOTS, dtype=np.int32)
        self.team_pets = np.zeros(self.TEAM_SLOTS, dtype=np.int32)
        self._needs_reset = True
        self._episode_steps = 0
        self._episode_return = 0.0
        self._last_battle = None

    def _roll_shop(self) -> None:
        self.shop_pets = self.np_random.integers(
            1, self.MAX_TIER + 1, size=self.SHOP_SLOTS, dtype=np.int32
        )

    def _obs(self) -> np.ndarray:
        return np.concatenate((
            [self.gold, self.turn], self.shop_pets, self.team_pets,
            [self.wins, self.lives, self.shop_actions],
        )).astype(np.float32)

    @classmethod
    def mask_from_obs(cls, obs) -> np.ndarray:
        """Compute legal actions from a legacy 10-value or extended observation.

        A 12-value board with wins/lives is also supported for external readers.
        Metadata never participates in team matching. Truncated observations
        retain their physical legal actions for value-function bootstrapping;
        won/lost observations have no legal actions.
        """
        obs = np.asarray(obs, dtype=float)
        if obs.ndim != 1 or obs.size not in (10, 12, 13):
            raise ValueError("observation must be a vector of 10, 12, or 13 values")
        if not np.isfinite(obs).all() or (obs < 0).any():
            raise ValueError("observation values must be finite and nonnegative")
        mask = np.zeros(6, dtype=bool)
        if obs.size >= 12 and (obs[10] >= cls.WINS_TO_WIN or obs[11] <= 0):
            return mask
        gold, shop, team = obs[0], obs[2:5], obs[5:10]
        has_empty = bool((team == 0).any())
        for i, pet in enumerate(shop):
            can_combine = 0 < pet < cls.MAX_PET_STRENGTH and bool((team == pet).any())
            mask[i] = gold >= cls.BUY_COST and pet > 0 and (has_empty or can_combine)
        mask[3] = gold >= cls.ROLL_COST
        mask[4] = bool((team > 0).any())
        mask[5] = True
        return mask

    def _action_mask(self) -> np.ndarray:
        return self.mask_from_obs(self._obs())

    def _info(self) -> dict:
        return {
            "turn": int(self.turn),
            "gold": int(self.gold),
            "wins": int(self.wins),
            "lives": int(self.lives),
            "team_strength": int(self.team_pets.sum()),
            "shop_actions": int(self.shop_actions),
            "episode_steps": self._episode_steps,
            "action_mask": self._action_mask(),
            "observation_version": self.OBSERVATION_VERSION,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.gold = self.MAX_GOLD
        self.turn = self.wins = self.shop_actions = 0
        self.lives = self.START_LIVES
        self.team_pets = np.zeros(self.TEAM_SLOTS, dtype=np.int32)
        self._episode_steps = 0
        self._episode_return = 0.0
        self._last_battle = None
        self._needs_reset = False
        self._roll_shop()
        return self._obs(), self._info()

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before stepping a new or finished episode")
        if (isinstance(action, (bool, np.bool_))
                or not isinstance(action, (int, np.integer))
                or not self.action_space.contains(action)):
            raise ValueError(f"action must be an integer in 0..5, got {action!r}")
        action = int(action)
        action_valid = bool(self._action_mask()[action])
        terminated = False
        self._last_battle = None
        if action == 5:
            reward, terminated = self._end_turn()
        else:
            self.shop_actions += 1
            if not action_valid:
                reward = self.ILLEGAL_ACTION_PENALTY
            elif action < self.SHOP_SLOTS:
                reward = self._buy(action)
            elif action == 3:
                reward = self._roll()
            else:
                reward = self._sell_weakest()
            reward -= self.STEP_PENALTY

        reason = None
        if terminated:
            reason = "victory" if self.wins >= self.WINS_TO_WIN else "defeat"
        elif self.turn >= self.MAX_TURNS:
            reason = "turn_limit"
        elif self.shop_actions >= self.max_shop_actions:
            reason = "shop_action_limit"
        truncated = reason in ("turn_limit", "shop_action_limit")
        self._episode_steps += 1
        self._episode_return += float(reward)
        self._needs_reset = bool(terminated or truncated)
        info = self._info()
        info["action_valid"] = action_valid
        if self._last_battle is not None:
            info["battle"] = self._last_battle.copy()
        if self._needs_reset:
            info["episode_reason"] = reason
            info["episode"] = {
                "r": self._episode_return,
                "l": self._episode_steps,
                "wins": int(self.wins),
                "lives": int(self.lives),
                "turns": int(self.turn),
                "outcome": reason,
            }
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), bool(terminated), truncated, info

    def _buy(self, slot: int) -> float:
        pet = int(self.shop_pets[slot])
        self.gold -= self.BUY_COST
        self.shop_pets[slot] = 0
        match = np.flatnonzero(self.team_pets == pet)
        if match.size:
            self.team_pets[match[0]] = min(pet + 1, self.MAX_PET_STRENGTH)
            return self.STRENGTH_REWARD * (int(self.team_pets[match[0]]) - pet)
        empty = np.flatnonzero(self.team_pets == 0)
        self.team_pets[empty[0]] = pet
        return self.STRENGTH_REWARD * pet

    def _roll(self) -> float:
        self.gold -= self.ROLL_COST
        self._roll_shop()
        return -0.02

    def _sell_weakest(self) -> float:
        occupied = np.flatnonzero(self.team_pets > 0)
        weakest = occupied[np.argmin(self.team_pets[occupied])]
        lost_strength = int(self.team_pets[weakest])
        self.team_pets[weakest] = 0
        self.gold += self.SELL_VALUE
        return -self.STRENGTH_REWARD * lost_strength

    def _end_turn(self):
        team_strength = int(self.team_pets.sum())
        enemy_strength = 3 + self.turn * 2 + int(self.np_random.integers(0, 4))
        if team_strength > enemy_strength:
            self.wins += 1
            reward, outcome = 1.0, "win"
        elif team_strength < enemy_strength:
            self.lives -= 1
            reward, outcome = -1.0, "loss"
        else:
            reward, outcome = 0.0, "draw"
        self._last_battle = {
            "turn": int(self.turn), "outcome": outcome,
            "team_strength": team_strength, "enemy_strength": enemy_strength,
        }
        self.turn += 1
        self.shop_actions = 0
        terminated = self.wins >= self.WINS_TO_WIN or self.lives <= 0
        if self.wins >= self.WINS_TO_WIN:
            reward += 5.0
        elif self.lives <= 0:
            reward -= 2.0
        # Keep the final board intact instead of inventing a post-game shop.
        # Time limits still expose a valid next decision for bootstrapping.
        if not terminated:
            self.gold = self.MAX_GOLD
            self._roll_shop()
        return reward, terminated

    def render(self):
        print(
            f"Turn {self.turn:2d} | Gold {self.gold:2d} | "
            f"Wins {self.wins} | Lives {self.lives} | "
            f"Shop {list(self.shop_pets)} | Team {list(self.team_pets)}"
        )

    def close(self):
        pass
