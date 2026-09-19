"""Super Auto Pets - lightweight Gymnasium environment.

A self-contained, dependency-light simulation of the Super Auto Pets shop/battle
loop, exposed through the standard Gymnasium API so any RL agent can train
against it without needing the game (or the heavier ``sapai`` engine) installed.

The real game only runs on the Steam desktop client; see ``autogui.py`` for the
screen-capture + mouse bridge that lets a trained policy play the actual client.
This module is the fast, headless surrogate used for training and testing.

Observation (Box, shape ``(10,)``, ``float32``)::

    [0]      gold        gold remaining this turn        (0..MAX_GOLD)
    [1]      turn        current turn index              (0..MAX_TURNS)
    [2:5]    shop_pets   strength of the 3 shop slots    (0..MAX_TIER)
    [5:10]   team_pets   strength of the 5 team slots    (0..~)

Actions (``Discrete(6)``)::

    0, 1, 2   buy shop slot 0/1/2 into the team   (costs BUY_COST gold)
    3         roll / refresh the shop             (costs ROLL_COST gold)
    4         sell the weakest team pet           (+SELL_VALUE gold)
    5         end the turn -> fight this round's opponent

An episode ends on victory (``WINS_TO_WIN`` wins) or defeat (0 lives), and is
truncated after ``MAX_TURNS`` turns.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SuperAutoPetsEnv(gym.Env):
    """A compact Super Auto Pets arena as a Gymnasium environment."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # --- game-balance constants ---------------------------------------------
    SHOP_SLOTS = 3
    TEAM_SLOTS = 5
    MAX_TIER = 6          # shop pets come in tiers 1..6
    MAX_GOLD = 10         # gold granted at the start of every turn
    START_LIVES = 5
    WINS_TO_WIN = 10
    MAX_TURNS = 30

    BUY_COST = 3
    ROLL_COST = 1
    SELL_VALUE = 1
    STEP_PENALTY = 0.05   # small cost per shop action, so dithering never pays

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        # obs = [gold, turn, shop(3), team(5)] -> 10 values
        high = np.array(
            [self.MAX_GOLD, self.MAX_TURNS]
            + [self.MAX_TIER] * self.SHOP_SLOTS
            + [50] * self.TEAM_SLOTS,  # team pets can be combined past MAX_TIER
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)

        # runtime state (populated by reset)
        self.gold = 0
        self.turn = 0
        self.wins = 0
        self.lives = 0
        self.shop_pets = np.zeros(self.SHOP_SLOTS, dtype=np.int32)
        self.team_pets = np.zeros(self.TEAM_SLOTS, dtype=np.int32)

    # ------------------------------------------------------------------ utils
    def _roll_shop(self) -> None:
        self.shop_pets = self.np_random.integers(
            1, self.MAX_TIER + 1, size=self.SHOP_SLOTS, dtype=np.int32
        )

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            ([self.gold, self.turn], self.shop_pets, self.team_pets)
        ).astype(np.float32)

    def _action_mask(self) -> np.ndarray:
        """Boolean mask (len 6) of actions that aren't illegal / no-ops."""
        mask = np.zeros(6, dtype=bool)
        has_empty = bool((self.team_pets == 0).any())
        for i in range(self.SHOP_SLOTS):
            pet = int(self.shop_pets[i])
            can_combine = pet > 0 and bool((self.team_pets == pet).any())
            mask[i] = self.gold >= self.BUY_COST and pet > 0 and (has_empty or can_combine)
        mask[3] = self.gold >= self.ROLL_COST                 # roll
        mask[4] = bool((self.team_pets > 0).any())            # sell weakest
        mask[5] = True                                        # end turn is always legal
        return mask

    def _info(self) -> dict:
        return {
            "turn": int(self.turn),
            "gold": int(self.gold),
            "wins": int(self.wins),
            "lives": int(self.lives),
            "team_strength": int(self.team_pets.sum()),
            "action_mask": self._action_mask(),
        }

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.gold = self.MAX_GOLD
        self.turn = 0
        self.wins = 0
        self.lives = self.START_LIVES
        self.team_pets = np.zeros(self.TEAM_SLOTS, dtype=np.int32)
        self._roll_shop()
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        terminated = False
        truncated = False

        if action in (0, 1, 2):
            reward = self._buy(action) - self.STEP_PENALTY
        elif action == 3:
            reward = self._roll() - self.STEP_PENALTY
        elif action == 4:
            reward = self._sell_weakest() - self.STEP_PENALTY
        elif action == 5:
            reward, terminated = self._end_turn()  # ending the turn isn't penalised
        else:
            raise ValueError(f"invalid action {action}")

        if not terminated and self.turn >= self.MAX_TURNS:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._obs(), float(reward), terminated, truncated, self._info()

    # ---------------------------------------------------------- action logic
    def _buy(self, slot: int) -> float:
        pet = int(self.shop_pets[slot])
        if pet == 0 or self.gold < self.BUY_COST:
            return -0.1  # empty slot or not enough gold: illegal
        self.gold -= self.BUY_COST
        self.shop_pets[slot] = 0
        # Combine with a matching team pet (a "level up") when possible.
        match = np.where(self.team_pets == pet)[0]
        if match.size:
            self.team_pets[match[0]] = min(self.team_pets[match[0]] + 1, 50)
            return 0.15 * pet
        empty = np.where(self.team_pets == 0)[0]
        if empty.size:
            self.team_pets[empty[0]] = pet
            return 0.1 * pet
        # Team full and no combine available: refund and treat as illegal.
        self.gold += self.BUY_COST
        self.shop_pets[slot] = pet
        return -0.1

    def _roll(self) -> float:
        if self.gold < self.ROLL_COST:
            return -0.1
        self.gold -= self.ROLL_COST
        self._roll_shop()
        return -0.02  # tiny cost so the agent doesn't roll forever

    def _sell_weakest(self) -> float:
        nonzero = np.where(self.team_pets > 0)[0]
        if nonzero.size == 0:
            return -0.1
        weakest = nonzero[np.argmin(self.team_pets[nonzero])]
        self.team_pets[weakest] = 0
        self.gold += self.SELL_VALUE
        return 0.0

    def _end_turn(self):
        team_strength = int(self.team_pets.sum())
        enemy_strength = 3 + self.turn * 2 + int(self.np_random.integers(0, 4))

        if team_strength > enemy_strength:
            self.wins += 1
            reward = 1.0
        elif team_strength < enemy_strength:
            self.lives -= 1
            reward = -1.0
        else:
            reward = 0.0

        # Start the next turn: gold refills, shop refreshes, team carries over.
        self.turn += 1
        self.gold = self.MAX_GOLD
        self._roll_shop()

        terminated = False
        if self.wins >= self.WINS_TO_WIN:
            reward += 5.0  # bonus for winning the run
            terminated = True
        elif self.lives <= 0:
            reward -= 2.0
            terminated = True
        return reward, terminated

    def render(self):
        print(
            f"Turn {self.turn:2d} | Gold {self.gold:2d} | "
            f"Wins {self.wins} | Lives {self.lives} | "
            f"Shop {list(self.shop_pets)} | Team {list(self.team_pets)}"
        )

    def close(self):
        pass
