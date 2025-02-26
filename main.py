import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SuperAutoPetsEnv(gym.Env):
    """A Super Auto Pets environment for RL."""

    def __init__(self):
        super(SuperAutoPetsEnv, self).__init__()
        self.gold = 10  # Starting gold
        self.turn = 0
        self.max_turns = 5
        self.action_space = spaces.Discrete(3)  # 0: Select pet, 1: Roll, 2: Start battle
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)  # Example of observation space (adjust based on actual game data)

    def step(self, action):
        # Simulate a step based on action (select pet, roll, start battle)
        reward = 0
        self.turn += 1

        if action == 0:  # Select a pet
            self.gold -= 3  # Assume pet costs 3 gold
            reward = 1  # Reward for purchasing a pet

        elif action == 1:  # Roll the shop
            self.gold -= 1  # Rolling costs 1 gold
            reward = -1  # Penalty for rolling

        elif action == 2:  # Start battle
            # Simulate battle result: win/lose
            win = np.random.choice([True, False])  # Random win/lose
            reward = 10 if win else -5  # Reward based on battle result

        done = self.turn >= self.max_turns
        return np.array([self.gold, self.turn], dtype=np.float32), reward, done, {}

    def reset(self):
        """Reset the environment to its initial state."""
        self.gold = 10
        self.turn = 0
        return np.array([self.gold, self.turn], dtype=np.float32)
