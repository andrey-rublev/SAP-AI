from sapai.pets import Pet
from sapai.teams import Team
from sapai.battle import Battle
from agent import QLearningAgent

# Define the possible actions in the game
actions = ['buy', 'roll', 'battle']

# Initialize the Q-learning agent
agent = QLearningAgent(actions)

# Define a function to simulate the environment's step method (to be used by the agent)
class SAPEnvironment:
    def __init__(self):
        self.team = Team([])  # Create an empty team (no pets at the start)
        self.enemy_team = Team(["sheep", "tiger"])  # Enemy team setup

    def reset(self):
        """Reset the environment (team setup and enemy team)."""
        self.team = Team([])  # Reset to an empty team
        self.enemy_team = Team(["sheep", "tiger"])  # Reset enemy team
        return str(self.team)  # Return a string representation of the team as state

    def step(self, action):
        """Simulate a step in the environment based on the action taken."""
        reward = 0
        done = False

        if action == 'buy':
            # Add a new pet to the team and handle any logic for buying pets
            pet = Pet("ant")  # Example of a pet being added
            # Find the first empty slot in the team
            for i in range(5):  # Assuming team has 5 slots
                if self.team.get_slot(i).is_empty():
                    self.team.move(pet, i)  # Place the pet in the empty slot
                    reward = 1  # Reward for buying a pet
                    break

        elif action == 'roll':
            # Logic for rolling the shop (e.g., refreshing available pets)
            reward = 0  # No immediate reward for rolling

        elif action == 'battle':
            # Simulate a battle
            battle = Battle(self.team, self.enemy_team)
            winner = battle.battle()
            if winner == 0:  # Team 0 wins
                reward = 10
            elif winner == 1:  # Team 1 wins (enemy wins)
                reward = -10
            else:  # Draw
                reward = 5
            done = True  # Battle ends the episode

        return str(self.team), reward, done, {}

# Initialize the environment
env = SAPEnvironment()

# Train the agent with the environment
agent.train(env, num_episodes=1000)

# Now the agent can take actions based on its learned policy
