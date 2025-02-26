from main import SuperAutoPetsEnv  # Import the environment

print("Creating environment...")  
env = SuperAutoPetsEnv()  # Initialize environment

print("Resetting environment...")  
obs, _ = env.reset()
print("Initial state:", obs)  # Should print initial gold value (e.g., [10.])

# Perform 5 random actions
for i in range(5):
    print(f"\nTurn {i+1}: Taking action...")
    action = env.action_space.sample()  # Pick a random action (0 or 1)
    print("Action chosen:", action)

    obs, reward, done, _ = env.step(action)  # Perform the action
    print(f"Action: {action}, Reward: {reward}, New Gold: {obs}, Done: {done}")

    env.render()  # Show game state

    if done:
        print("Game Over!")
        break  # Stop if game ends
