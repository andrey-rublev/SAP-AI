from game import SuperAutoPetsEnv

def main():
    env = SuperAutoPetsEnv()
    obs = env.reset()
    print("Initial state:", obs)
    
    done = False
    while not done:
        action = env.action_space.sample()  # Example of taking a random action
        obs, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, New State: {obs}, Done: {done}")
        env.render()  # Optionally, visualize game state
    
if __name__ == "__main__":
    main()
