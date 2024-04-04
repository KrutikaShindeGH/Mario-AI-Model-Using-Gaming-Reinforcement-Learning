from stable_baselines3 import PPO
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import numpy as np

# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0' ) 
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Load model
model = PPO.load('./train_3/best_model_3000000')

# Reset environment
state = env.reset()

# Start the game loop
while True: 
    # Ensure the numpy array is contiguous
    state = np.array(state, copy=True)
    
    # Get action from the model
    action, _ = model.predict(state)
    
    # Convert the action to integer
    action = int(action)
    
    # Step through the environment
    state, reward, done, info = env.step(action)
    env.render()

    # Check if the episode is done
    if done:
        print("Episode finished. Resetting environment...")
        state = env.reset()
