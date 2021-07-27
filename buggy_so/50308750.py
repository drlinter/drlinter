# Import the gym module
import gym

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4') #@DRLinter-->create_env
# Reset it, returns the starting frame
frame = env.reset() #@DRLinter-->initialize_env_correct
# Render
env.render()

is_done = False
while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  frame, reward, is_done, _ = env.step(env.action_space.sample()) #@DRLinter-->in_correct_step
  # Render
  env.render()
