## Example of Environment and Agent class
# Basic example to show the building blocks of env and agent in RL problems
# It assumes always the same state (s_t = s_t-1) and random rewards

#%% Import packages
import random
import numpy as np

#%% Build the environment class
class Environment:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.steps_n = 0

    def get_observation(self): # Function to return the state_t
        return [0.0, 0.0, 0.0]

    def get_action_space(self): # Returns the action space
        return [0, 1] # 2 options - 0 or 1

    def is_done(self): # True/False - Stopping condition
        return self.steps_n == self.max_steps # Returns True/False depending on steps_left == 0, True at the end of the episode (steps_left = 0)

    def run(self, action): # Function to apply the action of agent
        if self.is_done(): # When we finish the episode
            raise Exception("Game is over")
        self.steps_n += 1 # t = t + 1
        return random.random()

#%% Build the agent class
class Agent:
    def __init__(self, steps):
        self.actions = np.zeros(steps)
        self.reward = np.zeros(steps)
        self.total_reward = 0.0

    def step(self, env):
        n = env.steps_n
        current_obs = env.get_observation()
        self.actions[n] = random.choice(env.get_action_space())
        #actions = env.get_action_space()
        self.reward[n] = env.run(self.actions[n])
        self.total_reward += self.reward[n]

#%% Main script
if __name__ == "__main__":
    steps = 10 # no. of steps
    env = Environment(steps)
    agent = Agent(steps)

    while not env.is_done():
        agent.step(env)

    print(f'Total reward got: {agent.total_reward:.4f}')