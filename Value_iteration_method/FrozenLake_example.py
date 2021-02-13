## Value Iteration method for Frozen Lake problem
# This method numerically calculates the Q-function of any type of Discrete (state, action) type problem.
# The Frozen Lake problem is a well known classic problem. The idea is to find the best route between A and B in a grid env 4x4
# We have 16 states and 4 actions (up, down, left, right)
# Point A is on the top-left of the grid, and the point B (goal) is on the bottom-right corner
# There are holes in the grid and if the episode ends either when agent reaches point B or reaches one of the holes
# Reward:=1 for reaching point B, and reward:=0 for reaching one of the holes
# Since is a 'Frozen Lake' everytime the agents moves it gets slippery, There is 33% of moving 90 degrees for example:
#  - Deciding to move left - 33 % chance of happening
#       - 33 % chance of moving top (-90 degrees)
#       - 33 % chance of moving down (90 degrees)

#%% Import packages
import gym
from Val_Iter_RL_agent import *
from torch.utils.tensorboard import SummaryWriter

#%% Parameters
env_name = 'FrozenLake-v0'
gamma = 0.9 # Discount factor for the value function
episodes = 20 # Episodes for testing
ratio_success_rewards = 0.8 # Ratio of successful simulation -> Simulation with 80 % of successful episodes

#%% Main Script
if __name__ == '__main__':
    test_env = gym.make(env_name) # Separate the testing from the training part
    agent = VI_agent(env_name, gamma)
    writer = SummaryWriter('dashboard/logdir') #comment='-v-iteration')

    sim = 0 # Counter for the no of simulations
    best_reward = 0.0
    while True:
        sim += 1
        agent.get_samples(100) # 100th random samples
        agent.update_value_fun() # Update Value function for every state and action AFTER the random sample

        total_reward_sim = 0.0 # Sum of rewards per simulation
        for _ in range(episodes): # Run all episodes for the simulation iter_no
            total_reward_sim += agent.episode_sim(test_env) # This case we use the test_env to don't interfere with the env used by the agent
        total_reward_sim /= episodes # Success rate of the simulation
        writer.add_scalar("reward", total_reward_sim, sim) # Write in the tensorboard the result
        if total_reward_sim > best_reward: # Best reward of all simulations so far
            print(f'Sim {sim}: Success reward rate {total_reward_sim:.3f} -> Best reward {best_reward:.3f}')
            best_reward = total_reward_sim # Update the best reward info
        if total_reward_sim > ratio_success_rewards: # Boundary for when we finish the simulation, we can consider a bigger one
            print(f'Solved the {env_name} problem in {sim} simulations. Well done!!!')
            break
    writer.close()