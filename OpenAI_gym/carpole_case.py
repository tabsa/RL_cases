## Deep RL for Cart Pole problem
# Policy that maximizes the total cumulative reward at the end of an episode
# action_opt = pi_opt(state)
# We let the agent learn over 20 episodes of 100 time steps each.
# A reward of +1 is provided for every timestep that the pole remains upright.
# The episode ends if:
# 1. Pole Angle is more than ±12°
# 2. Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
# 3. Episode length is greater than 200
# The problem is considered solved when the average reward is greater than or equal to 195
# over 100 consecutive trials.

# Import packages
import gym
import numpy as np

#%% Deep RL algorithms
# Random policy
def rand_policy(env):
    # Return a random action: 0 (left) or 1 (right)
    action = env.action_space.sample()
    return action

#%% Main code
# Create an environment - Cart Pole case
env = gym.make('CartPole-v1')
print(env.action_space) # 2 actions: 0 (left) or 1 (right)
print(env.observation_space) # State defined by 4 parameters: cart_position, cart_velocity, pole_angle, pole_velocity

# Call the random policy
time_step = 100 # t \in T, timestep of 100. Each episode will have this timestep
action_np = np.zeros(time_step)
# Single episode example
for i in range(time_step):
    action_np[i] = rand_policy(env) # Selects a random action
print(action_np)

# 20 episodes to learn the pi_opt(state)
no_epi = 20
time_step = 100
action_np = np.zeros((no_epi, time_step))
state_np = np.zeros((no_epi, time_step))
reward_np = np.zeros((no_epi, time_step))
rd_sum_np = np.zeros((no_epi))
for i in range(no_epi): # For-loop i \in no_epi
    state_np[i,0] = env.reset() # Reset the env for each i
    for t in range(time_step-1): # For-loop t \in time_step
        env.render() # Display the env, just for visualization purpose (no reason in real application)
        # Select the next action_t+1
        action_np[i,t] = rand_policy(env) # Using the random policy
        # Env receives the action_t, responds with observation
        state_np[i,t+1], reward_np[i,t+1], done, info = env.step(action_np[i,t])
        # Cumulative reward
        rd_sum_np[i] += reward_np[i,t+1]
        # Check if the episode is done
        if done:
            print(f'episode {i} finished after {t+1} timesteps. Total reward: {rd_sum_np[i]}')
            break
env.close()