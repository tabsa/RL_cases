## Deep RL for Cart Pole problem
# Policy that maximizes the total cumulative reward at the end of an episode
# action_opt = pi_opt(state)
# We let the agent learn over 20 episodes of 100 time steps each.
# A reward of +1 is provided for every timestep that the pole remains upright.
# The episode ends if:
# 1. Pole Angle is more than ±12°
# 2. Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
# 3. Episode length is greater than 200
# The problem is considered solved when the average reward is greater than or equal to 195 over 100 consecutive trials.

# Import packages
import gym
import numpy as np

#%% Deep RL algorithms
# Random policy
def rand_policy(env):
    # Return a random action: 0 (left) or 1 (right)
    action = env.action_space.sample()
    return action
# Rule-based policy
def rule_policy(env, time):
    # Naive (and simple) policy
    action = 0
    if time <= 20:
        action = 0 # Go left
    elif time > 20:
        action = 1 # Go right
    return action
# Alternative-based policy
def alternative_policy(env, time):
    action = 0 # Go left
    if time%2 == 1: # If the timestep is odd number
        action = 1 # Go right
    return action

#%% Main code
# Create an environment - Cart Pole case
env_cart = gym.make('CartPole-v1')
print(env_cart.action_space) # 2 actions: 0 (left) or 1 (right)
print(env_cart.observation_space) # State defined by 4 parameters: cart_position, cart_velocity, pole_angle, pole_velocity

# Single episode example
# time_step = 100 # t \in T, timestep of 100. Each episode will have this timestep
# rewards = []
# state = env_cart.reset()
# k = 1 # epi_id
# for i in range(time_step):
#     env_cart.render()
#     action = rand_policy(env_cart)
#     state, reward, done, info = env_cart.step(int(action))
#     rewards.append(reward)
#     if done:
#         cumulative_rd = sum(rewards)
#         print(f'Episode {k} finished after {i} timesteps. Total reward: {cumulative_rd}')
#         break
# env_cart.close()

# 20 episodes to learn the pi_opt(state)
state_size = env_cart.observation_space.shape[0]
# state_size = len(env.observation_space.sample()) # Alternative way of doing it
no_epi = 20
time_step = 100
action_np = np.zeros((no_epi, time_step))
state_np = np.zeros((state_size, no_epi, time_step))
reward_np = np.zeros((no_epi, time_step))
for i in range(no_epi): # For-loop i \in no_epi
    state_np[:,i,0] = env_cart.reset() # Reset the env for each i
    reward = []
    for t in range(time_step-1): # For-loop t \in time_step
        env_cart.render() # Display the env, just for visualization purpose (no reason in real application)
        # Select the next action_t+1
        # action_np[i,t] = rand_policy(env_cart) # Using the random policy
        # action_np[i,t] = rule_policy(env_cart, t) # Using the rule-based policy
        action_np[i,t] = alternative_policy(env_cart, t) # Using the alternative-based policy
        # Env receives the action_t, responds with observation
        state_np[:,i,t+1], reward, done, info = env_cart.step(int(action_np[i,t]))
        reward_np[i,t+1] = reward # Reward per t+1
        # Check if the episode is done
        if done: # If the pole is > 15 deg from vertical or the cart move by > 2.4 unit from the centre
            print(f'episode {i} finished after {t+1} timesteps. Total reward: {reward_np[i,:].sum()}')
            break
# Print out the max reward.sum()
print('Maximum cumulative reward should be >= 195')
env_cart.close()
