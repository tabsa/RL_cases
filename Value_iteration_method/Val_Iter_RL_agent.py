## RL agent using Value iteration method
# Agent using Value Iteration to numerically calculate the Q-function (state, action).

#%% Import packages
import gym
import collections
import numpy as np

#%% Class section
class VI_agent:
    def __init__(self, env, gamma=0.99):
        self.env = gym.make(env) # Assign the gym environment
        self.gamma = gamma # Assign the discount factor
        self.state = self.env.reset() # Initial state
        self.rewards = collections.defaultdict(float) # Create dict for storing the reward - key: (state_t + action_t + state_t+1) = sum(reward)
        self.trans_prob = collections.defaultdict(collections.Counter) # Create dict for transition probability - key: (state_t + action_t) = dict(states_t+1 = Count)
        # collection.Counter builds a dict for counting the times for each state_t+1
        # Example: trans_prob - key: (0, 1) = dict(1: 3, 2: 0, 3: 2, 4:1)
        self.value_fun = collections.defaultdict(float) # Create dict for value function - key: (state) = sum (value equation)

    def get_samples(self, no_sample): # Generate random samples for the reward, trans_prob and value_fun
        for _ in range(no_sample):
            action = self.env.action_space.sample() # Random action
            new_state, reward, is_done, _ = self.env.step(action) # Get env observation: state_t+1, rd_t+1
            self.rewards[(self.state, action, new_state)] = reward # Update Reward signal (immediate rd(s,a,s'))
            self.trans_prob[(self.state, action)][new_state] += 1 # Count of times that new_state was selected
            self.state = self.env.reset() if is_done else new_state # Go to the next_state if is_done == False

    def eq_action_value(self, state, action): # Calculate Q(s,a) table after get_samples
        state_targets = self.trans_prob[(state, action)] # No times for all state_t+1 fixing (state_t, action_t)
        total = sum(state_targets.values()) # Total count, to calculate the prob(s_t+1, a_t) = state_targets / counts
        action_value = 0.0
        for next_state, no_count in state_targets.items():
            reward = self.rewards[(state, action, next_state)] # Get the reward for each element
            action_value += (no_count / total) * (reward + self.gamma * self.value_fun[next_state]) # Q(s,a) = sum(s_t+1, prob(s_t+1) * (rd(s_t+1) + gamma * Value(s_t+1)) )
            # Get the value of Q(s,a), REMEMBER that gamma * max(Q(s,a)) which is the value_fun term
        return action_value

    def select_action(self, state): # Select the action based on the Q(s,a) table - > max_a (Q(s,a))
        # This function does an extensive search of best action for the state_t
        # Get the list of Q(s_t, a_t) for all action in the state_t
        action_value = [self.eq_action_value(state, action) for action in range(self.env.action_space.n)] # For-loop for every action in the env
        return np.argmax(action_value) # Return the best action <- argmax[Q(s,a)]

    def episode_sim(self, env): # Run simulation per episode, AFTER the get_sample phase
        total_reward = 0.0
        state = env.reset() # Initial state (random)
        while True: # Endless loop until is_done == True
            action = self.select_action(state) # Select the best action
            new_state, reward, is_done, _ = env.step(action) # Get env observation - rd_t, state_t+1, is_done
            self.rewards[(state, action, new_state)] = reward # Update the reward for (state_t, action_t, state_t+1)
            self.trans_prob[(state, action)][new_state] += 1 # No times for the next_state - transition probablity
            total_reward += reward # Sum of episode reward = sum(t, rd_t)
            if is_done: # Finish the episode if is_done == True (return by gym env)
                break
            state = new_state # state = state_t+1
        return total_reward # Episode reward

    def value_iteration_update(self): # Calculate V(s_t) = max(a_t, sum(s_t+1, prob(s_t,a_t,s_t+1) * ( r(s_t,a_t) + gamma * V(s_t+1)) ) )
        # This function first needs to get the V(s_t+1), so knowing the value function of forward state_t+1
        # This way we get V(s_t) by choosing the max V(s_t+1) for all action_t of state_t
        # REMEMBER that each state_t we want to know what action_t maximizes its own Value
        for state in range(self.env.observation_space.n): # For-loop for every state in the env
            # Calculate the Q(s,a) for all next_state after our state_t, this way we can get the V(s_t+1) <- max( Q(s_t,a_t, s_t+1) )
            state_values = [self.eq_action_value(state, action) for action in range(self.env.action_space.n)]
            self.value_fun[state] = max(state_values) # max operation
