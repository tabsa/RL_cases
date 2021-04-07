## RL agent for the Energy P2P market as Multi-Armed Bandit (MAD) problem
# Application of MAD problem to the energy P2P market
#

#%% Import packages
import numpy as np
import pandas as pd
import pickle as pkl
import os
# Import Class and functions
from p2p_class_code import p2p_env, market_agents, p2p_RL_agent # Class of the P2P market example
from p2p_as_mad_class import trading_env, trading_agent
from plot_class import *

#%% Simulation parameters
## Simulation
no_trials = 40 # per episode
no_episodes = 100
no_RL_agents = 3 # each agent has a different policy
batch_size = 20 # exp replay buffer
## P2P market
no_agents = 15
#target_bounds = np.array([3, 25])
#target_sample = np.random.uniform(low=target_bounds[0], high=target_bounds[1], size=no_episodes)
target_bounds = 15
target_sample = target_bounds * np.ones(no_episodes)
#target_bounds = np.arange(start=5, stop=51, step=5)
#target_sample = np.repeat(target_bounds, 20)
## Output data
agent_list = [] # List with all RL agent
outcome_agent = [] # List of outcome DF per RL agent
# Name of the elements for the outcome_agent DF
df_col_name = ['total_rd', 'final_step', 'energy_target', 'final_state', 'final_theta',
                'mean_theta', 'std_theta', 'final_regret', 'mean_regret', 'std_regret']
policy_agent = [] # List of policy solutions (array) per RL agent
policy_sol_epi = np.zeros((6, no_trials, no_episodes)) # Array to store policy solutions per episode
## Saving file
wk_dir = os.getcwd() # Define other if you want
out_filename = 'sim_results_fixed_target_15_exp_replay.pkl'
out_filename = os.path.join(wk_dir, out_filename)

#%% Create environment and agent
env = trading_env(no_agents, no_trials, 'offers_input.csv', 'External_sample', target_sample)
# Assign RL agents in the agent_list, each RL agent has a different policy strategy
agent_policy = ['Random_policy', 'e-greedy_policy', 'Thompson_Sampler_policy']
agent_list.append(trading_agent(env, target_bounds, agent_policy[0])) # Agent using the Random policy
agent_list.append(trading_agent(env, target_bounds, agent_policy[1], time_learning=10, e_greedy=0.25)) # Agent using the e-Greedy policy
agent_list.append(trading_agent(env, target_bounds, agent_policy[2])) # Agent using the Thompson-Sampler policy

#%% Simulation phase
ag = 0  # id of agent
for agent in agent_list: # For-loop per RL agent
    # Simulate the agent interaction
    print(f'Run the agent with the {agent.policy_opt}:')
    for e in range(no_episodes): # For-loop per episode e
        # Episode print
        print(f'Episode {e} - Energy target {target_sample[e]}')
        if agent.is_reset or e == 0:
            env.run(agent, e) # Run environment, inputs we have RL_agent and episode id
            # Store info in the memory
            agent.memory.append((agent.a, agent.b, agent.total_reward, agent.id_n, agent.state_n[agent.id_n]))
            if len(agent.memory) > batch_size: # and len(agent.memory) <= 50:
                agent.exp_replay(batch_size)
            # Store final results in np.arrays
            policy_sol_epi[:, :, e] = agent.policy_sol
        # Reset of both agent and environment
        agent.reset()

    outcome_agent.append(pd.DataFrame(agent.outcome, columns=df_col_name))
    policy_agent.append(policy_sol_epi)
    policy_sol_epi = np.zeros((6, no_trials, no_episodes)) # Reset the array for next agent in agent_list
    # Next agent (1st For-loop)
    ag += 1
    print('\n')
print(f'All {no_episodes} Episodes are done')

#%% Save simulation results
# Build a dictionary
data = {}
agents = {}
simulation = {}
# agents info
agents['no'] = no_RL_agents
agents['id'] = agent_policy
# simulation info
simulation['target'] = target_sample
simulation['environment'] = env
simulation['episodes'] = no_episodes
simulation['trials'] = no_trials
data['agents'] = agents
data['simulation'] = simulation
data['outcome'] = outcome_agent
data['policy'] = policy_agent
file = open(out_filename, 'wb')
pkl.dump(data, file)
file.close()
