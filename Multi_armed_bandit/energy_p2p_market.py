## RL agent for the Energy P2P market as Multi-Armed Bandit (MAD) problem
# Application of MAD problem to the energy P2P market
#

#%% Import packages
import numpy as np
import pandas as pd
# Import Class and functions
from p2p_class_code import p2p_env, market_agents, p2p_RL_agent # Class of the P2P market example
from p2p_as_mad_class import trading_env, trading_agent
from plot_class import plot_action_choice, plot_reward_per_episode, plot_regret_prob

#%% Define parameters
# Simulation
no_trials = 40
no_episodes = 100
no_RL_agents = 3
batch_size = 20
# P2P market
no_agents = 15
target_bounds = np.array([3, 25])
target_sample = np.random.uniform(low=target_bounds[0], high=target_bounds[1], size=no_episodes)
# Output data
rd_score = np.zeros((no_RL_agents,no_episodes)) # Reward score per episode_e
rd_score_trial = np.zeros((no_RL_agents, no_episodes, no_trials))
regret_score = np.zeros((no_RL_agents, no_episodes, no_trials)) # Regret probability per (epi, agent, trial)
# Plot
ax_label = ['no Trials #', 'Offers', 'Energy']
plt_title = ['ABC', 'Energy traded per trial']

#%% Create environment and agent
env = trading_env(no_agents, no_trials, 'offers_input.csv', 'External_sample', target_sample)
agent_list = [] # List with all RL agent
outcome_agent = [] # List of outcome DF per RL agent
# Name of the elements for the outcome_agent DF
df_col_name = ['total_rd', 'final_step', 'energy_target', 'final_state', 'final_theta',
                'mean_theta', 'std_theta', 'final_regret', 'mean_regret', 'std_regret']
policy_agent = [] # List of policy solutions (array) per RL agent
policy_sol_epi = np.zeros((6, no_trials, no_episodes)) # Array to store policy solutions per episode
# Assign RL agents in the agent_list, each RL agent has a different policy strategy
agent_list.append(trading_agent(env, target_bounds, 'Random_policy')) # Agent using the Random policy
agent_list.append(trading_agent(env, target_bounds, 'e-greedy_policy', time_learning=10, e_greedy=0.25)) # Agent using the e-Greedy policy
agent_list.append(trading_agent(env, target_bounds, 'Thompson_Sampler_policy')) # Agent using the Thompson-Sampler policy

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
            # if len(agent.memory) > batch_size and len(agent.memory) <= 50:
            #     agent.exp_replay(batch_size)
            # Store final results in np.arrays
            policy_sol_epi[:, :, e] = agent.policy_sol
            rd_score[ag, e] = agent.total_reward
            rd_score_trial[ag, e, :] = agent.reward_n
            regret_score[ag, e, :] = agent.theta_regret_n
        # Reset of both agent and environment
        agent.reset()
        # Plot graphs per episode e
        #plt_title[0] = f'e-Greedy policy across {no_trials} trials of episode {e}'
        #plot_action_choice(eg_agent, ax_label, plt_title)

    outcome_agent.append(pd.DataFrame(agent.outcome, columns=df_col_name))
    policy_agent.append(policy_sol_epi)
    policy_sol_epi = np.zeros((6, no_trials, no_episodes)) # Reset the array for next agent in agent_list
    # Next agent (1st For-loop)
    ag += 1
    print('\n')

print(f'All {no_episodes} Episodes are done')
# Plot of regret probability comparing the 3 policies here
epi_sel = [0, 10, 23, 45, 49]
label = ['random', 'e-greedy', 'thompson']
ax_label = ['no Trials #', 'Regret probability']
# for i in epi_sel:
#     plt_title = [f'Cumulative regret across steps of episode {i}',
#                  f'Regret per step n of episode {i}']
#     plot_regret_prob(regret_score, i, label, ax_label, plt_title)
# Plot comparing the 3 policies implemented to the MAD problem
marker = ['o', '*', '^']
ax_label = ['Simulation #', 'Total Reward']
plot_reward_per_episode(rd_score, label, marker, ax_label, f'Comparison of policies in MAD problem across {no_episodes} simulations')

#%% Script - Using the p2p_class_code
# no_hours = 100
# no_agents = 10
# time_sample = 5 # 5min
# time_step = int(60/time_sample) # 60min (=hour) --> no. time steps
# no_time = int(no_hours * time_step) # Total no of time steps
#
# # Build the agents in the P2P market
# p2p_agents = market_agents(no_agents, no_time, 'market_agents_input.csv') # Class that defines EVERY with the market agents
#
# # Call the p2p_env
# env = p2p_env(p2p_agents, no_time, time_step)
# #rd_agent = p2p_RL_agent(env, 'Random_policy')
# env.set_simulation()
