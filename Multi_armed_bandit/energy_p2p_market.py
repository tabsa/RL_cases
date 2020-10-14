## RL agent for the Energy P2P market as Multi-Armed Bandit (MAD) problem
# Application of MAD problem to the energy P2P market
#

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import Class and functions
from p2p_class_code import p2p_env, market_agents, p2p_RL_agent # Class of the P2P market example
from p2p_as_mad_class import trading_env, trading_agent

#%% Script - Using the p2p_as_mad_class
# no_trials = 40
# no_agents = 15
#
# # Build the p2p market env
# env = trading_env(no_agents, no_trials, 'offers_input.csv')
# # Agent performance
# rd_agent = trading_agent(env, 'Random_policy')
# eg_agent = trading_agent(env, 'e-greedy_policy', time_learning=10, e_greedy=0.25) # Agent using the e-Greedy policy
# # Simulation with rd_agent
# env.run(rd_agent)
# env.run(eg_agent)
#
# # Plot parameters
# cmap = plt.get_cmap("tab10", 15)
# sns.set_style("whitegrid")
# rd_agent.plot_action_choice(cmap, f'Random policy across {no_trials} trials')
# eg_agent.plot_action_choice(cmap, f'e-Greedy policy across {no_trials} trials')

#%% Informs simulation
no_trials = 40
no_agents = 15
no_episodes = 100
no_RL_agents = 3
target_sample = np.zeros(no_episodes)
target_bounds = np.array([3, 15])
# Plot parameters
cmap = plt.get_cmap("tab10", 15)
sns.set_style("whitegrid")

rd_score = np.zeros((no_RL_agents,no_episodes)) # Reward score per simulation
# For-loop per simulation [1,...,no_episodes]
for e in range(no_episodes):
    target_sample[e] = np.random.uniform(low=target_bounds[0], high=target_bounds[1])
    env = trading_env(no_agents, no_trials, 'offers_input.csv', 'External_sample', target_sample[e])
    # Episode print
    print(f'Episode {e} - Energy target {target_sample[e]}')
    # Simulate the Random-policy
    rd_agent = trading_agent(env, 'Random_policy')
    env.run(rd_agent)
    #rd_agent.plot_action_choice(cmap, f'Random policy across {no_trials} trials of episode {e}')
    rd_score[0, e] = rd_agent.total_reward
    # Simulate the eGreedy-policy
    eg_agent = trading_agent(env, 'e-greedy_policy', time_learning=10, e_greedy=0.25)  # Agent using the e-Greedy policy
    env.run(eg_agent)
    #eg_agent.plot_action_choice(cmap, f'e-Greedy policy across {no_trials} trials of episode {e}')
    rd_score[1, e] = eg_agent.total_reward
    # Simulate the Thompson-Sampler-policy
    ts_agent = trading_agent(env, 'Thompson_Sampler_policy')
    env.run(ts_agent)
    #ts_agent.plot_action_choice(cmap, f'Thompson-Sampler policy across {no_trials} steps of episode {e}')
    rd_score[2, e] = ts_agent.total_reward

print(f'All {no_episodes} Simulations Done')
# Plot comparing the 3 policies implemented to the MAD problem
plt.figure(figsize=(10,7))
x = np.arange(0, no_episodes)
plt_label = ['Random', 'e-Greedy', 'Thompson']
plt_marker = ['o', '*', '^']
for i in range(no_RL_agents):
    plt.plot(x,rd_score[i,:], label=plt_label[i], marker=plt_marker[i], linestyle='--')
# Legend and labels of the plot
plt.legend(fontsize=16)
plt.ylabel('Total Reward', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Episode #', fontsize=16)
# plt.title(f'Comparison of policies in MAD problem across {n_tests} simulations', fontsize=20)
plt.show()

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
