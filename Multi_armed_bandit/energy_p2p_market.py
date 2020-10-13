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
no_episodes = 10
target_sample = np.zeros(no_episodes)
target_bounds = np.array([3, 15])
# Plot parameters
cmap = plt.get_cmap("tab10", 15)
sns.set_style("whitegrid")

# For-loop per simulation
for e in range(no_episodes):
    target_sample[e] = np.random.uniform(low=target_bounds[0], high=target_bounds[1])
    env = trading_env(no_agents, no_trials, 'offers_input.csv', 'External_sample', target_sample[e])
    # Episode print
    print(f'Episode {e} - Energy target {target_sample[e]}')
    # Simulate the Random-policy
    rd_agent = trading_agent(env, 'Random_policy')
    env.run(rd_agent)
    rd_agent.plot_action_choice(cmap, f'Random policy across {no_trials} trials of episode {e}')
    # Simulate the eGreedy-policy
    eg_agent = trading_agent(env, 'e-greedy_policy', time_learning=10, e_greedy=0.25)  # Agent using the e-Greedy policy
    env.run(eg_agent)
    eg_agent.plot_action_choice(cmap, f'e-Greedy policy across {no_trials} trials of episode {e}')
    # Simulate the Thompson-Sampler-policy
    ts_agent = trading_agent(env, 'Thompson_Sampler_policy')
    env.run(ts_agent)
    ts_agent.plot_action_choice(cmap, f'Thompson-Sampler policy across {no_trials} steps of episode {e}')

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

#%% Input of the Multi-Armed Bandit (MAD) problem
#
# # Agent performance
# rd_agent = Agent(env, 'Random_policy') # Agent using the Random policy
# eg_agent = Agent(env, 'e-greedy_policy', time_learning=500, e_greedy=0.1) # Agent using the e-Greedy policy
# ts_agent = Agent(env, 'Thompson_Sampler_policy') # Agent using the Thompson-Sampler policy
# # Simulation with rd_agent
# env.run(rd_agent)
# rd_agent.plot_action_choice(cmap, f'Random policy across {n_trials} steps')
# # Simulation with eg_agent
# env.run(eg_agent)
# eg_agent.plot_action_choice(cmap, f'e-Greedy policy across {n_trials} steps')
# # Simulation with ts_agent
# env.run(ts_agent)
# ts_agent.plot_action_choice(cmap, f'Thompson-Sampler policy across {n_trials} steps')
#
# # Plot regret over trial (opportunity cost of selecting a better action)
# # Shows the results for the agents of the last test - n_tests = 10
# plt.figure(figsize=(10,7))
# # Subplot 1
# plt.subplot(211) # 2 rows and 1 column
# plt.plot(np.cumsum(1-rd_agent.theta_regret_t), label='random') # Agent with random policy
# plt.plot(np.cumsum(1-eg_agent.theta_regret_t), label='e-greedy') # Agent with e-Greedy policy
# plt.plot(np.cumsum(1-ts_agent.theta_regret_t), label='thompson') # Agent with Thompson-Sampler policy
# plt.xlabel('no Trials #', fontsize=16)
# plt.ylabel('Regret', fontsize=16)
# plt.title(f'Cumulative regret across {n_trials} steps')
# plt.legend()
# # Subplot 2
# plt.subplot(212)
# plt.plot(1-rd_agent.theta_regret_t, label='random')
# plt.plot(1-eg_agent.theta_regret_t, label='e-greedy')
# plt.plot(1-ts_agent.theta_regret_t, label='thompson')
# plt.xlabel('no Trials #', fontsize=16)
# plt.ylabel('Regret', fontsize=16)
# plt.title(f'Regret per time t (1,...,{n_trials})')
# plt.legend()
# plt.show()
#
# # Simulation part
# n_tests = 10
# rd_score = np.zeros((n_agents,n_tests)) # Reward score per simulation
# # For-loop i from [1,...,n_tests]
# for i in range(n_tests):
#     print(f'Simulation {i+1}')
#     # Agent using the Random policy
#     rd_agent = Agent(env, 'Random_policy')
#     env.run(rd_agent)
#     rd_score[0,i] = rd_agent.total_reward
#     # Agent using the e-Greedy policy
#     eg_agent = Agent(env, 'e-greedy_policy', time_learning=500, e_greedy=0.1)
#     env.run(eg_agent)
#     rd_score[1,i] = eg_agent.total_reward
#     # Agent using the Thompson-Sampler policy
#     ts_agent = Agent(env, 'Thompson_Sampler_policy')
#     env.run(ts_agent)
#     rd_score[2,i] = ts_agent.total_reward
#
# print(f'All {n_tests} Simulations Done')
# # Plot comparing the 3 policies implemented to the MAD problem
# plt.figure(figsize=(10,7))
# x = np.arange(0, n_tests)
# plt_label = ['random', 'e-greedy', 'thompson']
# plt_marker = ['o', '*', '^']
# for i in range(n_agents):
#     plt.plot(x,rd_score[i,:], label=plt_label[i], marker=plt_marker[i], linestyle='--')
# # Legend and labels of the plot
# plt.legend(fontsize=16)
# plt.ylabel('Total Reward', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel('Simulation #', fontsize=16)
# plt.title(f'Comparison of policies in MAD problem across {n_tests} simulations', fontsize=20)
# plt.show()
