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
from plot_class import plot_action_choice, plot_reward_per_episode, plot_regret_prob

#%% Script - Using the p2p_as_mad_class
no_trials = 40
no_agents = 15
no_episodes = 100
no_RL_agents = 3
target_sample = np.zeros(no_episodes)
target_bounds = np.array([3, 15])
rd_score = np.zeros((no_RL_agents,no_episodes)) # Reward score per episode_e
rd_score_trial = np.zeros((no_RL_agents, no_episodes, no_trials))
regret_score = np.zeros((no_RL_agents, no_episodes, no_trials)) # Regret probability per (epi, agent, trial)
ax_label = ['no Trials #', 'Offers', 'Energy']
plt_title = ['ABC', 'Energy traded per trial']
# For-loop per episode
#target_sample = np.random.uniform(low=target_bounds[0], high=target_bounds[1], size=no_episodes)
rd_score = np.zeros((no_RL_agents,no_episodes)) # Reward score per simulation
# Create environment and agent
env = trading_env(no_agents, no_trials, 'offers_input.csv')
agent_list = [] # List with all RL agent
#agent_list.append(trading_agent(env, target_bounds, 'Random_policy')) # Agent using the Random policy
#agent_list.append(trading_agent(env, target_bounds, 'e-greedy_policy', time_learning=10, e_greedy=0.25)) # Agent using the e-Greedy policy
agent_list.append(trading_agent(env, target_bounds, 'Thompson_Sampler_policy')) # Agent using the Thompson-Sampler policy

ag = 0 # id of agent
batch_size = 10
for agent in agent_list:
    for e in range(no_episodes):
        # Episode print
        print(f'Episode {e} - Energy target {target_sample[e]}')
        # Simulate the agent interaction
        if agent.is_reset or e == 0:
            env.run(agent) # Run environment
            agent.memory.append((agent.a, agent.b, agent.total_reward, agent.id_n, agent.state_n[agent.id_n]))
            if len(agent.memory) > batch_size:
                agent.exp_replay(batch_size)

            # Store final results in np.arrays
            rd_score[ag, e] = agent.total_reward
            rd_score_trial[ag, e, :] = agent.reward_n
            regret_score[ag, e, :] = agent.theta_regret_n
        agent.reset()
        # plt_title[0] = f'e-Greedy policy across {no_trials} trials of episode {e}'
        # plot_action_choice(eg_agent, ax_label, plt_title)

    # Next agent (1st For-loop)
    ag += 1

# print(f'All {no_episodes} Episodes are done')
# # Plot of regret probability comparing the 3 policies here
# epi_sel = [0, 10, 23, 45, 49]
# label = ['random', 'e-greedy', 'thompson']
# ax_label = ['no Trials #', 'Regret probability']
# for i in epi_sel:
#     plt_title = [f'Cumulative regret across steps of episode {i}',
#                  f'Regret per step n of episode {i}']
#     plot_regret_prob(regret_score, i, label, ax_label, plt_title)
# # Plot comparing the 3 policies implemented to the MAD problem
# marker = ['o', '*', '^']
# ax_label = ['Simulation #', 'Total Reward']
# plot_reward_per_episode(rd_score, label, marker, ax_label, f'Comparison of policies in MAD problem across {no_episodes} simulations')

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
