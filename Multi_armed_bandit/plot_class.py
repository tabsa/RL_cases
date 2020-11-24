## Class for the plots in the MAD examples
# class p2p_env - Defines the environment of the energy P2P market
# class Agents_characteristics - Defines the characteristics of each agents that prosumer_i trades energy from/to
# class prosumer_i - Defines the prosumer (agent) that interacts with the p2p_env

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%% Plot parameters
plt_colmap = plt.get_cmap("tab10", 15)
sns.set_style("whitegrid")

#%% Plot for the total reward over episodes
def plot_reward_per_episode(score, plt_label, plt_marker, axis_label, plt_title):
    # Get the plot parameters
    no_RL_agents = score.shape[0]
    no_episodes = score.shape[1]
    plt.figure(figsize=(10,7))
    x = np.arange(0, no_episodes)
    for i in range(no_RL_agents):
        plt.plot(x,score[i,:], label=plt_label[i], marker=plt_marker[i], linestyle='--')
    # Legend and labels of the plot
    plt.legend(fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(axis_label[0], fontsize=16)
    plt.title(plt_title, fontsize=20)
    plt.show()

def plot_action_choice(agent, axis_label, plt_title):
    plt.figure(figsize=(10,7))
    trials = np.arange(0, agent.env.no_trials)
    # Subplot 1
    plt.subplot(211)  # 2 rows and 1 column
    plt.scatter(trials, agent.action_n[0,:], cmap=plt_colmap, c=agent.action_n[0,:], marker='.', alpha=1)
    plt.title(plt_title[0], fontsize=16)
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.yticks(list(range(agent.env.no_offers)))
    plt.colorbar()
    # Subplot 2
    plt.subplot(212)
    plt.bar(trials, agent.action_n[1,:])
    #plt.bar(trials, self.state_n)
    plt.title(plt_title[1], fontsize=16)
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[2], fontsize=16)
    #plt.legend()
    plt.show()

def plot_regret_prob(regret_prob, epi_id, plt_label, axis_label, plt_title):
    # Plot regret over trial (opportunity cost of selecting a better action)
    agents = regret_prob.shape[0] # no RL agents
    plt.figure(figsize=(10, 7))
    # Subplot 1
    plt.subplot(211)  # 2 rows and 1 column
    for a in range(agents):
        plt.plot(np.cumsum(1 - regret_prob[a, epi_id, :]), label=plt_label[a])  # Plot per RL_agent
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.title(plt_title[0], fontsize=16)
    plt.legend()
    # Subplot 2
    plt.subplot(212)
    for a in range(agents):
        plt.plot(1 - regret_prob[a, epi_id, :], label=plt_label[a])
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.title(plt_title[1], fontsize=16)
    plt.legend()
    plt.show()

# def plot_exp_regret(target, offer_info, rd_signal):
#     # Plot the regret probability over step_n
#