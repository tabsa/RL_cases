## RL agent for the Multi-Armed Bandit (MAD) problem
# Environment is composed by different slot machines (variant_n) each with expected revenue (machine_payout)
# The objective is to find the policy that maximizes the reward of the user for playing with all machines
# RL agent to learning the slot machine (var_n) that returns the best
# A reward of +1 is provided for every timestep that the selected machine (var_n) returns a revenue.
# Environment process per time_t:
#  1 - RL agent selects an action_t (select machine var_n)
#  2 - Env observes if var_n returns a reward (payout) - Machine revenue is simulated as Binomial distribution
#  3 - RL agent receives the observed state_t and reward_t (0/1)

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import Class and functions
from mad_class_code import MAD_env, Agent # Class of the MAD example
from plot_class import plot_action_choice, plot_reward_per_episode, plot_regret_prob

#%% Class section
class dashboard:
    def __init__(self, path):
        self.path = path
        self.data = self.get_data()
        self.epi = self.data['simulation']['episodes']
        self.steps = self.data['simulation']['trials']
        self.no_agents = self.data['agents']['no']
        self.agents_policy = self.data['agents']['id']

    @st.cache()
    def get_data(self):
        return pkl.load(open(self.path, 'rb'))

    def run_app(self):
        self.head_title()
        self.sec_reward()

    def head_title(self):
        st.title('Reinforcement learning models for Energy P2P markets')
        st.write('Evaluate the results of the Multi Armed Bandit application.')
        st.write('- Simulation done for **%s** episodes with **%s** steps per episode' %(self.epi, self.steps))
        st.write('- Using %s policies per episode: %s' %(self.no_agents, self.agents_policy))

    def sec_reward(self):
        st.header('Total reward analysis')

#%% Main script
# Input of the Multi-Armed Bandit (MAD) problem
slot_machi = np.arange(10) # Id of the slot machines
machi_payout = np.array([0.023, 0.03, 0.029, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.11])
labels = ["V" + str(i) + (str(p)) for i, p in zip(slot_machi, machi_payout)]
assert len(slot_machi) == len(machi_payout), 'Size of Var_n is NOT equal to Revenue_var_n'
# Parameters and Class of the MAD problem
n_tests = 10
n_trials = 10000
n_agents = 3
rd_score = np.zeros((n_agents,n_tests)) # Reward score per simulation
rd_score_trial = np.zeros((n_agents, n_tests, n_trials))
regret_score = np.zeros((n_agents, n_tests, n_trials)) # Regret probability per (epi, agent, trial)
# Assign the environment
env = MAD_env(slot_machi, machi_payout, n_trials) # Env class
# For-loop i from [1,...,n_tests]
for i in range(n_tests):
    print(f'Simulation {i+1}')
    # Agent using the Random policy
    rd_agent = Agent(env, 'Random_policy')
    env.run(rd_agent)
    rd_score[0,i] = rd_agent.total_reward
    rd_score_trial[0, i, :] = rd_agent.reward_t
    regret_score[0, i, :] = rd_agent.theta_regret_t
    # Agent using the e-Greedy policy
    eg_agent = Agent(env, 'e-greedy_policy', time_learning=500, e_greedy=0.1)
    env.run(eg_agent)
    rd_score[1,i] = eg_agent.total_reward
    rd_score_trial[1, i, :] = eg_agent.reward_t
    regret_score[1, i, :] = eg_agent.theta_regret_t
    # Agent using the Thompson-Sampler policy
    ts_agent = Agent(env, 'Thompson_Sampler_policy')
    env.run(ts_agent)
    rd_score[2,i] = ts_agent.total_reward
    rd_score_trial[2, i, :] = ts_agent.reward_t
    regret_score[2, i, :] = ts_agent.theta_regret_t

print(f'All {n_tests} Simulations Done')
# Plot of regret probability comparing the 3 policies here
epi_sel = n_tests
label = ['random', 'e-greedy', 'thompson']
ax_label = ['no Trials #', 'Regret probability']
for i in range(epi_sel):
    plt_title = [f'Cumulative regret across steps of episode {i}',
                 f'Regret per step n of episode {i}']
    plot_regret_prob(regret_score, i, label, ax_label, plt_title)

# Plot comparing the 3 policies implemented to the MAD problem
marker = ['o', '*', '^']
ax_label = ['Simulation #', 'Total Reward']
plot_reward_per_episode(rd_score, label, marker, ax_label, f'Comparison of policies in MAD problem across {n_tests} simulations')
