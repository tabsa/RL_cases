
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta

sns.set_style("whitegrid")

# Build the Environment
class MAD_env:
    def __init__(self, variants, var_revenue, no_trials, variance=False): # Define parameters of the environment
        self.variants = variants # Array of var_n, each n represent an option (slot machine n) to choose from
        if variance: # Gaussian distribution of the revenue per variant_n in case we only have the avg revenue
            self.var_revenue = np.clip(var_revenue + np.random.normal(0, 0.04, size=len(variants)), 0, .2)
        else:
            self.var_revenue = var_revenue # Assign the array of revenues per variant_n
        self.no_trials = no_trials # 'Time-step' in the RL framework, duration of each episode [1,...,no_trials]
        self.total_reward = 0 # Initialize the reward function per time t
        self.action_size = len(variants) # Size of the action space - no of variants n
        self.env_size = (self.action_size, self.no_trials)

    def run(self, agent): # Run the MAD_environment
        # Run the simulation, environment response, to the agent action per time_t [1,...,no_trials]
        for t in range(self.no_trials): # Per time_t
            # Agent makes action in time_t - action_t
            action_t = agent.action() # Calls action function of agent_class
            # Environment responds in time_t - state_t, reward_t
            reward_t = np.random.binomial(1, p=self.var_revenue[action_t]) # Reward is a binomial (0,1) distribution, probability is defined by the expected revenue per var_n (each var_n has a exp_rev that represents the prob of success)
            # Represents the success of a slot machine 0-losing and 1-winning

            # Agent receives state_t and reward_t
            agent.reward = reward_t # This case the state_t info is irrelevant
            # Agent updates the strategy to time_t+1
            agent.update() # Calls function that updates agent info for next time_t+1
            # Stores reward over time_t. We can get the total reward for the simulation.
            self.total_reward += reward_t # This way, we place the class MAD_env to be used inside a 'Training strategy' of RL_agent with for-loop i per no_epi: Agent_class(i), MAD_env(i)

        agent.collect_data() # Function that stores the info, associated to agent_class
        # Return of this function
        return self.total_reward

# Build the simulation
class Simulation:
    def __init__(self, env, no_epi=None, no_learning=None, e_greedy=0.05):
        # Parameters - Simulation
        self.no_epi = no_epi # Number of episodes for the simulation
        self.no_learning = no_learning # Type of learning
        self.e_greedy = e_greedy # Probability for exploring (epsilon_greedy)
        self.e_exploit = 1-e_greedy # Probability for exploiting (1 - epsilon_greedy)
        self.ep = np.random.uniform(0, 1, size=env.no_trials) #FIGURE IT OUT
        # Parameters - Environment
        self.env = env # Call the MAD_env class we build above, to associate with the MAD environment
        self.sim_shape = (env.action_size, no_epi) # Full shape of the simulation - action_size x no_episodes (no_samples)
        self.no_trials = env.no_trials # Duration of each episode [1,...,env.no_trials]

    def collect_data(self): # Function to manipulate data into DataFrames
        self.data = pd.DataFrame(dict(ad=self.ad_i, reward=self.r_i, regret=self.regret_i))
