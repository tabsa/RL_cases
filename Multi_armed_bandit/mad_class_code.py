## Class for the Multi-Armed Bandit problem
# class MAD_env - Defines the environment of the MAD problem
# class Agent - Defines the agent interacting with the MAD_env

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta

#%% Build the Environment
class MAD_env:
    def __init__(self, variants, var_revenue, no_trials, variance=False):  # Define parameters of the environment
        self.variants = variants  # Array of var_n, each n represent an option (slot machine n) to choose from
        if variance:  # Gaussian distribution of the revenue per variant_n in case we only have the avg revenue
            self.var_revenue = np.clip(var_revenue + np.random.normal(0, 0.04, size=len(variants)), 0, .2)
        else:
            self.var_revenue = var_revenue  # Assign the array of revenues per variant_n
        self.no_trials = no_trials  # 'Time-step' in the RL framework, duration of each episode [1,...,no_trials]
        self.env_simulation = 0  # Flag indicating the simulation is completed
        self.action_size = len(variants)  # Size of the action space - no of variants n
        self.env_size = (self.action_size, self.no_trials)

    def run(self, agent):  # Run the MAD_environment
        # Run the simulation, environment response, to the agent action per time_t [1,...,no_trials]
        for t in range(self.no_trials):  # per time_t
            # Agent makes action in time_t - action_t
            action_t = agent.action()  # Calls action function of agent_class
            # Environment responds in time_t - state_t, reward_t
            reward_t = np.random.binomial(1, p=self.var_revenue[action_t])  # Reward is a binomial (0,1) distribution, (cont.)...
            # probability is defined by the expected revenue per var_n (each var_n has a exp_rev that represents the prob of success)
            # Represents the success of a slot machine 0-losing and 1-winning

            # Agent receives state_t and reward_t
            agent.reward = reward_t  # This case the state_t info is irrelevant
            # Agent updates the strategy to time_t+1
            agent.update()  # Calls function that updates agent info for next time_t+1
            # Stores reward over time_t. We can get the total reward for the simulation.
            agent.total_reward += reward_t  # This way, we place the class MAD_env to be used inside a 'Training strategy' of RL_agent with for-loop i per no_epi: Agent_class(i), MAD_env(i)

        agent.collect_data()  # Function that stores the info, associated to agent_class
        # Return of this function
        self.env_simulation = 1
        return self.env_simulation


#%% Build the agent
class Agent:
    def __init__(self, env, policy_opt, no_sample=None, time_learning=None, e_greedy=0.05):
        # Parameters - Agent info
        self.policy_opt = policy_opt  # String with the policy strategy
        self.no_sample = no_sample  # Number of samples for the simulation
        self.time_learning = time_learning  # Number of time_t for learning, e.g. 1000 time_steps (trials) given for learning
        self.e_greedy = e_greedy  # Probability for exploring (epsilon_greedy)
        self.e_exploit = 1 - e_greedy  # Probability for exploiting (1 - epsilon_greedy)
        self.ep = np.random.uniform(0, 1, size=env.no_trials)  # FIGURE IT OUT
        self.reward = 0
        self.total_reward = 0
        self.action_choice = 0  # Action decision to be used every time_t
        self.id_t = 0  # Index of time_t
        # Parameters - Environment info
        self.env = env  # Call the MAD_env class we build above, to associate with the MAD environment
        self.sim_shape = (env.action_size, no_sample)  # Full shape of the simulation - action_size x no_episodes (no_samples)
        self.no_trials = env.no_trials  # Duration of each episode [1,...,env.no_trials]
        self.var_revenue = env.var_revenue  # Same as env.var_revenue

        # Initialize arrays to store info over time_t (no_trials) and action (variants_n) dimensions
        # Store info over time_t (action_t, state_t, reward_t)
        self.action_t = np.zeros(env.no_trials)
        self.reward_t = np.zeros(env.no_trials)
        self.theta_t = np.zeros(env.no_trials) # Cumulative probability of success per time_t
        self.regret_t = np.zeros(env.no_trials) # Opportunity cost per time _t
        self.theta_regret_t = np.zeros(env.no_trials) # Probability of the opportunity cost per time_t
        # Store info over action space [1,...,action_size], REMEMBER action_size is equal to the number of slot_machines
        self.a = np.ones(env.action_size) # Array with cumulative reward per time_t for each var_n (slot_machine)
        self.b = np.ones(env.action_size) # Array with the count of times var_n was chosen
        self.var_theta = np.ones(env.action_size)  # Cumulative probability of success (Prob = sum(reward_t) / sum(no_times_t)) per var_n (action) that is updated over time_t
        self.data = None  # DataFrame storing the final info of each episode simulation

    def collect_data(self):  # Function to manipulate data into DataFrames
        self.data = pd.DataFrame(dict(action=self.action_t, reward=self.reward_t, regret=self.regret_t))

    def plot_action_choice(self, plt_colmap, title):
        plt.figure(figsize=(10,7))
        trials = np.arange(0, self.no_trials)
        plt.scatter(trials, self.action_t, cmap=plt_colmap, c=self.action_t, marker='.', alpha=1)
        plt.title(title, fontsize=16)
        plt.xlabel('no trials #', fontsize=16)
        plt.ylabel('Machine', fontsize=16)
        plt.yticks(list(range(self.env.action_size)))
        plt.colorbar()
        plt.show()

    def action(self):  # Function to make the action of the agent over time_t
        if self.policy_opt == 'Random_policy':
            # Algorithm with random choice
            self.action_choice = self.rand_policy()
        elif self.policy_opt == 'e-greedy_policy':
            # Algorithm with e-greedy approach
            self.action_choice = self.eGreedy_policy()
        elif self.policy_opt == 'Thompson_Sampler_policy':
            # Algorithm with Thompson Sampler approach
            self.action_choice = self.tpSampler_policy()
        # Return the result
        return self.action_choice

    def update(self): # Function to update the arrays over time_t and var_n
        # Update the Cumulative probability for the var_n. It is updated everytime var_n is selected by action_t
        self.a[self.action_choice] += self.reward
        # self.b counts the no of times var_n was selected for self.policy_opt --> 'Random' and 'e-Greedy'
        # When self.policy_opt --> 'Thompson-Sampler', self.b increments 1 when var_n has reward = 0 (we miss). This way the cumulative_reward (self.a) is spreaded on the Beta Bernoulli distribution
        # It is like the ratio of self.a/self.b drops everytime we miss revenue with var_n (machine). Increase the change of another var_n being selected later on
        self.b[self.action_choice] += 1 - self.reward if self.policy_opt == 'Thompson_Sampler_policy' else 1
        # Update the Prob of all variants_n (slot machines) of the action space [1,...,self.env.action_size]
        self.var_theta = self.a / self.b

        # Calculate for time_t the Cumulative probability of regret (opportunity cost)
        if self.policy_opt == 'Random_policy':
            # As random policy, the opportunity cost (theta) per time_t -- is the difference of the Max-Cum-Prob and Cum-Prob of var_n gets over time
            self.theta_regret_t[self.id_t] = np.max(self.var_theta) - self.var_theta[self.action_choice]
        elif self.policy_opt == 'e-greedy_policy' or self.policy_opt == 'Thompson_Sampler_policy':
            # As e-greedy, the opportunity cost (theta) depends on NOT exploiting others 'non-seen' (less prob) var_n then var_n[self.action_choice]
            # (1-e-greedy) of time_t the Agent will select the var_n with the highest self.var_theta, the regret comes on NOT exploring other action_options
            self.theta_t[self.id_t] = self.var_theta[self.action_choice]
            self.theta_regret_t[self.id_t] = np.max(self.theta_t) - self.var_theta[self.action_choice]

        # Update the arrays over time_t
        self.action_t[self.id_t] = self.action_choice
        self.reward_t[self.id_t] = self.reward # Reward obtained from env.run()
        self.id_t += 1 # time_t + 1

    def rand_policy(self): # Implements the random choice algorithm
        return np.random.choice(self.env.variants)

    def eGreedy_policy(self): # Implements the e-greedy algorithm to explore_exploit
        # e_greedy indicates the Percentage of time used to explore the action_space (taking random choices)
        # If id_t < time_learning: Random_choice, Else: action = var_n[highest var_theta] (Highest probability of success)
        aux = np.random.choice(self.env.variants) if self.id_t < self.time_learning else np.argmax(self.var_theta)

        # Step that controls IF we choice Exploration or Exploitation
        # If ep_prob[id_t] < 1 - e_greedy: aux (best action), Else: Random_choice
        aux = aux if self.ep[self.id_t] <= self.e_exploit else np.random.choice(self.env.variants)
        # Return the result of the function
        return aux

    def tpSampler_policy(self):
        # Thompson Sampler policy
        # Sampling a probability based on distribution (We assumed Beta Bernoulli) based on cumulative reward (self.a) and no. selected-times (self.b) per var_n
        # The Beta-dist captures the uncertainty per var_n that is changing over time_t...
        # Per time_t indicates the probablity of getting reward per var_n, it is dynamic over trials until the Beta-dist goes towards the var_n with highest reward
        # Rule of thumb - largest mean and smallest std.dev results in greater prob of being selected...
        # ...var_n with low revenue is expected a wider distribution with small mean and large std.dev resulting in high uncertainty for future action choice
        self.var_theta = np.random.beta(self.a, self.b)
        # Select the var_n (machine) with highest expected revenue (based on cumulative probability over time_t)
        return self.env.variants[np.argmax(self.var_theta)]
