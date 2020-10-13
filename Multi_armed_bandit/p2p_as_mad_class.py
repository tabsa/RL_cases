## Class to implement the P2P market as MAD problem
# Version 2 of the code 'p2p_class_code'
# Here we consider an episodic environment - as the carpole example
# Agent has fixed no.trials to achieve the desired energy target
# The episode is done when the energy target is reached
#   - we get a positive episode, meaning that total_reward is <= 0 (the closest to zero the best)
#   - Otherwise the episode fails and we reach a total_reward = -inf
#
# Policy that maximizes the total cumulative rewward - Still to DEFINE
# Specify the rward function
# The episode ends when:
# 1. Agent reaches the energy target
# 2. Episode reaches the last trial
#
# As class of the code we have the following:
# class trading_env - Defines the environment of the decentralized energy trading
# class trading_offers - Defines the characteristics of the offers during the all episode, j index for offer
# class trading_agent - Defines the RL_agent and prosumer that interacts with the trading_env

#%% Import packages
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta

#%% Build the Environment
class trading_env:
    def __init__(self, no_offers, no_trials, offer_file, sample_type=None, sample_seed=None, no_preferences=2):  # Define parameters of the environment
        # time-space parameters
        self.no_trials = no_trials  # Time-step of the energy P2P market in the RL framework, duration of each episode [1,...,no_trials]
        self.env_simulation = 0  # Flag indicating the simulation is completed
        self.end_trial = 0
        self.env_size = (no_offers, no_trials)
        # Offers parameters
        self.no_offers = no_offers
        self.offers_id = np.arange(no_offers)
        self.offer_file = offer_file
        self.sample_type = sample_type # Indicates if the sampling is as external input, or generated by the env
        self.sample_seed = sample_seed # Sample seed, here we just consider as trading_agent.energy_target
        self.offers_info = np.zeros((no_offers, 3))
        self.preference = np.zeros((no_offers, no_preferences))
        self.sigma_n = np.zeros((no_offers, no_trials))

    def run(self, trading_agent):  # Run the trading_env
        # Run the simulation, environment response, to the agent action per time_t [1,...,no_trials]
        if self.sample_type == 'External_sample':
            trading_agent.energy_target = self.sample_seed
        else:
            trading_agent.energy_target = trading_agent.profile_sampling()  # Energy value as Target, where market.offers have reach from [1,...,no_trials]
        target_T = trading_agent.energy_target
        self.offers_sample(target_T)
        for n in range(self.no_trials):  # per trial_n
            # Agent makes action in trial_n - action_n
            action_n = trading_agent.action()  # Calls action function of agent_class
            # Environment responds in time_t - state_t, reward_t
            reward_n = np.random.binomial(1, p=self.offers_info[action_n,2])  # Reward is a binomial (0,1) distribution, (cont.)...
            # probability is defined by the expected revenue per var_n (each var_n has a exp_rev that represents the prob of success)
            # Represents the success of a slot machine 0-losing and 1-winning

            # Agent receives state_t and reward_t
            trading_agent.reward = reward_n  # This case the state_t info is irrelevant
            trading_agent.update()  # Calls function that updates agent info for next time_t+1
            # Stores reward over time_t. We can get the total reward for the simulation.
            trading_agent.total_reward += reward_n  # This way, we place the class MAD_env to be used inside a 'Training strategy' of RL_agent with for-loop i per no_epi: Agent_class(i), MAD_env(i)

            if target_T == trading_agent.state_n[n]:
                self.env_simulation = 1
                self.end_trial = n
                break

        # trading_agent.collect_data()  # Function that stores the info, associated to agent_class
        # Return of this function
        self.env_simulation = 1
        return self.env_simulation

    def offers_sample(self, target): # Function to sample offers [1,...,j] per trial_n
        # Energy offering sampling - Depends on the target (energy from trading_agent)
        ## Run the Sampling method - Input file, Monte Carlo, real-time, etc
        if self.sample_type == None or self.sample_type == 'External_sample':
            # Read the csv-file with all offer info
            offer_data = pd.read_csv(self.offer_file)  # Read the csv-file with all offer info
            self.offers_info = offer_data[['energy', 'price', 'sigma']].values
            self.preference = offer_data[['distance', 'co2']].values  # Preference (Distance and CO2)
        elif self.sample_type == 'Monte-Carlo':
            # Monte-Carlo sampling - Assuming we change offer_energy per trial_n (time)
            offer_rnd_sample = np.random.sample(size=self.no_agents) # Random sample distribution per agent_j
            offer_rnd_sample = offer_rnd_sample/sum(offer_rnd_sample) # Get the ratio to distribute as Pro-rata
            self.energy = offer_rnd_sample*target
            # Price offering sampling - Assuming we have variance price per trial_n (time)
            for j in range(self.no_agents): # For-loop per agent_j
                j_mean = self.price_bounds[j,0] # Mean price per agent j
                j_std = self.price_bounds[j, 1] # Std dev price per agent j
                self.price[j,:] = np.random.normal(loc=j_mean, scale=j_std, size=self.price.shape[1])

#%% Build the RL_agent and represented prosumer_i
class trading_agent: # Class of RL_agent to represent the prosumer i
    def __init__(self, env, policy_opt, prosumer_type=None, no_sample=None, time_learning=None, e_greedy=0.05):
        # Parameters - Environment info
        self.env = env  # Call the trading_env class we build above
        self.sim_shape = (env.no_offers, no_sample)  # Full shape of the simulation - action_size x no_episodes (no_samples)
        # Parameters - RL_Agent info
        self.prosumer_type = prosumer_type
        self.policy_opt = policy_opt  # String with the policy strategy
        self.no_sample = no_sample  # Number of samples for the simulation
        self.time_learning = time_learning  # Number of time_t for learning, e.g. 1000 time_steps (trials) given for learning
        self.e_greedy = e_greedy  # Probability for exploring (epsilon_greedy)
        self.e_exploit = 1 - e_greedy  # Probability for exploiting (1 - epsilon_greedy)
        self.ep = np.random.uniform(0, 1, size=env.no_trials)  # FIGURE IT OUT
        self.reward = 0
        self.total_reward = 0
        self.action_choice = 0  # Action decision to be used every trial_n
        self.id_n = 0  # Index of trial_n
        # Store info over trial_n (action_t, state_t, reward_t)
        self.state_n = np.zeros(env.no_trials) # Settle energy for each trial_n
        #self.accept_energy = np.zeros(env.no_trials)  # Accepted energy for every state_t per trial_n
        self.action_n = np.zeros((2, env.no_trials)) # Arrays No.Agents x No.Trials [0, 1]
        self.reward_n = np.zeros(env.no_trials)
        self.theta_n = np.zeros(env.no_trials) # Cumulative probability of success per trial_n
        self.regret_n = np.zeros(env.no_trials) # Opportunity cost per time_t
        self.theta_regret_n = np.zeros(env.no_trials) # Probability of the opportunity cost per trial_n
        # Store info over action space [1,...,action_size], REMEMBER action_size is equal to the number of slot_machines
        self.a = np.ones(env.no_offers) # Array with cumulative reward per time_t for each var_n (slot_machine)
        self.b = np.ones(env.no_offers) # Array with the count of times var_n was chosen
        self.var_theta = np.ones(env.no_offers)  # Cumulative probability of success (Prob = sum(reward_t) / sum(no_times_t)) per var_n (action) that is updated over time_t
        self.data = None  # DataFrame storing the final info of each episode simulation
        # Parameters - Prosumer info
        self.energy_target_bounds = np.array([2.05, 15]) # Max and Min energy per target_time. Energy_target <0 Consumer; >0 Producer

    def collect_data(self):  # Function to manipulate data into DataFrames
        self.data = pd.DataFrame(dict(action=self.action_t, reward=self.reward_t, regret=self.regret_t))

    def profile_sampling(self): # Function to generate/collect the energy profile of prosumer_i
        ## Run a Uniform sampling - Single value but it can be a time-series
        return np.random.uniform(low=self.energy_target_bounds[0], high=self.energy_target_bounds[1])

    def plot_action_choice(self, plt_colmap, title):
        plt.figure(figsize=(10,7))
        trials = np.arange(0, self.env.no_trials)
        # Subplot 1
        plt.subplot(211)  # 2 rows and 1 column
        plt.scatter(trials, self.action_n[0,:], cmap=plt_colmap, c=self.action_n[0,:], marker='.', alpha=1)
        plt.title(title, fontsize=16)
        plt.xlabel('no trials #', fontsize=16)
        plt.ylabel('Offers', fontsize=16)
        plt.yticks(list(range(self.env.no_offers)))
        plt.colorbar()
        # Subplot 2
        plt.subplot(212)
        plt.bar(trials, self.action_n[1,:])
        #plt.bar(trials, self.state_n)
        plt.title('Energy traded per trial', fontsize=16)
        plt.xlabel('no Trials #', fontsize=16)
        plt.ylabel('Energy', fontsize=16)
        #plt.legend()
        plt.show()

    def update(self): # Function to update the arrays over time_t and var_n
        # Update the state, action and reward per trial_n
        state_n_1 = self.state_n[self.id_n - 1] if self.id_n > 0 else 0
        offer_id = self.action_choice
        energy_n = self.env.offers_info[offer_id, 0] if self.reward == 1 else 0
        if state_n_1 + energy_n > self.energy_target:
            energy_n = self.energy_target - state_n_1
        self.action_n[0, self.id_n] = offer_id
        self.action_n[1, self.id_n] = energy_n
        self.reward_n[self.id_n] = self.reward  # Reward obtained from env.run()
        self.state_n[self.id_n] = state_n_1 + self.action_n[1, self.id_n]

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
            self.theta_regret_n[self.id_n] = np.max(self.var_theta) - self.var_theta[self.action_choice]
        elif self.policy_opt == 'e-greedy_policy':
            # As e-greedy, the opportunity cost (theta) depends on NOT exploiting others 'non-seen' (less prob) var_n then var_n[self.action_choice]
            # (1-e-greedy) of time_t the Agent will select the var_n with the highest self.var_theta, the regret comes on NOT exploring other action_options
            self.theta_n[self.id_n] = self.var_theta[self.action_choice]
            self.theta_regret_n[self.id_n] = np.max(self.theta_n) - self.var_theta[self.action_choice]

        # Next state_n+1
        self.id_n += 1 # time_t + 1

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

    def rand_policy(self): # Implements the random choice algorithm
        return np.random.choice(self.env.offers_id)

    def eGreedy_policy(self): # Implements the e-greedy algorithm to explore_exploit
        # e_greedy indicates the Percentage of time used to explore the action_space (taking random choices)
        # If id_t < time_learning: Random_choice, Else: action = var_n[highest var_theta] (Highest probability of success)
        aux = np.random.choice(self.env.offers_id) if self.id_n < self.time_learning else np.argmax(self.var_theta)

        # Step that controls IF we choice Exploration or Exploitation
        # If ep_prob[id_t] < 1 - e_greedy: aux (best action), Else: Random_choice
        aux = aux if self.ep[self.id_n] <= self.e_exploit else np.random.choice(self.env.offers_id)
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
        return self.env.offers_id[np.argmax(self.var_theta)]
