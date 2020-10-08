## Class for the energy P2P market example
# class p2p_env - Defines the environment of the energy P2P market
# class Agents_characteristics - Defines the characteristics of each agents that prosumer_i trades energy from/to
# class prosumer_i - Defines the prosumer (agent) that interacts with the p2p_env

#%% Import packages
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta

#%% Build the Environment
class p2p_env:
    def __init__(self, agents, no_time, target_step):  # Define parameters of the environment
        self.agents = agents  #  Class of p2p_agent_n, each n represent a prosumer j to trade energy
        self.no_time = no_time  # Time-step of the energy P2P market in the RL framework, duration of each episode [1,...,no_trials]
        self.target_step = target_step # Last time-step for the Energy target. Example assumes per hour we have a target to reach
        self.target_time_array = np.arange(start=target_step-1, stop=no_time, step=target_step) # Array of ALL last time-steps, e.g. [11, 23, 35,...,1199] if no_time=1200 and target_step=12

        self.env_simulation = 0  # Flag indicating the simulation is completed
        self.action_size = agents.no_agents  # Size of the action space - no of prosumer j
        self.env_size = (self.action_size, self.no_time)

    def set_simulation(self, RL_agent):
        self.agents.read_input()
        target_T = RL_agent.profile_sampling()  # Energy value for each last time-step (target_step), IMPORTANT to match the offers from the env.agents to the target_T
        self.agents.time_series_update(target_T, self.target_step)

    def run(self, RL_agent):  # Run the MAD_environment
        # Start by setting the simulatio - EXPLAIN BETTER
        self.set_simulation(RL_agent)
        # Run the simulation, environment response, to the agent action per time_t [1,...,no_trials]
        for t in range(self.no_time):  # per time_t
            # Agent makes action in time_t - action_t
            action_t = RL_agent.action()  # Calls action function of agent_class
            # Environment responds in time_t - state_t, reward_t
            reward_t = np.random.binomial(1, p=self.var_revenue[action_t])  # Reward is a binomial (0,1) distribution, (cont.)...
            # probability is defined by the expected revenue per var_n (each var_n has a exp_rev that represents the prob of success)
            # Represents the success of a slot machine 0-losing and 1-winning

            # Agent receives state_t and reward_t
            RL_agent.reward = reward_t  # This case the state_t info is irrelevant
            # Agent updates the strategy to time_t+1
            RL_agent.update()  # Calls function that updates agent info for next time_t+1
            # Stores reward over time_t. We can get the total reward for the simulation.
            RL_agent.total_reward += reward_t  # This way, we place the class MAD_env to be used inside a 'Training strategy' of RL_agent with for-loop i per no_epi: Agent_class(i), MAD_env(i)

        RL_agent.collect_data()  # Function that stores the info, associated to agent_class
        # Return of this function
        self.env_simulation = 1
        return self.env_simulation

#%% Build Characteristics of the other Prosumers - j index [1,...,no_agents]
class market_agents:
    def __init__(self, no_agents, no_time, file, no_preferences=2): # Define the paramters of this class
        self.input_file = file
        self.no_agents = no_agents # No of agents
        self.agents_id = np.arange(no_agents) # Array with agents_j id [0,...,no_agents-1]
        self.no_preferences = no_preferences # No. of pref - Default is 2 (Distance, CO2)
        self.energy = np.zeros((self.no_agents, no_time)) # Energy offered by agent j to prosumer i: Energy^t_{i,j}
        self.price = np.zeros((self.no_agents, no_time)) # Perceived price of each pair (i,j): Lambda^t_{i,j}
        self.price_bounds = np.zeros((self.no_agents, 2)) # Max_price, Min_price per agent j
        self.preference = np.zeros((self.no_agents, self.no_preferences))
        self.sigma = np.zeros((self.no_agents, no_time)) # Sigma distribution of accepted offer pair (i,j) - Dictates if Energy^t_{i,j} is accepted or not
        self.sigma_ref = np.zeros(self.no_agents) # Reference for the sigma, from agent j perspective, each agent j expects to accept offers with sigma_ref[j] probability

    def read_input(self): # Read file
        ## Read the csv-file with all input information
        input_data = pd.read_csv(self.input_file)
        # Convert the DataFrame into arrays
        self.price_bounds = input_data[['max_price', 'min_price']].values
        self.sigma_ref = input_data['sigma'].values
        self.preference = input_data[['distance','co2']].values  # Convert pd.frame into

    def time_series_update(self, target, rep_step): # Function to generate/collect the time-series of each agent_j
        ## Run a Monte Carlo sampling - FUTURE it can be something else - Read input file, other type of sampling
        # Energy offering sampling - Depends on the target (energy from RL_agent)
        tg_size = len(target) # Indicates the size of the array, tg_size = no_time/rep_step
        ag_target = np.random.uniform(low=target, high=1.5*target, size=tg_size)
        target_repmat = np.matlib.repmat(ag_target, 1, rep_step).reshape(1, tg_size*rep_step, order='F')
        # Rep_mat replicates the target for each time_t of that target, e.g. target_[0] will be for target_rp[step0, step1,...stepTarget]
        # Example we have time=5min and rep_step=12, so the target[0] will be equal for 12*time
        # Random sample distribution per agent_j for each time_t
        ener_rnd_sample = np.random.sample(size=(self.no_agents, tg_size*rep_step))
        ener_rnd_sample = ener_rnd_sample/sum(ener_rnd_sample) # Get the ratio to distribute as Pro-rata
        self.energy = ener_rnd_sample*target_repmat
        # Price offering sampling
        for j in range(self.no_agents): # For-loop per agent_j
            j_mean = self.price_bounds[j,0] # Mean price per agent j
            j_std = self.price_bounds[j, 1] # Std dev price per agent j
            self.price[j,:] = np.random.normal(loc=j_mean, scale=j_std, size=self.price.shape[1])

#%% Build the RL_agent and represented prosumer_i
class p2p_RL_agent: # Class of RL_agent to represent the prosumer i
    def __init__(self, env, policy_opt, prosumer_type=None, no_sample=None, time_learning=None, e_greedy=0.05):
        # Parameters - RL_Agent info
        self.policy_opt = policy_opt  # String with the policy strategy
        self.no_sample = no_sample  # Number of samples for the simulation
        self.time_learning = time_learning  # Number of time_t for learning, e.g. 1000 time_steps (trials) given for learning
        self.e_greedy = e_greedy  # Probability for exploring (epsilon_greedy)
        self.e_exploit = 1 - e_greedy  # Probability for exploiting (1 - epsilon_greedy)
        self.ep = np.random.uniform(0, 1, size=env.no_time)  # FIGURE IT OUT
        self.reward = 0
        self.total_reward = 0
        self.action_choice = 0  # Action decision to be used every time_t
        self.id_t = 0  # Index of time_t
        # Store info over time_t (action_t, state_t, reward_t)
        self.action_t = np.zeros((env.action_size, env.no_time)) # Arrays No.Agents x No.Time [0, 1]
        self.reward_t = np.zeros(env.no_time)
        self.theta_t = np.zeros(env.no_time) # Cumulative probability of success per time_t
        self.regret_t = np.zeros(env.no_time) # Opportunity cost per time_t
        self.theta_regret_t = np.zeros(env.no_time) # Probability of the opportunity cost per time_t
        # Store info over action space [1,...,action_size], REMEMBER action_size is equal to the number of slot_machines
        self.a = np.ones(env.action_size) # Array with cumulative reward per time_t for each var_n (slot_machine)
        self.b = np.ones(env.action_size) # Array with the count of times var_n was chosen
        self.var_theta = np.ones(env.action_size)  # Cumulative probability of success (Prob = sum(reward_t) / sum(no_times_t)) per var_n (action) that is updated over time_t
        self.data = None  # DataFrame storing the final info of each episode simulation

        # Parameters - Environment info
        self.env = env  # Call the MAD_env class we build above, to associate with the MAD environment
        self.sim_shape = (env.action_size, no_sample)  # Full shape of the simulation - action_size x no_episodes (no_samples)
        self.var_revenue = env.var_revenue  # Same as env.var_revenue
        # Parameters - Prosumer info
        self.prosumer_type = prosumer_type
        self.select_energy = np.zeros((env.action_size, env.no_time)) # Selected energy for every action_t per time_t
        self.accept_energy = np.zeros((env.action_size, env.no_time)) # Accepted energy for every state_t per time_t
        self.energy_target_bounds = np.array([-0.05, -3]) # Max and Min energy per target_time. Energy_target <0 Consumer; >0 Producer
        self.energy_target = np.zeros(len(env.target_time_array)) # Array for Energy_target per target_time [1,...,env.target_time]

    def collect_data(self):  # Function to manipulate data into DataFrames
        self.data = pd.DataFrame(dict(action=self.action_t, reward=self.reward_t, regret=self.regret_t))

    def profile_sampling(self): # Function to generate/collect the energy profile of prosumer_i
        ## Run a Uniform sampling
        self.energy_target = np.random.uniform(low=self.energy_target_bounds[0], high=self.energy_target_bounds[1],
                                               size=len(self.energy_target))
        return self.energy_target
    # def plot_action_choice(self, plt_colmap, title):
    #     plt.figure(figsize=(10,7))
    #     trials = np.arange(0, self.no_trials)
    #     plt.scatter(trials, self.action_t, cmap=plt_colmap, c=self.action_t, marker='.', alpha=1)
    #     plt.title(title, fontsize=16)
    #     plt.xlabel('no trials #', fontsize=16)
    #     plt.ylabel('Machine', fontsize=16)
    #     plt.yticks(list(range(self.env.action_size)))
    #     plt.colorbar()
    #     plt.show()

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
        elif self.policy_opt == 'e-greedy_policy':
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
