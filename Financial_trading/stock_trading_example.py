## Deep Q-learning for Stock trading problem
# Description missing (see Carpole_example - objective, state, action, env definition)
#
# The Q-function (state, action) is represented by the Q-table with rows as states and columns as actions.
# We can then represent the sequence of states+action (t, t+1, t+2,...,t+n) with reward function and transition probability P[s(t) --> s(t+1)].
# Since we have a POMDP (Partially Observable Markov Decision Process), the Q-table is approximated with learning process (called Q-learning).
# We use Deep Neural Network (NN) to implement the Q-learning, basically the NN learns the reward function.
# This way, Q-table is updated when the NN is learning the Q-function over the episodes in our RL framework.

# Import packages
from q_learning_agent import trader_agent
import numpy as np
import math

#%% Functions of the main scrip
# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])

#%% Main script
stock_name = 'FB_train'
window_size = 10
episode_count = 200

agent = trader_agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
	print(f'Episode: {e}/{episode_count}')
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print(f'Buy: {formatPrice(data[t]}')

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print(f'Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}')

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print('--------------------------------')
			print(f'Total Profit: {formatPrice(total_profit)}')
			print('--------------------------------')

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save(f'models/model_ep{e}')
