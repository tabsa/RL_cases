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
from functions import *
import sys

#%% Main script
if len(sys.argv) != 4:
	print "Usage: python train.py [stock] [window] [episodes]"
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in xrange(episode_count + 1):
	print "Episode " + str(e) + "/" + str(episode_count)
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in xrange(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print "Buy: " + formatPrice(data[t])

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print "Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price)

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print "--------------------------------"
			print "Total Profit: " + formatPrice(total_profit)
			print "--------------------------------"

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
