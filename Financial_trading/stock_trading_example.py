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
import pandas as pd

#%% Functions of the main scrip
# Printing the formatted stock price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# Returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	filename = key + '.csv'
	info = pd.read_csv(filename, index_col='Date')
	length = info.shape[0]
	data = info['Close'].to_numpy()
	return data, info, length

# Returns the n-day state representation ending at time t
def getState(data, t, n):
	d = t - n
	#block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	if d >= 0:
		block = data[d:t+1]
	else:
		aux = data[0] * np.ones(-d) # Replicate t0 state
		block = np.concatenate([aux, data[0:t+1]], axis=0)
	block = np.diff(block)
	res = 1 / (1 + np.exp(-block)) # Sigmoid function to normalize
	return res # State_t

#%% Main script
if __name__ == '__main__':
	# Stock info
	stock_name = 'FB'
	window_size = 5
	episode_count = 200
	stock_price, df_stock, stock_vol = getStockDataVec(stock_name) # Extract the important info
	# Call trader class
	agent = trader_agent(window_size)
	batch_size = 32 # Define the batch memory for the exp Replay update

	for e in range(episode_count + 1):
		print(f'Episode: {e}/{episode_count}')
		state = getState(stock_price, 0, window_size)
		total_profit = 0
		agent.inventory = []
		# For-loop of time
		for t in range(stock_vol):
			action = agent.act(state) # Take action
			# Get next state for t+1
			next_state = getState(stock_price, t+1, window_size)

			reward = 0
			if action == 1: # Buy
				agent.inventory.append(stock_price[t])
				print(f'Date {df_stock.index[t]} | Buy: {formatPrice(stock_price[t])}')

			elif action == 2 and len(agent.inventory) > 0: # Sell
				bought_price = agent.inventory.pop(0)
				profit_t = stock_price[t] - bought_price
				reward = profit_t if profit_t >= 0 else 0
				total_profit += profit_t
				print(f'Date {df_stock.index[t]} | Sell: {formatPrice(stock_price[t])} | Profit: {formatPrice(profit_t)}')

			done = True if t == stock_vol - 1 else False
			# Store in memory for the exp Replay update
			agent.memory.append((state, action, reward, next_state, done))
			state = next_state # move to state_t+1

			if len(agent.memory) > batch_size: # Activate exp Replay update
				agent.expReplay(batch_size)

		# Print result when epi e is done
		print('--------------------------------')
		print(f'Total Profit: {formatPrice(total_profit)}')
		print('--------------------------------')

		if e % 10 == 0:
			agent.model.save(f'models/model_ep{e}')
