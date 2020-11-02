## Deep Q-learning for Stock trading problem
# Description missing (see Carpole_example - objective, state, action, env definition)
#
# The Q-function (state, action) is represented by the Q-table with rows as states and columns as actions.
# We can then represent the sequence of states+action (t, t+1, t+2,...,t+n) with reward function and transition probability P[s(t) --> s(t+1)].
# Since we have a POMDP (Partially Observable Markov Decision Process), the Q-table is approximated with learning process (called Q-learning).
# We use Deep Neural Network (NN) to implement the Q-learning, basically the NN learns the reward function.
# This way, Q-table is updated when the NN is learning the Q-function over the episodes in our RL framework.

# Import packages
# Typical pkg
import numpy as np
import random as rnd
from collections import deque # Double queue that works similar as list, and with faster time when using append
# Deque has a complexity of O(1) while list has complexity O(n)

# ML framework - Keras API (neural networks)
from tensorflow import keras
from keras.models import Sequential # NN model architecture. Sequential defines a Multi-layer sequence connecting Input with Output. Most common architecture used
from keras.models import load_model
from keras.layers import Dense # Defines the type of connection in a layer (input_neuron <-> output_neuron). It is a regular connection out = acti(wei * in + bias), and most commonly used layer.
from keras.optimizers import Adam

# Create class trader agent
class trader_agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # Normalized previous days
        self.action_size = 3  # 3 actions - Buy, Sell, Hold
        self.memory = deque(maxlen=1000) # Creates an empty queue with max lenght of 1000, it will work as memory of the NN
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        # Parameters of the Q-learning
        self.gamma = 0.95 # Q-learning rate, rate of learning the Q-table
        self.epsilon = 1.0
        self.epsilon_min = 0.01 # min exploration prob (epsilon)
        self.epsilon_decay = 0.995 # Decay of exploration probability (epsilon)
        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def _model(self): # Build the NN model for the Q-learning -- Sequential Multi-layer NN architecture
        model = Sequential() # Multi-layer with 4 layers connected one-by-one
        # 1st-layer (input = n-days of the state; output = 64 elements). Dictates the weight shape to connect input <-> output
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu")) # units - no of output elements, input_dim - no of input el, activation - Function (RelU most common one)
        model.add(Dense(units=32, activation="relu")) # 2nd-layer with input = 64 neurons <-> output = 32 neurons. REMEMBER: 64 el is because output of 1st layer is 64 (units=64). Weights.shape(32, 64)
        model.add(Dense(units=8, activation="relu")) # 3rd-layer with input = 32 <-> output = 8. Weights.shape(8, 32)
        model.add(Dense(self.action_size, activation="linear")) # end-layer with input = 8 <-> output = 3 (action - Buy, Sell, Hold). Weights.shape(3, 8)
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def act(self, state): # Select the next action t+1
        # Exploration phase - is_eval = True and rand <= epsilon (learning probability)
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return rnd.randrange(self.action_size) # Takes random action to fill the Q-table
        # Exploitation phase - uses NN to predict the next action_t+1 based on the state_t
        options = self.model.predict(state) # NN to predict the action t+1
        return np.argmax(options[0]) # Returns the index of the max value of option.array

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay