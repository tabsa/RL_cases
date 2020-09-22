
# Import packages
# Typical pkg
import numpy as np
import random as rnd
from collections import deque # Double queue that works similar as list, and with faster time when using append
# Deque has a complexity of O(1) while list has complexity O(n)

# ML framework - Keras API (neural networks)
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
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
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def _model(self): # NN model for the Q-learning
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu")) # 1st-layer (input layer, n-days of the state)
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear")) # end-layer (output layer, n-actions in this example is 3)
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def act(self, state): # Select the next action t+1
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return rnd.randrange(self.action_size)

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