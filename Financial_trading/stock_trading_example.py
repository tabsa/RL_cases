
# Import packages
# Typical pkg
import numpy as np
import random as rnd
from collections import deque
# ML framework - Keras API (neural networks)
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

# Create class trading agent
class trader_agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # Normalized previous days
        self.action_size = 3  # 3 actions - Buy, Sell, Hold
        self.memory = deque(maxlen=1000)
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