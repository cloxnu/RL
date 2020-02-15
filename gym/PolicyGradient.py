import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt


class PolicyGradient:

    def __init__(
        self, 
        n_actions,
        n_features,
        learning_rate=0.01, 
        reward_decay=0.95, 
        output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def _build_net(self):
        input = layers.Input(shape=(self.n_features,))
        x = layers.Dense(10, activation="tanh")(input)
        self.all_act = layers.Dense(self.n_actions)(x)

        loss = losses.sparse_categorical_crossentropy()
        rmsprop = optimizers.RMSprop(lr=self.learning_rate)

        self.model = models.Model(input, self.all_act)
        self.model.compile(loss=loss, optimizer=rmsprop, metrics=['accuracy'])

    def choose_action(self, observation):
        self.tf_obs = 
        prob_weights =

