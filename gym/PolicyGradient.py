import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
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
        

