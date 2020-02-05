import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt


class DQN:
    def __init__(self,
                 num_actions,
                 num_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 ):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else e_greedy

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, num_features * 2 + 2))

        self._build_net()

    def target_replace_op(self):
        v1 = self.model2.get_weights()
        self.model1.set_weights(v1)

    def _build_net(self):
        eval_input = layers.Input(shape=(self.num_features,))
        x = layers.Dense(64, activation='relu')(eval_input)
        x = layers.Dense(64, activation='relu')(x)
        self.q_eval = layers.Dense(self.num_actions)(x)

        target_inputs = layers.Input(shape=(self.num_features,))
        x = layers.Dense(64, activation='relu')(target_inputs)
        x = layers.Dense(64, activation='relu')(x)
        self.q_next = layers.Dense(self.num_actions)(x)

        self.model1 = models.Model(target_inputs, self.q_next)
        self.model2 = models.Model(eval_input, self.q_eval)
        rmsprop = optimizers.RMSprop(lr=self.learning_rate)
        self.model1.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.model1.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.num_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.model1.predict(batch_memory[:, -self.num_features:]), self.model2.predict(batch_memory[:, :self.num_features])
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.num_features].astype(int)
        reward = batch_memory[:, self.num_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.model_log = self.model2.fit(batch_memory[:, :self.num_features], q_target, epochs=10)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_greedy else self.epsilon_greedy
        self.learn_step_counter += 1

    def plot_cost(self):
        acc = self.model_log.history['accuracy']
        loss = self.model_log.history['loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(acc)), acc, label='Training Accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(loss)), loss, label='Training Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')

