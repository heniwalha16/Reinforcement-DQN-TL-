import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        """
        Initialize Q-learning agent.
        :param state_size: The size of the state space.
        :param action_size: The size of the action space.
        :param learning_rate: The learning rate for the Q-learning algorithm.
        :param discount_factor: The discount factor for future rewards.
        :param exploration_rate: Initial exploration rate.
        :param exploration_decay: Decay rate for the exploration.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        """
        Choose the action based on the current state.
        :param state: The current state.
        :return: action
        """
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the learning algorithm.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state.
        :param done: Whether the episode is finished.
        """
        q_update = reward
        if not done:
            q_update += self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + self.learning_rate * q_update

        # Decay exploration rate
        if self.exploration_rate > 0.01:
            self.exploration_rate *= self.exploration_decay

    def save(self, filename):
        """
        Save the trained Q-table.
        :param filename: The file name to save the Q-table.
        """
        np.save(filename, self.q_table)

    def load(self, filename):
        """
        Load a trained Q-table.
        :param filename: The file name to load the Q-table.
        """
        self.q_table = np.load(filename)
