import torch
import random
import numpy as np
from collections import deque
from maze import Direction
from model import Linear_QNet, QTrainer


class Agent:
    MAX_MEMORY = 1_000_000

    def __init__(
        self,
        model: Linear_QNet = None,
        trainer: QTrainer = None,
        batch_size=1000,
        random_state=42,
    ):
        random.seed(random_state)
        self.n_episode = 1
        self.memory = deque(maxlen=self.MAX_MEMORY)
        if trainer is None and model is None:
            raise ValueError("One of trainer and model must be set")
        self.trainer = trainer
        self.model = model if self.trainer is None else self.trainer.model
        self.batch_size = batch_size
        self.epsilon_end = 0.01

        # Possible movement directions
        self.directions = list(Direction.__members__.values())

    @property
    def epsilon(self) -> float:
        x = self.n_episode
        theta1, theta2, theta3 = 1 - self.epsilon_end, 0.05, 50
        beta = 1
        y = theta1 / (1 + beta * np.exp(-theta2 * (x - theta3))) ** (1 / beta)

        return round(1 - y, 2)

    def memorize(self, state, action, status, reward, next_state):
        self.memory.append((state, action, status, reward, next_state))

    def train_long_memory(self):
        if self.trainer:
            batch_size = min(self.batch_size, len(self.memory))
            sample = random.sample(self.memory, batch_size)
            self.trainer.train(*zip(*sample))

    def train_short_memory(self):
        if self.trainer:
            self.trainer.train(*self.memory[-1])

    def get_action(self, state, explore=True):
        if self.epsilon > random.random() and explore:
            # Exploration
            action_id = random.randint(0, len(self.directions) - 1)
        else:
            # Exploitation
            state = torch.tensor(np.array(state), dtype=torch.float)
            state = state.unsqueeze(0)
            action_id = torch.argmax(self.model(state)).item()

        return action_id
