import random

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from maze import Direction, MazeStatus
from model import ActorQNet, CriticQNet


class Agent:
    MAX_MEMORY = 1_000_000

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_size: tuple[int, ...] = (128,),
        batch_size: int = 1_000,
        lr: float = 0.001,
        gamma: float = 0.90,
        tau: float = 0.005,
        random_state: int = 42,
    ) -> None:
        random.seed(random_state)

        self.actor = ActorQNet(state_dim, action_dim, hidden_size)
        self.actor_target = ActorQNet(state_dim, action_dim, hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticQNet(state_dim, action_dim, hidden_size)
        self.critic_target = CriticQNet(state_dim, action_dim, hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon_end = 0.01
        self.criterion = torch.nn.MSELoss()

        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.episode = 1

        # Possible movement directions
        self.directions = list(Direction.__members__.values())

    @property
    def epsilon(self) -> float:
        x = self.episode
        theta1, theta2, theta3 = 1 - self.epsilon_end, 0.05, 50
        beta = 1
        y = theta1 / (1 + beta * np.exp(-theta2 * (x - theta3))) ** (1 / beta)

        return round(1 - y, 2)

    def memorize(
        self,
        state: np.ndarray[tuple[int, ...], float],
        direction: Direction,
        status: MazeStatus,
        reward: float,
        next_state: np.ndarray[tuple[int, ...], float],
    ) -> None:
        _direction = direction.value
        _status = float(status.value is None)
        self.memory.append((state, _direction, _status, reward, next_state))

    def train_long_memory(self) -> None:
        batch_size = min(self.batch_size, len(self.memory))
        sample = random.sample(self.memory, batch_size)
        self.train(*zip(*sample))

    def train_short_memory(self) -> None:
        self.train(*self.memory[-1])

    def get_action(self, state, explore=True) -> Direction:
        if self.epsilon > random.random() and explore:
            # Exploration
            action_id = random.randint(0, len(self.directions) - 1)
        else:
            # Exploitation
            state = torch.tensor(np.array(state), dtype=torch.float)
            state = state.unsqueeze(0)
            action_id = torch.argmax(self.actor(state)).item()

        return self.directions[action_id]

    def train(
        self,
        state,
        action,
        status,
        reward,
        state_next,
    ):
        state = torch.tensor(np.array(state), dtype=torch.float)
        state_next = torch.tensor(np.array(state_next), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float).unsqueeze(0)

        if state.ndim == 1:
            state = torch.unsqueeze(state, 0)
            state_next = torch.unsqueeze(state_next, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            status = (status,)

        reward = reward.reshape(-1, 1)
        status = torch.Tensor([s for s in status]).unsqueeze(0)
        status = status.reshape(-1, 1)

        # Predicted Q-values and targets
        q_pred = self.critic(state, action)

        with torch.no_grad():
            action_next = self.actor_target(state_next)
            q_next = self.critic_target(state_next, action_next)

        q_target = reward + self.gamma * q_next * status

        # Backpropagate the loss to compute gradients
        self.critic_opt.zero_grad()
        critic_loss: torch.Tensor = self.criterion(q_pred, q_target)
        critic_loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        action_pred = self.actor(state)
        actor_loss = -torch.mean(self.critic(state, action_pred))
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data)
