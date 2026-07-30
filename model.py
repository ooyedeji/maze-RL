import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from maze import MazeStatus


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        layers = []
        previous_size = input_size
        for size in np.atleast_1d(hidden_size):
            layers.append(nn.Linear(previous_size, int(size)))
            previous_size = int(size)
        layers.append(nn.Linear(previous_size, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)

        return x


class QTrainer:
    def __init__(self, model: nn.Module, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self, state, action, status, reward, state_next):
        state = torch.tensor(np.array(state), dtype=torch.float)
        state_next = torch.tensor(np.array(state_next), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            state_next = torch.unsqueeze(state_next, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            status = (status,)

        # Predicted Q-values and targets
        pred_actions: torch.Tensor = self.model(state)
        # Q_sa = pred_actions.clone()
        Q_sa_next = self.model(state_next)

        Q_sa = torch.zeros(len(status), requires_grad=True).clone()
        Q_sa_pred = torch.zeros(len(status), requires_grad=True).clone()

        # Update Q-values based on rewards and the Bellman equation
        for i in range(len(status)):
            Q_sa[i] = pred_actions[i, action[i]]

            Q_sa_pred[i] = reward[i]
            if status[i] == MazeStatus.RUNNING:
                Q_sa_pred[i] += self.gamma * torch.max(Q_sa_next[i])

        # Backpropagate the loss to compute gradients
        self.optimizer.zero_grad()
        # print(Q_sa.shape, action)
        loss: torch.Tensor = self.criterion(Q_sa_pred, Q_sa)
        loss.backward()

        # Update the model parameters
        self.optimizer.step()

    def save(self, file_path="models/model.pth"):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        torch.save(self.model, file_path)
