import os

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticQNet(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: tuple[int, ...],
    ) -> None:
        super().__init__()

        layers = []
        previous_size = state_dim + action_dim
        for size in np.atleast_1d(hidden_size):
            layers.append(nn.Linear(previous_size, int(size)))
            previous_size = int(size)
        layers.append(nn.Linear(previous_size, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.hstack([x1, x2])

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)

        return x


class ActorQNet(nn.Module):

    def __init__(
        self,
        state_dim: int,
        output_size: int,
        hidden_size: tuple[int, ...],
    ) -> None:
        super().__init__()

        layers = []
        previous_size = state_dim
        for size in np.atleast_1d(hidden_size):
            layers.append(nn.Linear(previous_size, int(size)))
            previous_size = int(size)
        layers.append(nn.Linear(previous_size, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)

        return F.softmax(x, dim=1)

    def save(self, file_path: str | Path = "models/model.pth") -> None:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        torch.save(self, file_path)
