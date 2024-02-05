from torch import nn
import torch

from Hefesto.models.model import Model

class DiffusionModel(Model):
    def __init__(self, input_dim, hidden_dim, n_steps):
        super().__init__(input_dim, hidden_dim, n_steps)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        for _ in range(self.n_steps):
            z = z * 0.9 + torch.randn_like(z)
            x = self.decoder(z)
        return x
