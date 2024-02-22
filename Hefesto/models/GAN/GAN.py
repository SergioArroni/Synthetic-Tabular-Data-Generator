from torch import nn
import torch

from Hefesto.models.model import Model


class GANModel(Model):
    def __init__(self, input_dim, hidden_dim, generator_dim=10):
        super().__init__(input_dim, hidden_dim)

        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, generator_dim),
            nn.Tanh(),  # Assuming generator output is in the range [-1, 1]
        )

        self.discriminator = nn.Sequential(
            nn.Linear(generator_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output a probability between 0 and 1
        )

    def forward(self, x):
        z = self.generator(x)
        return self.discriminator(z)

    def __str__(self):
        return "GANModel"
