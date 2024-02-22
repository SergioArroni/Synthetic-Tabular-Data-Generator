from torch import nn
import torch
import torch.nn.functional as F

from Hefesto.models.model import Model


class VAEModel(Model):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        self.encoder = nn.Linear(input_dim, hidden_dim * 2)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h = F.relu(self.encoder(x))
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded

    def __str__(self):
        return "VAEModel"
