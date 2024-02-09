from torch import nn
import torch
from torch.utils.data import DataLoader
from Hefesto.models.model import Model


class DiffusionModel(Model):
    def __init__(self, input_dim, hidden_dim, n_steps, n_transformer_layers=2):
        super().__init__(input_dim, hidden_dim, n_steps)

        self.n_transformer_layers = n_transformer_layers

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            *self._make_transformer_layers(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            *self._make_transformer_layers(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def _make_transformer_layers(self, hidden_dim):
        layers = []
        for _ in range(self.n_transformer_layers):
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True
            )
            layers.append(nn.TransformerEncoder(transformer_layer, num_layers=1))
        return layers

    def forward(self, x: DataLoader):
        z = self.encoder(x)
        for _ in range(self.n_steps):
            z = z * 0.9 + torch.randn_like(z)
            x = self.decoder(z)
        return x

    def __str__(self):
        return "DiffusionModel"
