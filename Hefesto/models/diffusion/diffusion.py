from torch import nn
import torch
from torch.utils.data import DataLoader
from Hefesto.models.model import Model


class DiffusionModel(Model):
    def __init__(self, input_dim, hidden_dim, T, betas, device, n_transformer_layers=2):
        super().__init__(input_dim, hidden_dim)

        self.betas = betas
        self.T = T
        self.n_transformer_layers = n_transformer_layers
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *self.__make_transformer_layers(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            *self.__make_transformer_layers(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def __make_transformer_layers(self, hidden_dim):
        layers = []
        for _ in range(self.n_transformer_layers):
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
            )
            layers.append(nn.TransformerEncoder(transformer_layer, num_layers=1))
        return layers

    def forward(self, x: DataLoader) -> torch.Tensor:
        z = self.encoder(x)
        for t in range(self.T):
            beta_t = self.betas[t]
            noise = torch.randn_like(z) * torch.sqrt(beta_t)
            z = torch.sqrt(1.0 - beta_t) * z + noise
            x = self.decoder(z)
        return x

    def __str__(self):
        return "DiffusionModel"
