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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=n_transformer_layers
        )

        self.encoder.apply(self.init_weights)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.decoder.apply(self.init_weights)

    def forward(self, x: DataLoader) -> torch.Tensor:
        z = self.encoder(x)
        z = self.transformer_encoder(z)
        for t in range(self.T):
            beta_t = self.betas[t]
            noise = torch.randn_like(z) * torch.sqrt(beta_t)
            z = torch.sqrt(1.0 - beta_t) * z + noise
            x = self.decoder(z)
        return x

    def train_model(self, model, input) -> torch.Tensor:
        y_pred = model(input)
        loss = self.loss_fn(y_pred.squeeze(), input)
        return loss

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __str__(self):
        return "DiffusionModel"
