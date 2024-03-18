from torch import nn
import torch
from torch.utils.data import DataLoader
from Hefesto.models.model import Model


class DiffusionModel(Model):
    def __init__(self, input_dim, hidden_dim, T, device, alpha, betas=None):
        super().__init__(input_dim, hidden_dim, device=device)

        self.t_value = T
        self.betas = self._init_betas() if betas is None else betas
        self.device = device
        # Aquí 'alpha' es un hiperparámetro que determina cuánto afecta la std al ruido
        self.alpha = alpha  # Este valor es un ejemplo; ajusta según necesites

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
        )

        self.transformer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, device=self.device, batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, input_dim),
        )

        self.apply(self._init_weights)

    def forward(self, x: DataLoader) -> torch.Tensor:
        x = x.to(self.device)
        z = self.encoder(x)
        z = self.transformer(z)

        for t in range(self.t_value):
            beta_t = self.betas[t]

            # Calcular la desviación estándar de las activaciones de 'z'
            std_z = torch.std(z)

            # Ajustar el nivel de ruido basado en la desviación estándar
            adjusted_noise_scale = torch.sqrt(beta_t * (1 + self.alpha * std_z))

            noise = torch.randn_like(z) * adjusted_noise_scale.to(self.device)
            z = torch.sqrt(1.0 - beta_t).to(self.device) * z + noise

        x = self.decoder(z)
        return x

    def train_model(self, model, input, optimizer) -> torch.Tensor:
        y_pred = model(input)
        loss = self.loss_fn(y_pred.squeeze(), input)
        return loss

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_betas(self):
        scale = 5  # Ajusta esto según necesites para controlar la 'rapidez' de la curva
        betas = torch.exp(
            torch.linspace(-scale, scale, self.t_value)
        )  # Curva exponencial
        betas = (betas - betas.min()) / (betas.max() - betas.min())  # Normalización
        betas = 0.1 + (0.9 - 0.1) * betas  # Ajuste al rango [0.1, 0.9]
        return betas

    def __str__(self):
        return "DiffusionModel"
