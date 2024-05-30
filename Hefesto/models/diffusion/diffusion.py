import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from Hefesto.models.model import Model
from Hefesto.preprocess.load_data import do_data_loader


class DiffusionModel(Model):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
        T,
        device,
        alpha,
        seed,
        betas=None,
    ):
        super().__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, seed=seed, device=device
        )
        self.dropout = dropout

        self.t_value = T
        self.betas = self._init_betas() if betas is None else betas
        self.device = device
        # Aquí 'alpha' es un hiperparámetro que determina cuánto afecta la std al ruido
        self.alpha = alpha

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 3),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            # nn.Linear(hidden_dim * 3, hidden_dim * 2),
            # nn.LeakyReLU(),
            # nn.Dropout(self.dropout),
        )

        # self.mu_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        # self.log_var_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 3, nhead=4, dropout=self.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=4)

        # decoder_layers = nn.TransformerDecoderLayer(
        #     d_model=hidden_dim * 3, nhead=4, dropout=self.dropout, batch_first=True
        # )
        
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=4)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            # nn.Linear(hidden_dim * 2, hidden_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(self.dropout),
            nn.Linear(hidden_dim * 3, input_dim),
        )
        

        self.apply(self._init_weights)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x) -> torch.Tensor:

        x = x.to(self.device)
        z = self.encoder(x)
        # mu = self.mu_layer(z)
        # log_var = torch.clamp(self.log_var_layer(z), min=-10, max=10)
        # z = self.reparameterize(mu, log_var)
        # z = self.transformer(z)

        for t in range(self.t_value):
            beta_t = self.betas[t]

            # Calcular la desviación estándar de las activaciones de 'z'
            std_z = torch.std(z)

            # Ajustar el nivel de ruido basado en la desviación estándar
            adjusted_noise_scale = torch.sqrt(beta_t * (1 + self.alpha * std_z))

            noise = torch.randn_like(z) * adjusted_noise_scale.to(self.device)
            z = torch.sqrt(1.0 - beta_t).to(self.device) * z + noise

        # z = self.transformer_decoder(z, z)
        # z = z.unsqueeze(1)
        # z = self.transformer(z).squeeze(1)

        x = self.decoder(z)
        
        return x

    def train_model(self, model, input, optimizer, train=True) -> None:

        if train:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        y_pred = model(input)
        loss = self.loss_fn(y_pred.squeeze(), input)

        # print(loss)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            self.epoch_train_loss += loss.item()
        else:
            self.epoch_val_loss += loss.item()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_betas(self):
        scale = 5
        betas = torch.exp(
            torch.linspace(-scale, scale, self.t_value)
        )  # Curva exponencial
        betas = (betas - betas.min()) / (betas.max() - betas.min())  # Normalización
        betas = 0.1 + (0.9 - 0.1) * betas  # Ajuste al rango [0.1, 0.9]
        return betas

    def __str__(self):
        return "DiffusionModel"
