from torch import nn
import torch
from Hefesto.models.model import Model


class VAEModel(Model):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super().__init__(input_dim, hidden_dim, device)

        self.latent_dim = latent_dim
        self.overall_loss = 0.0
        # self.EPS = 1e-10  # Pequeña constante para evitar varianza cero

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        self.decoder.apply(self.init_weights)

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    # def loss_function(self, x, x_hat, mean, log_var):
    #     reproduction_loss = nn.functional.binary_cross_entropy(
    #         x_hat, x, reduction="sum"
    #     )
    #     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    #     return reproduction_loss + KLD

    def train_model(self, model, input, optimizer) -> torch.Tensor:
        reconstruction, mu, log_var = model(input)
        loss = self.loss_fn(reconstruction.squeeze(), input)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss += kl_divergence
        self.overall_loss += loss.item()

        return loss

    def test_model_gen(self, model, input) -> torch.Tensor:
        model.eval()
        gen, mu, log_var = model(input)
        return gen.round()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __str__(self):
        return "VAE"
