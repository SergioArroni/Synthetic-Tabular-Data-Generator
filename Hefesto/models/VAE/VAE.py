from torch import nn
import torch
from Hefesto.models.model import Model


class VAEModel(Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim)
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Outputs both mu and log_var
        )
        
        self.encoder.apply(self.init_weights)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self.decoder.apply(self.init_weights)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(
            encoded, 2, dim=1
        )  # Split the encoded values into mu and log_var
        z = self.reparameterize(mu, log_var)

        # Decode
        return self.decoder(z), mu, log_var

    def train_model(self, model, input) -> torch.Tensor:
        reconstruction, mu, log_var = model(input)
        loss = self.loss_fn(reconstruction, input)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - (log_var.exp() + 1e-8))
        loss += kl_div
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
