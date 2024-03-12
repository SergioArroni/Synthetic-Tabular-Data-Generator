import torch
from torch import nn
from Hefesto.models.model import Model


class GANModel(Model):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Update the loss function for GAN-specific training
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        raise NotImplementedError("This method is not utilized for GANs.")

    def train_model(self, input, optimizer_gen, optimizer_disc) -> torch.Tensor:
        batch_size = input.size(0)
        real_labels = torch.ones(batch_size, 1, device=input.device)
        fake_labels = torch.zeros(batch_size, 1, device=input.device)

        ### Entrenamiento del Discriminador ###
        optimizer_disc.zero_grad()

        # Real
        real_output = self.discriminator(input)
        d_loss_real = self.loss_fn(real_output, real_labels)

        # Falso
        z = torch.randn(batch_size, self.input_dim, device=input.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images.detach())
        d_loss_fake = self.loss_fn(fake_output, fake_labels)

        # Combinar pérdidas
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_disc.step()

        ### Entrenamiento del Generador ###
        optimizer_gen.zero_grad()

        # Generar imágenes falsas para el entrenamiento del generador
        z = torch.randn(batch_size, self.input_dim, device=input.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images)
        g_loss = self.loss_fn(fake_output, real_labels)

        g_loss.backward()
        optimizer_gen.step()

        return d_loss + g_loss

    def __str__(self):
        return "GANModel"
