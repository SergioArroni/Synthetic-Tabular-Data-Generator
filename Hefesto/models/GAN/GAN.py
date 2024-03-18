import torch
from torch import nn
from Hefesto.models.model import Model


class Generator(Model):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__(input_dim, hidden_dim, device=device)
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 8, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.generator(x)

    def __str__(self):
        return "Generator"


class Discriminator(Model):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__(input_dim, hidden_dim, device=device)

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.discriminator(x)

    # def train_model(self, input, optimizer_gen, optimizer_disc) -> torch.Tensor:
    #     batch_size = input.size(0)
    #     real_labels = torch.ones(batch_size, 1, device=input.device)
    #     fake_labels = torch.zeros(batch_size, 1, device=input.device)

    #     ### Entrenamiento del Discriminador ###
    #     optimizer_disc.zero_grad()

    #     # Real
    #     real_output = self.discriminator(input)
    #     d_loss_real = self.loss_fn(real_output, real_labels)

    #     # Falso
    #     z = torch.randn(batch_size, self.input_dim, device=input.device)
    #     fake_images = self.generator(z)
    #     fake_output = self.discriminator(fake_images.detach())
    #     d_loss_fake = self.loss_fn(fake_output, fake_labels)

    #     # Combinar pérdidas
    #     d_loss = d_loss_real + d_loss_fake
    #     d_loss.backward()
    #     optimizer_disc.step()

    #     ### Entrenamiento del Generador ###
    #     optimizer_gen.zero_grad()

    #     # Generar imágenes falsas para el entrenamiento del generador
    #     z = torch.randn(batch_size, self.input_dim, device=input.device)
    #     fake_images = self.generator(z)
    #     fake_output = self.discriminator(fake_images)
    #     g_loss = self.loss_fn(fake_output, real_labels)

    #     g_loss.backward()
    #     optimizer_gen.step()

    #     return d_loss + g_loss

    def __str__(self):
        return "Discriminator"


class GANModel(Model):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__(input_dim, hidden_dim, device=device)
        lr = 0.00001

        self.generator = Generator(input_dim, hidden_dim, device)
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=lr)

        self.discriminator = Discriminator(input_dim, hidden_dim, device)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def forward(self, x):
        raise NotImplementedError("This method is not implemented")

    def get_noise(self, n_samples, noise_vector_dimension, device="cpu"):
        return torch.randn(n_samples, noise_vector_dimension, device=device)

    def get_disc_loss(
        self, gen, disc, criterion, real, num_images, noise_dimension, device
    ):
        # Generate noise and pass to generator
        fake_noise = self.get_noise(num_images, noise_dimension, device=device)
        fake = gen(fake_noise)

        # Pass fake features to discriminator
        # All of them will got label as 0
        # .detach() here is to ensure that only discriminator parameters will get update
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

        # Pass real features to discriminator
        # All of them will got label as 1
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

        # Average of loss from both real and fake features
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss

    def get_gen_loss(self, gen, disc, criterion, num_images, noise_dimension, device):
        # Generate noise and pass to generator
        fake_noise = self.get_noise(num_images, noise_dimension, device=device)
        fake = gen(fake_noise)

        # Pass fake features to discriminator
        # But all of them will got label as 1
        disc_fake_pred = disc(fake)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss

    def train_model(self, train_loader, val_loader):

        # output_layer = decoder(encoder(input_layer)[2])
        # vae = Model(input_layer, output_layer, name="autoencoder")

        # reconstruction_axis = (1, 2)
        # if is_rgb:
        #     reconstruction_axis = (1, 2, 3)

        # reconstruction_loss = tf.reduce_mean(
        #     1000.0 * tf.square(input_layer - output_layer), axis=reconstruction_axis
        # )

        # kl_loss = -0.5 * K.sum(
        #     1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1
        # )

        # vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        # vae.add_loss(vae_loss)
        # vae.add_metric(
        #     tf.reduce_sum(
        #         1000.0 * tf.square(input_layer - output_layer), axis=reconstruction_axis
        #     ),
        #     name="reconstruction_loss",
        #     aggregation="mean",
        # )
        # vae.add_metric(kl_loss, name="kl_loss", aggregation="mean")
        # vae.compile(optimizer="adam")

        # https://medium.com/@morgan_lynch/generative-ai-with-variational-autoencoders-86d1926df6e8

        return vae, reconstruction_loss

    def __str__(self):
        return "GANModel"
