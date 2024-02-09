import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, data_dim, T, betas):
        """
        Inicializa el modelo de difusión.
        
        Parámetros:
        - data_dim: Dimensiones de los datos de entrada.
        - T: Número total de pasos en el proceso de difusión.
        - betas: Lista de parámetros beta para el proceso de difusión.
        """
        super(DiffusionModel, self).__init__()
        self.T = T
        self.betas = betas
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.ReLU(),
            nn.Linear(512, data_dim),
        )

    def forward(self, x):
        """
        Aplica el modelo a los datos de entrada.
        """
        return self.model(x)
    
    def forward_diffusion_process(self, x_0):
        """
        Realiza el proceso de difusión para un tensor de entrada.
        """
        x_t = x_0
        for t in range(self.T):
            beta_t = self.betas[t]
            noise = torch.randn_like(x_0) * torch.sqrt(beta_t)
            x_t = torch.sqrt(1.0 - beta_t) * x_t + noise
        return x_t

    def reverse_diffusion_process(self, x_T):
        """
        Realiza el proceso de reversión para un tensor de entrada.
        """
        x_t = x_T
        for t in reversed(range(self.T)):
            beta_t = self.betas[t]
            # Aquí asumimos una simplificación donde el modelo predice directamente x_{t-1}
            x_t = self.forward(x_t)  # Esta línea es altamente simplificada
            x_t = (x_t - torch.sqrt(beta_t) * torch.randn_like(x_t)) / torch.sqrt(1.0 - beta_t)
        return x_t
