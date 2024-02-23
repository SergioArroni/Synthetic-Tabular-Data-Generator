from torch import nn
import torch


class Model(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x) -> torch.Tensor:
        pass
    
