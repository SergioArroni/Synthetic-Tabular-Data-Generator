from torch import nn


class Model(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        pass
