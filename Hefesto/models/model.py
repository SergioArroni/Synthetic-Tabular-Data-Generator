from torch import nn
import torch


class Model(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, seed: int, device: torch.device
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.epoch_val_loss = 0.0
        self.epoch_train_loss = 0.0
        self.loss_fn = nn.MSELoss()
        self.device = device

    def train_model(self, model, input, optimizer, train=True) -> None:

        pass

    def test_model_gen(self, model, input) -> torch.Tensor:
        model.eval()
        gen = model(input).round()
        return gen
