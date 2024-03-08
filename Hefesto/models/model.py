from torch import nn
import torch


class Model(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epoch_val_loss = 0.0
        self.epoch_train_loss = 0.0
        self.loss_fn = nn.MSELoss()

    def train_model(self, model, input) -> torch.Tensor:
        # Add your implementation here
        pass

    def train_model_gen(self, model, input, optimizer, train=True):

        if train:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        loss = self.train_model(
            model, input
        )  # Assign the result of the train_model method to the loss variable
        if train:
            print(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            self.epoch_train_loss += loss.item()
        else:
            self.epoch_val_loss += loss.item()

    def test_model_gen(self, model, input) -> torch.Tensor:
        model.eval()
        gen = model(input).round()
        return gen
