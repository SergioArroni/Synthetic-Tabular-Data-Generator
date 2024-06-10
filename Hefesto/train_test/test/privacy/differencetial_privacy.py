import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Hefesto.train_test.test.privacy import Privacy


class DifferentialPrivacyModel(nn.Module):
    def __init__(self, input_dim):
        super(DifferentialPrivacyModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


class DifferentialPrivacy(Privacy):
    def __init__(self, data, gen_data, path: str, epsilon=1.0, delta=1e-5):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.epsilon = epsilon
        self.delta = delta
        self.model = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def add_laplace_noise(self, scale):
        return torch.from_numpy(np.random.laplace(0, scale, 1).astype(np.float32))

    def train_with_differential_privacy(self, epochs=50, batch_size=32, lr=0.01):
        if isinstance(self.data, torch.Tensor):
            data_array = self.data.clone().detach().cpu().numpy()
        else:
            data_array = self.data.to_numpy()

        if isinstance(self.gen_data, torch.Tensor):
            gen_data_array = self.gen_data.clone().detach().cpu().numpy()
        else:
            gen_data_array = self.gen_data.to_numpy()

        input_dim = data_array.shape[1]
        self.model = DifferentialPrivacyModel(input_dim).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        data_loader = DataLoader(
            TensorDataset(
                torch.tensor(data_array, dtype=torch.float32),
                torch.tensor(gen_data_array, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.clone().detach().to(
                    self.model.fc.weight.device
                ), batch_y.clone().detach().to(self.model.fc.weight.device)

                self.optimizer.zero_grad()
                outputs = self.model.forward(batch_x)
                loss = self.loss_fn(outputs, batch_y)

                loss.backward()

                # Add Laplace noise to gradients
                for param in self.model.parameters():
                    noise_scale = (2 * batch_size * np.log(1.25 / self.delta)) / (
                        len(data_array) * self.epsilon
                    )
                    param.grad.data += self.add_laplace_noise(noise_scale).to(
                        param.device
                    )

                self.optimizer.step()

    def calculate_privacy_loss(self):
        # This function can be expanded to calculate and save more details about the privacy loss.
        # For this example, we just save the epsilon and delta.
        with open(self.path, "w") as file:
            file.write(
                f"Privacy parameters (epsilon, delta): ({self.epsilon}, {self.delta})\n"
            )

    def execute(self):
        self.train_with_differential_privacy()
        self.calculate_privacy_loss()
