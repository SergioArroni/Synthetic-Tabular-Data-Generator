from matplotlib import pyplot as plt
import numpy as np
import torch
import torch
from torch import optim
from torch.nn import MSELoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from Hefesto.models.model import Model
from Hefesto.utils.utils import save_model


class Train:

    def __init__(self, model: Model, device: torch.device, timestamp: float) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.timestamp = timestamp

    def train_model(self, train_loader, val_loader, epochs):
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = MSELoss()
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            # Training phase
            for features in tqdm(train_loader.dataset.features):
                features = features.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(features)
                loss = loss_fn(y_pred.squeeze(), features)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # Validation phase
            with torch.no_grad():
                self.model.eval()  # Set model to evaluation mode
                for features in tqdm(val_loader.dataset.features):
                    features = features.to(self.device)
                    y_pred = self.model(features)
                    loss = loss_fn(y_pred.squeeze(), features)
                    epoch_val_loss += loss.item()
                self.model.train()  # Set model back to train mode

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

        # Plotting
        epochs_range = np.arange(1, epochs+1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Train and Validation Loss for {self.model} model')
        plt.grid()
        plt.savefig(f'./img/train/train_val_loss_{self.model}_{self.timestamp}.png')
        plt.show()
 