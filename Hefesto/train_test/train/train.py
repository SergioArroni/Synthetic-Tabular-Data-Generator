from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn import MSELoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from Hefesto.models.model import Model
from Hefesto.utils.utils import save_model


class Train:

    def __init__(
        self,
        model: Model,
        device: torch.device,
        timestamp: float,
        epochs: int,
        patience: int = 10,
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.timestamp = timestamp
        self.train_losses = []
        self.val_losses = []
        self.epochs = epochs
        self.patience = (
            patience  # Número de épocas para esperar después de la última mejora.
        )

    def train_model(self, train_loader, val_loader) -> None:
        best_loss = float("inf")
        epochs_no_improve = 0
        x = 100  # Threshold for early stopping
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(self.epochs):

            # Training phase
            for features, _ in tqdm(train_loader):
                inputs = features.to(self.device)

                self.model.train_model(self.model, inputs, optimizer)

            # Validation phase
            with torch.no_grad():
                for features, _ in tqdm(val_loader):
                    self.model.train_model(self.model, inputs, optimizer, False)

                self.model.train()  # Set model back to train mode

            avg_train_loss = self.model.epoch_train_loss / len(train_loader)
            avg_val_loss = self.model.epoch_val_loss / len(val_loader)
            print(
                f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            # Early stopping
            # if avg_val_loss - avg_train_loss > x:
            #     epochs_no_improve += 1
            #     if epochs_no_improve == self.patience:
            #         print("Early stopping!")
            #         break
            # else:
            #     epochs_no_improve = 0
        
        self.plotting()

    def plotting(self) -> None:
        # Plotting the training and validation loss
        epochs_range = np.arange(1, self.epochs + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, self.train_losses, label="Train Loss")
        plt.plot(epochs_range, self.val_losses, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Train and Validation Loss for {self.model} model")
        plt.grid()
        plt.savefig(f"./img/train/train_val_loss_{self.model}_{self.timestamp}.png")
