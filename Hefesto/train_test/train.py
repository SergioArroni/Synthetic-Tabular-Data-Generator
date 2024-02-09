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

    def __init__(self, model: Model, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def train_model(self, train_loader, epochs):
        
        self.model.train()

        # Define the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Define the loss function
        loss_fn = MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for features, _ in tqdm(train_loader):
                features = features.to(self.device)
                
                optimizer.zero_grad()

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.model(features)

                # Compute and print loss
                loss = loss_fn(y_pred.squeeze(), features)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}"
            )
