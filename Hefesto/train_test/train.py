from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from Hefesto.models.model import Model
from Hefesto.utils.utils import save_model


class Train:

    def __init__(self, type_mode: Model):
        self.type_model = type_mode
        pass

    def __train(
        self,
        data: pd.DataFrame,
        optimizer: AdamW,
        n_epochs: int,
        tolerance: int = None,
    ) -> (Model, list):
        loss_a = []
        loss_function = torch.nn.MSELoss()
        epoch = 0

        while True:
            loss = 0
            loss_ant = 1

            # Iterar sobre el DataFrame
            for row in data.iterrows():
                # Obtener los datos de la fila
                x = row[1].values

                # Convertir los datos a un tensor
                x_tensor = torch.tensor(x)

                # Generar una muestra
                x_gen = self.model(x_tensor)

                # Calcular la pérdida
                loss += loss_function(x_gen, x_tensor)

            # Retropropagar la pérdida
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Época {epoch + 1}: pérdida = {loss.item()}")
            loss_a.append(loss.item())
            epoch += 1
            if tolerance is not None and abs(loss.item() - loss_ant) < tolerance:
                break
            elif epoch == n_epochs:
                break
            loss_ant = loss.item()

        return loss_a

    def do_train(self, df_train: pd.DataFrame, epochs: int, tolerance: int) -> None:
        self.model = self.type_model(df_train.shape[1], 128, 10)

        optimizer = AdamW(self.model.parameters(), lr=0.001)

        loss = self.__train(df_train, optimizer, epochs)

        save_model("./save_models/model_Diffusion.pt", self.model)

        ep = np.arange(1, epochs + 1)

        self.__plot_epochs(ep, loss)

    def __plot_epochs(self, epochs: list, losses: list) -> None:
        plt.plot(epochs, losses)
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.show()
