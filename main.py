from numpy import float32
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_steps):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        for _ in range(self.n_steps):
            z = z * 0.9 + torch.randn_like(z)
            x = self.decoder(z)
        return x


def train(
    model: DiffusionModel,
    data: pd.DataFrame,
    optimizer: AdamW,
    n_epochs: int,
    tolerance: int = 0.01,
) -> (DiffusionModel, list):
    loss_a = []
    loss_function = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        loss = 0

        # Iterar sobre el DataFrame
        for row in data.iterrows():
            # Obtener los datos de la fila
            x = row[1].values

            # Convertir los datos a un tensor
            x_tensor = torch.tensor(x)

            # Generar una muestra
            x_gen = model(x_tensor)

            # Calcular la pérdida
            loss += loss_function(x_gen, x_tensor)

        # Retropropagar la pérdida
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Época {epoch + 1}: pérdida = {loss.item()}")
        loss_a.append(loss.item())
        if loss.item() < tolerance:
            break

    return model, loss_a


def plot_epochs(epochs: list, losses: list) -> None:
    plt.plot(epochs, losses)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.show()


def save_model(path: str, model: DiffusionModel) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, shape: int) -> DiffusionModel:
    model = DiffusionModel(shape, 128, 10)
    model.load_state_dict(torch.load(path))
    return model


def do_train(df_train: pd.DataFrame, epochs: int) -> None:
    model = DiffusionModel(df_train.shape[1], 128, 10)

    optimizer = AdamW(model.parameters(), lr=0.001)

    trained_model, loss = train(model, df_train, optimizer, epochs)

    save_model("./models/model.pt", trained_model)

    ep = np.arange(1, epochs + 1)

    plot_epochs(ep, loss)


def main():
    seed = 0
    df = pd.read_csv("data\cardio\cardio_train.csv")

    df = df.astype(float32)

    df = df.sample(frac=1, random_state=seed)

    n = 1000
    m = 500
    v = 1000

    df_train = df.iloc[:n]
    df_test = df.iloc[n : n + m]
    df_val = df.iloc[n + m : n + m + v]

    epochs = 200

    trained_model = load_model("./models/model.pt", df_train.shape[1])

    # do_train(model, df_train, epochs)

    test_tensor = torch.tensor(df_test.iloc[0].values)
    x_gen = trained_model(test_tensor)

    clf = IsolationForest(random_state=seed).fit(df_train.values)

    good_ele = []
    bad_ele = []

    for ele in df_val.values:
        if clf.predict([ele]) == 1:
            good_ele.append(ele)
        else:
            bad_ele.append(ele)

    a = open("./results/results.txt", "a")

    a.write(f"Epochs: \n{epochs}\n")
    a.write(f"Array In: \n{df_test.iloc[0].values}\n")
    a.write(f"Array Gen: \n{x_gen.detach().numpy()}\n")
    # a.write(f"Data Real: {clf.predict([df_test.iloc[0].values])}\n")
    # a.write(f"Data Gen: {clf.predict([x_gen.detach().numpy()])}\n")
    a.write(f"Good Data Gen: {len(good_ele)}\n")
    a.write(f"Bad Data Gen: {len(bad_ele)}\n")
    a.write(f"Acierto: {(len(good_ele)/df_val.shape[0])*100}%\n")

    a.write("---------------------------------------------------------------\n")
    a.close()


if __name__ == "__main__":
    main()
