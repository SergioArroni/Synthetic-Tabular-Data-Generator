import time
from matplotlib import pyplot as plt

import torch

from Hefesto.models.model import Model


def save_model(path, model):
    # Save the model
    torch.save(model.state_dict(), path)


def load_model(path: str, model: Model) -> Model:
    model.load_state_dict(torch.load(path))
    return model.to(model.device)


def save_data(path, data):
    # Save the data
    data.to_csv(path, sep=";", index=False)


def write_results(
    epochs,
    good_ele,
    bad_ele,
    path: str,
    size: int,
    model: Model,
    seed: int,
    metrics: tuple,
):
    with open(path, "a") as a:
        a.write(f"Seed: {seed}\n")
        a.write(f"Model: {model}\n")
        a.write(f"Epochs: {epochs}\n")
        a.write(f"Good Data Gen: {len(good_ele)}\n")
        a.write(f"Bad Data Gen: {len(bad_ele)}\n")
        a.write(f"Acierto: {(len(good_ele)/size)*100}%\n")
        a.write(f"F1: {metrics[0]}\n")
        a.write(f"Accuracy: {metrics[1]}\n")
        a.write("---------------------------------------------------------------\n")


def plot_statistics(df, path: str):
    # Crear un boxplot para cada atributo
    for i, column in enumerate(df.columns, 1):
        fig, ax = plt.subplots()
        df.boxplot(column, ax=ax)

        # Obtener estadísticas descriptivas
        median = df[column].median()
        mean = df[column].mean()
        std = df[column].std()

        # Añadir un texto para la desviación estándar
        # Añadir líneas horizontales para la media y la mediana
        ax.axhline(median, color="red", linestyle="-", label=f"Median: {median:.2f}")
        ax.axhline(mean, color="green", linestyle="--", label=f"Mean: {mean:.2f}")
        # ax.axhline(std, color="blue", linestyle="-.", label=f"Std: {std:.2f}")

        ax.text(
            0.95,
            0.01,
            f"Std: {std:.2f}",
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=ax.transAxes,
            color="blue",
            fontsize=10,
        )

        # Mostrar la leyenda
        plt.legend()

        plt.tight_layout()  # Ajustar automáticamente los subplots para que encajen en la figura
        plt.savefig(path + f"_{column}.png")
        # plt.show()
        plt.close()
