import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Metrics(Stadistics):
    def __init__(self, original_data, synthetic_data, path: str):
        super().__init__(data=original_data, path=path)
        self.synthetic_data = synthetic_data
        self.metrics = None

    def calculate_metrics(self):
        # Calculamos la mediana
        median = np.median(self.data)

        # Calculamos la varianza
        variance = self.data.var(axis=0)

        # Calculamos la media
        mean = np.mean(self.data)

        # Calculamos la moda
        mode_value, _ = mode(self.data)

        # Calculamos los cuartiles
        quartiles = np.percentile(self.data, [25, 50, 75])

        self.metrics = {
            "median": median,
            "variance": variance,
            "mean": mean,
            "mode": mode_value[0] if mode_value.size > 0 else None,
            "25th_percentile": quartiles[0],
            "50th_percentile": quartiles[1],
            "75th_percentile": quartiles[2],
        }

    def write_metrics(self):
        with open(self.path + "metrics.txt", "w") as file:
            for key, value in self.metrics.items():
                file.write(f"{key}: {value}\n")

    def plot_density_per_variable(self):
        # Verificar que ambos conjuntos de datos son DataFrames y tienen las mismas columnas
        if not (
            isinstance(self.data, pd.DataFrame)
            and isinstance(self.synthetic_data, pd.DataFrame)
        ):
            raise ValueError(
                "Data should be pandas DataFrame for both original and synthetic data."
            )

        if set(self.data.columns) != set(self.synthetic_data.columns):
            raise ValueError("Both data sets must have the same variables (columns).")

        # Generar un gráfico de densidad para cada columna
        for column in self.data.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(
                self.data[column], ax=ax, fill=True, label="Original", bw_adjust=2
            )
            sns.kdeplot(
                self.synthetic_data[column],
                ax=ax,
                fill=True,
                label="Synthetic",
                bw_adjust=2,
                warn_singular=False,
            )

            # Calcular estadísticas descriptivas para ambos conjuntos de datos
            median_orig = self.data[column].median()
            mean_orig = self.data[column].mean()
            median_syn = self.synthetic_data[column].median()
            mean_syn = self.synthetic_data[column].mean()

            # Añadir líneas verticales para la media y la mediana de ambos conjuntos de datos
            ax.axvline(
                median_orig,
                color="red",
                linestyle="-",
                label=f"Original Median: {median_orig:.2f}",
            )
            ax.axvline(
                mean_orig,
                color="green",
                linestyle="--",
                label=f"Original Mean: {mean_orig:.2f}",
            )
            ax.axvline(
                median_syn,
                color="purple",
                linestyle="-",
                label=f"Synthetic Median: {median_syn:.2f}",
            )
            ax.axvline(
                mean_syn,
                color="orange",
                linestyle="--",
                label=f"Synthetic Mean: {mean_syn:.2f}",
            )

            # Configurar título y leyendas
            ax.set_title(f"Density Plot of {column}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            plt.legend()

            plt.tight_layout()  # Ajustar automáticamente los subplots para que encajen en la figura
            plt.savefig(self.path + "density_plots/" + f"{column}_density.png")
            plt.close()

    def plot_statistics(self):
        # Crear un boxplot para cada atributo
        for i, column in enumerate(self.data.columns, 1):
            fig, ax = plt.subplots()
            self.data.boxplot(column, ax=ax)

            # Obtener estadísticas descriptivas
            median = self.data[column].median()
            mean = self.data[column].mean()
            std = self.data[column].std()

            # Añadir un texto para la desviación estándar
            # Añadir líneas horizontales para la media y la mediana
            ax.axhline(
                median, color="red", linestyle="-", label=f"Median: {median:.2f}"
            )
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
            plt.savefig(self.path + "boxplot/" + f"{column}.png")
            # plt.show()
            plt.close()

    def execute(self):
        self.calculate_metrics()
        self.write_metrics()
        self.plot_density_per_variable()
        self.plot_statistics()
