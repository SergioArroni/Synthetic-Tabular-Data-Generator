import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Metrics(Stadistics):
    def __init__(self, original_data, synthetic_data, path: str):
        super().__init__(
            original_data=original_data, synthetic_data=synthetic_data, path=path
        )

    def plot_density_per_variable(self):
        # Verificar que ambos conjuntos de datos son DataFrames y tienen las mismas columnas
        if not (
            isinstance(self.original_data, pd.DataFrame)
            and isinstance(self.synthetic_data, pd.DataFrame)
        ):
            raise ValueError(
                "Data should be pandas DataFrame for both original and synthetic data."
            )

        if set(self.original_data.columns) != set(self.synthetic_data.columns):
            raise ValueError("Both data sets must have the same variables (columns).")

        # Generar un gráfico de densidad para cada columna
        for column in self.original_data.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(
                self.original_data[column],
                ax=ax,
                fill=True,
                label="Original",
                bw_adjust=2,
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
            median_orig = self.original_data[column].median()
            mean_orig = self.original_data[column].mean()
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

    def standardize_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # Standardize data
        pass

    def plot_statistics(self):
        # Crear un boxplot para cada atributo
        for i, column in enumerate(self.original_data.columns, 1):
            fig, ax = plt.subplots()
            # print(self.original_data.head())
            self.original_data.boxplot(column, ax=ax)  # Plot synthetic data

            # Obtener estadísticas descriptivas para ambos conjuntos de datos
            median_orig = self.original_data[column].median()
            mean_orig = self.original_data[column].mean()
            std_orig = self.original_data[column].std()
            median_syn = self.synthetic_data[column].median()
            mean_syn = self.synthetic_data[column].mean()
            std_syn = self.synthetic_data[column].std()

            # Añadir un texto para la desviación estándar de los datos sintéticos
            ax.text(
                0.95,
                0.01,
                f"Synthetic Std: {std_syn:.2f}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
                color="blue",
                fontsize=10,
            )

            # Añadir líneas horizontales para la media y la mediana de ambos conjuntos de datos
            ax.axhline(
                median_orig,
                color="red",
                linestyle="-",
                label=f"Original Median: {median_orig:.2f}",
            )
            ax.axhline(
                mean_orig,
                color="green",
                linestyle="--",
                label=f"Original Mean: {mean_orig:.2f}",
            )
            ax.axhline(
                median_syn,
                color="purple",
                linestyle="-",
                label=f"Synthetic Median: {median_syn:.2f}",
            )
            ax.axhline(
                mean_syn,
                color="orange",
                linestyle="--",
                label=f"Synthetic Mean: {mean_syn:.2f}",
            )

            # Mostrar la leyenda
            plt.legend()

            plt.tight_layout()  # Ajustar automáticamente los subplots para que encajen en la figura
            plt.savefig(self.path + "boxplot/" + f"{column}.png")
            plt.close()

    def execute(self):
        self.standardize_data()
        self.plot_density_per_variable()
        self.plot_statistics()
