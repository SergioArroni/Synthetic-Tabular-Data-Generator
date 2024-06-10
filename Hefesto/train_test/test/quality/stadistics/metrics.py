import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from Hefesto.train_test.test.quality.stadistics import Stadistics

class Metrics(Stadistics):
    def __init__(self, data, path: str):
        super().__init__(data=data, path=path)
        self.metrics = None

    def calculate_metrics(self):
        # Calculamos la mediana
        median = np.median(self.data)

        # Calculamos la varianza
        variance = np.var(self.data)

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
            "75th_percentile": quartiles[2]
        }

    def write_metrics(self):
        with open(self.path, "w") as file:
            for key, value in self.metrics.items():
                file.write(f"{key}: {value}\n")

    def plot_metrics(self):
        if self.metrics is None:
            raise ValueError("Metrics have not been calculated. Call calculate_metrics() first.")

        # Crear un gráfico de barras para las métricas
        plt.figure(figsize=(10, 6))
        metrics_names = list(self.metrics.keys())
        metrics_values = list(self.metrics.values())

        plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow'])
        plt.title('Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.path + "_metrics.png")
        plt.close()

        # Crear un boxplot para los datos
        plt.figure(figsize=(10, 6))
        plt.boxplot(self.data, vert=False)
        plt.title('Boxplot of Data')
        plt.xlabel('Value')
        plt.savefig(self.path + "_boxplot.png")
        plt.close()

        # Crear un gráfico de densidad para los datos
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.data, fill=True)
        plt.title('Density Plot of Data')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.savefig(self.path + "_density.png")
        plt.close()

    def execute(self):
        self.calculate_metrics()
        self.write_metrics()
        # self.plot_metrics() TODO
