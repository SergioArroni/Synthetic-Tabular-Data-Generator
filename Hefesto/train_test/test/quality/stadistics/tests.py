import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, entropy
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Tests(Stadistics):
    """Clase para realizar pruebas estadísticas entre dos conjuntos de datos."""

    def __init__(self, data, data2, path: str):
        super().__init__(data=data, path=path)
        self.data2 = data2
        self.statistic = None
        self.p_value = None
        self.kl_divergence = None

    def test_ks(self):
        # Realizamos el test de Kolmogorov-Smirnov
        self.statistic, self.p_value = ks_2samp(self.data, self.data2)

    def test_kl(self, base=None):
        # Calculamos la divergencia Kullback-Leibler
        # Agregamos una pequeña cantidad a los datos para evitar problemas de log(0)
        data_normalized = np.asarray(self.data, dtype=np.float64) + 1e-10
        data2_normalized = np.asarray(self.data2, dtype=np.float64) + 1e-10

        self.kl_divergence = entropy(data_normalized, data2_normalized, base=base)

    def write_tests(self):
        with open(self.path, "w") as file:
            file.write(f"KS statistic: {self.statistic}\n")
            file.write(f"KS p-value: {self.p_value}\n")
            file.write(f"KL divergence: {self.kl_divergence}\n")

    def plot_density(self):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.data, label="Data", color="blue", fill=True)
        sns.kdeplot(self.data2, label="Data2", color="red", fill=True)
        plt.title("Kernel Density Estimate")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"{self.path}_density.png")
        plt.close()

    def execute(self):
        self.test_ks()
        self.test_kl()
        self.write_tests()
        self.plot_density()
