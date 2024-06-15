import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, entropy
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Tests(Stadistics):
    """Class to perform statistical tests between two datasets."""

    def __init__(self, data, data2, path: str):
        super().__init__(data=data, path=path)
        self.data2 = data2
        self.statistic = None
        self.p_value = None
        self.kl_divergence = None

    def test_ks(self):
        """Perform the Kolmogorov-Smirnov test."""
        self.statistic, self.p_value = ks_2samp(self.data, self.data2)

    def test_kl(self, base=None):
        """Calculate the Kullback-Leibler divergence, ensuring no zero probabilities."""
        data_normalized = np.asarray(self.data, dtype=np.float64) + 1e-10
        data2_normalized = np.asarray(self.data2, dtype=np.float64) + 1e-10

        # Normalize if they represent probabilities
        data_normalized /= data_normalized.sum()
        data2_normalized /= data2_normalized.sum()

        # Verificar si los arrays pueden ser transmitidos
        try:
            np.broadcast_shapes(data_normalized.shape, data2_normalized.shape)
        except ValueError as e:
            print("Error de transmisión:", e)
            # Redimensionar los arrays para hacerlos transmisibles, por ejemplo:
            data_normalized = data_normalized.reshape(
                -1, 1
            )  # Ejemplo de redimensionamiento
            data2_normalized = data2_normalized.reshape(
                -1, 1
            )  # Ejemplo de redimensionamiento

        # Continuar con el cálculo de la entropía
        self.kl_divergence = entropy(data_normalized, data2_normalized, base=base)

    def write_tests(self):
        """Write test results to a file."""
        try:
            with open(self.path, "w") as file:
                file.write(f"KS statistic: {self.statistic}\n")
                file.write(f"KS p-value: {self.p_value}\n")
                file.write(f"KL divergence: {self.kl_divergence}\n")
        except IOError as e:
            print(f"Error writing to file: {e}")

    def execute(self):
        """Execute all tests and output handling."""
        self.test_ks()
        self.test_kl()
        self.write_tests()
