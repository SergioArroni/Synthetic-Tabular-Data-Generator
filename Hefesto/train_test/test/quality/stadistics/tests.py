import numpy as np
from scipy.stats import entropy, ks_2samp
from Hefesto.train_test.test.quality.stadistics import Stadistics
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Tests(Stadistics):
    """Class to perform statistical tests between two datasets."""

    def __init__(self, original_data, synthetic_data, path: str, all_data: bool):
        super().__init__(
            original_data=original_data, synthetic_data=synthetic_data, path=path
        )
        self.statistic = None
        self.p_value = None
        self.kl_divergence = None
        self.all_data = all_data

    def test_ks(self):
        """Perform the Kolmogorov-Smirnov test."""
        if self.all_data:
            self.statistic, self.p_value = ks_2samp(
                self.original_data.values.flatten(),
                self.synthetic_data.values.flatten(),
            )
        else:
            self.statistic, self.p_value = ks_2samp(
                self.original_data, self.synthetic_data
            )

    def obtener_histogramas_normalizados(self, datos, num_bins=50):
        histogramas = {}
        for columna in datos.columns:
            hist, bin_edges = np.histogram(datos[columna], bins=num_bins, density=True)
            hist += 1e-10  # Añadir un valor pequeño para evitar ceros
            histogramas[columna] = hist
        return histogramas

    def test_kl(self, base=None):
        """Calculate the KL divergence for the entire dataset."""
        if self.all_data:
            histogramas_orig = self.obtener_histogramas_normalizados(self.original_data)
            histogramas_sint = self.obtener_histogramas_normalizados(
                self.synthetic_data
            )

            kl_divergencias = []
            for columna in histogramas_orig.keys():
                kl_div = entropy(histogramas_orig[columna], histogramas_sint[columna])
                kl_divergencias.append(kl_div)

            self.kl_divergence = np.mean(kl_divergencias)
        else:
            histogramas_orig = self.obtener_histogramas_normalizados(self.original_data)
            histogramas_sint = self.obtener_histogramas_normalizados(
                self.synthetic_data
            )
            kl_divergencias = []
            for columna in histogramas_orig.keys():
                kl_div = entropy(histogramas_orig[columna], histogramas_sint[columna])
                kl_divergencias.append(kl_div)
            self.kl_divergence = kl_divergencias

    def write_tests(self):
        """Write test results to a file."""
        try:
            with open(self.path, "w") as file:
                file.write(f"KS statistic: {self.statistic}\n")
                file.write(f"KL divergence: {self.kl_divergence}\n")
        except IOError as e:
            print(f"Error writing to file: {e}")

    def standardize_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # Standardize data
        self.original_data = pd.DataFrame(
            StandardScaler().fit_transform(self.original_data),
            columns=self.original_data.columns,
        )
        self.synthetic_data = pd.DataFrame(
            StandardScaler().fit_transform(self.synthetic_data),
            columns=self.synthetic_data.columns,
        )

    def execute(self):
        """Execute all tests and output handling."""
        self.standardize_data()
        self.test_ks()
        self.test_kl()
        self.write_tests()
