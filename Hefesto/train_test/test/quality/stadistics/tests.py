import numpy as np
from scipy.stats import ks_2samp, entropy
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Tests(Stadistics):
    """Clase para realizar pruebas estadísticas entre dos conjuntos de datos."""

    def __init__(self, data, data2):
        super().__init__(data)
        self.data2 = data2

    def test_ks(self):
        # Realizamos el test de Kolmogorov-Smirnov
        statistic, p_value = ks_2samp(self.data, self.data2)
        return statistic, p_value

    def test_kl(self, base=None):
        # Calculamos la divergencia Kullback-Leibler
        # Agregamos una pequeña cantidad a los datos para evitar problemas de log(0)
        data_normalized = np.asarray(self.data, dtype=np.float64) + 1e-10
        data2_normalized = np.asarray(self.data2, dtype=np.float64) + 1e-10

        kl_divergence = entropy(data_normalized, data2_normalized, base=base)
        return kl_divergence
