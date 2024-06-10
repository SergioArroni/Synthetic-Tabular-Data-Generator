import numpy as np
from scipy.stats import mode
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Metrics(Stadistics):
    def __init__(self, data):
        super().__init__(data)

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

        return {
            "median": median,
            "variance": variance,
            "mean": mean,
            "mode": mode_value[0] if mode_value.size > 0 else None,
            "quartiles": quartiles,
        }
