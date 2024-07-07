from sklearn.preprocessing import QuantileTransformer
import pandas as pd


class Stadistics:
    """Clase base para realizar pruebas estadísticas entre dos conjuntos de datos."""

    def __init__(self, original_data, synthetic_data, path: str):
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.original_data_all = None
        self.synthetic_data_all = None
        self.path = path

    def standardize_data(self, all_data: bool = True):
        self.original_data_all = self.original_data
        self.synthetic_data_all = self.synthetic_data
        column_names = self.original_data.columns
        transformer = QuantileTransformer(output_distribution="normal")
        self.original_data = transformer.fit_transform(self.original_data)
        self.synthetic_data = transformer.transform(self.synthetic_data)
        self.original_data = pd.DataFrame(self.original_data, columns=column_names)
        self.synthetic_data = pd.DataFrame(self.synthetic_data, columns=column_names)
        print("Data has been standardized.")
    

    def execute(self):
        raise NotImplementedError
