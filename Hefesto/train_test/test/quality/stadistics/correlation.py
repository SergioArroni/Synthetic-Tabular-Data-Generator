import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Correlation(Stadistics):
    def __init__(self, original_data, synthetic_data, path: str):
        super().__init__(
            original_data=original_data, synthetic_data=synthetic_data, path=path
        )

    def matrix_correlation(self, df: pd.DataFrame, type_data: str):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            type_data (str): _description_

        Returns:
            _type_: _description_
        """
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(corr, cmap="coolwarm")

        # Agregar los números a la matriz de correlación
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(
                    j,
                    i,
                    round(corr.iloc[i, j], 2),
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.colorbar(cax)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.savefig(self.path + type_data)
        # plt.show()

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
        self.standardize_data()
        self.matrix_correlation(self.original_data, "_original.png")
        self.matrix_correlation(self.synthetic_data, "_synthetic.png")
