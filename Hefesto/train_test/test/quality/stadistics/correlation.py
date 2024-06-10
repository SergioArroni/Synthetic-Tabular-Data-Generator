import pandas as pd
import matplotlib.pyplot as plt
from Hefesto.train_test.test.quality.stadistics import Stadistics


class Correlation(Stadistics):
    def __init__(self, data):
        super().__init__(data)

    def matrix_correlation(self, name: str):
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        corr = self.data.corr()
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
        plt.savefig(f"./img/correlation/correlation_matrix_{name}.png")
        # plt.show()
