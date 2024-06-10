import pandas as pd

from Hefesto.train_test.test.utility.efficiency import Effiency


class TTSR(Effiency):
    def __init__(self, df, df_test, seed, path):
        df_cocktel = pd.concat([df, df_test], axis=0)
        X = df_cocktel.drop("cardio", axis=1)
        y = df_cocktel["cardio"]
        super().__init__(X=X, y=y, seed=seed, path=path)
