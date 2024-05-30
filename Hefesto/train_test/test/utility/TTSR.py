import pandas as pd

from Hefesto.train_test.test.utility import Utility


class TTSR(Utility):
    def __init__(self, df, df_test, seed, file_name="./new_results/utility/TTSR.txt"):
        df_cocktel = pd.concat([df, df_test], axis=0)
        X = df_cocktel.drop("cardio", axis=1)
        y = df_cocktel["cardio"]
        super().__init__(X=X, y=y, seed=seed, file_name=file_name)
