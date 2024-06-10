from Hefesto.train_test.test.utility.efficiency import Effiency


class TRTS(Effiency):
    """Clase para evaluar la eficiencia de un modelo usando TRTS."""
    def __init__(self, df, df_test, seed, file_name="./new_results/utility/TRTS.txt"):
        self.df = df
        self.df_test = df_test
        super().__init__(seed=seed, file_name=file_name)

    def process(self):
        self.X_train = self.df_test.drop("cardio", axis=1)
        self.y_train = self.df_test["cardio"]
        self.X_test = self.df.drop("cardio", axis=1)
        self.y_test = self.df["cardio"]
