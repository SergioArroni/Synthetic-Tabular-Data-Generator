from Hefesto.train_test.test.utility.efficiency import Effiency


class TSTR(Effiency):
    def __init__(self, df, df_test, seed, path):
        self.df = df
        self.df_test = df_test
        super().__init__(seed=seed, path=path)

    def process(self):
        self.X_train = self.df.drop("cardio", axis=1)
        self.y_train = self.df["cardio"]
        self.X_test = self.df_test.drop("cardio", axis=1)
        self.y_test = self.df_test["cardio"]
