from Hefesto.train_test.test.utility import Utility


class TSTR(Utility):
    def __init__(self, df, df_test, seed, file_name="./new_results/utility/TSTR.txt"):
        self.df = df
        self.df_test = df_test
        super().__init__(seed=seed, file_name=file_name)

    def process(self):
        self.X_train = self.df.drop("cardio", axis=1)
        self.y_train = self.df["cardio"]
        self.X_test = self.df_test.drop("cardio", axis=1)
        self.y_test = self.df_test["cardio"]
