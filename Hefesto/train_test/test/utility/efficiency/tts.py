from Hefesto.train_test.test.utility.efficiency import Effiency


class TTS(Effiency):
    def __init__(self, df, df_test, seed, path):
        self.df = df
        self.df_test = df_test
        self.seed = seed
        self.path = path
        self.metrics_gen = None
        self.metrics_test = None

    def print_result(self):
        # Compare the two metrics in a file
        with open(self.path, "w") as f:
            f.write("f1_gen, accuracy_gen, f1_test, accuracy_test\n")
            f.write(
                f"{self.metrics_gen[0]}, {self.metrics_gen[1]}, {self.metrics_test[0]}, {self.metrics_test[1]}\n"
            )

    def _exute_TTS(self):
        X = self.df.drop("cardio", axis=1)
        y = self.df["cardio"]
        super().__init__(X=X, y=y, seed=self.seed, path=self.path)
        self.process()
        self.run()
        self.metrics_gen = self.result

    def _exute_TTR(self):
        X = self.df_test.drop("cardio", axis=1)
        y = self.df_test["cardio"]
        super().__init__(X=X, y=y, seed=self.seed, path=self.path)
        self.process()
        self.run()
        self.metrics_test = self.result

    def execute(self):
        self._exute_TTS()
        self._exute_TTR()
        self.print_result()
