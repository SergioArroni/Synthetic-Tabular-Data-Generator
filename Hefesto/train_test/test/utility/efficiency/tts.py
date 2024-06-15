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
            f.write("Metrics for the generated data\n")
            f.write(f"F1: {self.metrics_gen[0]}\n")
            f.write(f"Accuracy: {self.metrics_gen[1]}\n")
            f.write(f"Recall: {self.metrics_gen[2]}\n")
            f.write(f"Precision: {self.metrics_gen[3]}\n")
            f.write(f"ROC: {self.metrics_gen[4]}\n")
            f.write("\n")
            f.write("Metrics for the test data\n")
            f.write(f"F1: {self.metrics_test[0]}\n")
            f.write(f"Accuracy: {self.metrics_test[1]}\n")
            f.write(f"Recall: {self.metrics_test[2]}\n")
            f.write(f"Precision: {self.metrics_test[3]}\n")
            f.write(f"ROC: {self.metrics_test[4]}\n")
            

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
