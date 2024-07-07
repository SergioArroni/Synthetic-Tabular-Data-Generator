import torch
import pandas as pd
from Hefesto.preprocess.load_data import read_data, split_data
from Hefesto.preprocess.preprocess import Preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PrepExe:
    def __init__(
        self,
        device,
        cant_train: int,
        cant_test: int,
        cant_val: int,
        seed: int = 42,
        path_train: str = "data/cardio/cardio_train.csv",
    ):
        self.device = device
        self.seed = seed
        self.path_train = path_train
        self.df = read_data(self.path_train)
        self.df = self.df.drop("id", axis=1)
        self.preprocess = Preprocess(df=self.df, seed=self.seed)
        self.cant_train = cant_train
        self.cant_test = cant_test
        self.cant_val = cant_val

    def prep_exe(self):
        torch.cuda.set_device(self.device)

        if self.hard_prep:
            self.preprocess.quantile_transform()
            self.df = pd.DataFrame(
                self.preprocess.transformed_data, columns=self.df.columns
            )

        n = self.cant_train
        m = self.cant_test
        v = self.cant_val

        df_train, df_test, df_val = split_data(self.df, n, m, v)
        df_train.to_csv("data/cardio/split/cardio_train.csv", sep=";", index=False)
        df_test.to_csv("data/cardio/split/cardio_test.csv", sep=";", index=False)
        df_val.to_csv("data/cardio/split/cardio_val.csv", sep=";", index=False)

        # matrix_correlation(df, "val")

        X = df_val.drop("cardio", axis=1)
        y = df_val["cardio"]
