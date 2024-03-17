import time

import torch
from torch.utils.data import DataLoader

from Hefesto.preprocess.preprocess import do_data_loader, read_data, split_data
from Hefesto.preprocess.correlations import matrix_correlation
from Hefesto.utils.utils import plot_statistics

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    df = read_data("data/cardio/cardio_train.csv")
    matrix_correlation(df)
    df = df.drop("id", axis=1)
    df["new_column"] = 0

    n = 20000
    m = 20000
    v = 20000

    df_train, df_test, df_val = split_data(df, n, m, v)
    df_train.to_csv("data/cardio/split/cardio_train.csv", sep=";", index=False)
    df_test.to_csv("data/cardio/split/cardio_test.csv", sep=";", index=False)
    df_val.to_csv("data/cardio/split/cardio_val.csv", sep=";", index=False)



if __name__ == "__main__":
    main()
