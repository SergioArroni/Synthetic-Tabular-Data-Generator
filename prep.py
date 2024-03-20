import torch

from Hefesto.preprocess.load_data import read_data, split_data
from Hefesto.preprocess.correlations import matrix_correlation
from Hefesto.utils.utils import plot_statistics
from Hefesto.preprocess.preprocess import Preprocess


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    df = read_data("data/cardio/cardio_train.csv")
    df = df.drop("id", axis=1)
    plot_statistics(df, f"./img/stadistics/cardio/bruto/boxplot")
    # prep = Preprocess(df)
    # prep.scaler_method()
    # df = prep.df
    
    matrix_correlation(df)
    # plot_statistics(df, f"./img/stadistics/cardio/standar/boxplot")

    df["new_column"] = 0

    n = 5000
    m = 5000
    v = 5000

    df_train, df_test, df_val = split_data(df, n, m, v)
    df_train.to_csv("data/cardio/split/cardio_train.csv", sep=";", index=False)
    df_test.to_csv("data/cardio/split/cardio_test.csv", sep=";", index=False)
    df_val.to_csv("data/cardio/split/cardio_val.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
