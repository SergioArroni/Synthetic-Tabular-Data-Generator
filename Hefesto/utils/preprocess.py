from numpy import float32
import pandas as pd


def read_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, seed):
    df = df.astype(float32)
    df = df.sample(frac=1, random_state=seed)
    return df


def split_data(df, n, m, v):
    df_train = df.iloc[:n]
    df_test = df.iloc[n : n + m]
    df_val = df.iloc[n + m : n + m + v]
    return df_train, df_test, df_val
