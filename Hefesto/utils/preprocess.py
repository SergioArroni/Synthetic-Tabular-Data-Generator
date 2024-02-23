from numpy import float32
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TabularDataset(Dataset):
    def __init__(self, features, labels, columns=None):
        self.features = features
        self.labels = labels
        self.columns = columns

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def do_data_loader(df, batch_size, columns, seed=0, shuffle=True):
    """_summary_

    Args:
        df (_type_): _description_
        batch_size (_type_): _description_
        columns (_type_): _description_
        seed (int, optional): _description_. Defaults to 0.
        shuffle (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    df = preprocess_data(df, seed)
    features, labels = df_to_tensor(df)
    dataset = TabularDataset(features, labels, columns)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def df_to_tensor(df):
    # Assuming the last column is the target variable
    features = df.values
    labels = df.iloc[:, -1].values
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return features_tensor, labels_tensor


def read_data(file_path):
    return pd.read_csv(file_path, sep=";")


def preprocess_data(df, seed):
    df = df.astype(float32)
    df = df.sample(frac=1, random_state=seed)
    return df


def split_data(df, n, m, v):
    df_train = df.iloc[:n]
    df_test = df.iloc[n : n + m]
    df_val = df.iloc[n + m : n + m + v]
    return df_train, df_test, df_val
