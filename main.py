import time

import torch
from torch.utils.data import DataLoader

from Hefesto.models.diffusion.diffusion import DiffusionModel
from Hefesto.models.GAN.GAN import GANModel
from Hefesto.models.transformers.pre_transformers import PreTransformersModel
from Hefesto.models.transformers.transformers import TransformersModel
from Hefesto.models.VAE.VAE import VAEModel
from Hefesto.train_test.test import Test
from Hefesto.train_test.train import Train
from Hefesto.utils.preprocess import do_data_loader, read_data, split_data
from Hefesto.utils.utils import load_model, plot_statistics, save_model, write_results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    seed = 0
    bach_size = 256
    df = read_data("data/cardio/cardio_train.csv")
    df = df.drop("id", axis=1)
    df["new_column"] = 0
    # print(df.head())
    plot_statistics(df, "./img/stadistics/cardio/boxplot")
    columnas = df.columns

    n = 5000
    m = 5000
    v = 5000

    df_train, df_test, df_val = split_data(df, n, m, v)
    df_train.to_csv("data/cardio/split/cardio_train.csv", sep=";", index=False)
    df_test.to_csv("data/cardio/split/cardio_test.csv", sep=";", index=False)
    df_val.to_csv("data/cardio/split/cardio_val.csv", sep=";", index=False)
    train_loader: DataLoader = do_data_loader(df_train, bach_size, columnas)
    test_loader: DataLoader = do_data_loader(df_test, bach_size, columnas)
    val_loader: DataLoader = do_data_loader(df_val, bach_size, columnas)

    epochs = 200
    T = 200
    betas = torch.linspace(0.1, 0.9, T)
    tolerance = 0.001
    n_transformers = 2
    input_dim = train_loader.dataset.features.shape[1]
    hidden_dim = 128
    timestamp = time.time()

    model = DiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        T=T,
        betas=betas,
        device=device,
    )
    # model = VAEModel(
    #     input_dim=train_loader.dataset.features.shape[1], hidden_dim=128
    # )
    # model = PreTransformersModel(
    #     input_dim=train_loader.dataset.features.shape[1], hidden_dim=128, n_steps=20
    # )
    # model = GANModel

    train = Train(model, device, timestamp)

    # model = load_model(
    #     "./save_models/model_DiffusionModel_1709810144.782568.pt", model
    # )

    train.train_model(train_loader, val_loader, epochs)

    if train is Train:
        model = train.model

    save_model(f"./save_models/model_{model}_{timestamp}.pt", model)

    test = Test(model, test_loader, val_loader, seed, device)
    good_ele, bad_ele = test.evaluate_model()

    write_results(epochs, good_ele, bad_ele, "./results/results.txt", m, model, seed)


if __name__ == "__main__":
    main()
