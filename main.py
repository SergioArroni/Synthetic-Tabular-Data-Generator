import time

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from Hefesto.models.VAE.VAE import VAEModel

from Hefesto.models.diffusion.diffusion import DiffusionModel
from Hefesto.models.GAN.GAN import GANModel
from Hefesto.models.transformers.transformers import TransformerModel
from Hefesto.train_test.test import Test
from Hefesto.train_test.train import Train
from Hefesto.preprocess.load_data import do_data_loader, read_data, split_data
from Hefesto.preprocess.correlations import matrix_correlation
from Hefesto.utils.utils import load_model, plot_statistics, save_model, write_results


def main():
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    bach_size = 256

    df_test = read_data("data/cardio/split/cardio_test.csv")
    df_train = read_data("data/cardio/split/cardio_train.csv")
    df_val = read_data("data/cardio/split/cardio_val.csv")
    columnas = df_test.columns
    size = len(df_test)

    train_loader: DataLoader = do_data_loader(df_train, bach_size, columnas, seed=seed)
    test_loader: DataLoader = do_data_loader(df_test, bach_size, columnas, seed=seed)
    val_loader: DataLoader = do_data_loader(df_val, bach_size, columnas, seed=seed)

    epochs = 200
    T = 200
    betas = torch.linspace(0.1, 0.9, T)
    input_dim = train_loader.dataset.features.shape[1]
    hidden_dim = 128
    timestamp = time.time()
    alpha = 0.5

    model = DiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        T=T,
        dropout=0.4,
        device=device,
        alpha=alpha,
        seed=seed,
        betas=betas,
    )
    # model = VAEModel(
    #     input_dim=train_loader.dataset.features.shape[1],
    #     hidden_dim=128,
    #     latent_dim=2,
    #     device=device,
    # )
    # model = GANModel(
    #     input_dim=input_dim, hidden_dim=hidden_dim, device=device
    # )
    # model = TransformerModel(
    #     input_dim=input_dim, hidden_dim=hidden_dim
    # )

    train = Train(model, device, timestamp, epochs)

    # model = load_model(
    #     "./save_models/model_DiffusionModel_1709810144.782568.pt", model
    # )

    train.train_model(train_loader, val_loader)

    if train is Train:
        model = train.model

    save_model(f"./save_models/model_{model}_{timestamp}.pt", model)

    test = Test(model, test_loader, val_loader, seed, device)
    good_ele, bad_ele = test.evaluate_model()

    write_results(epochs, good_ele, bad_ele, "./results/results.txt", size, model, seed)


if __name__ == "__main__":
    main()
