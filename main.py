import torch
from Hefesto.utils.preprocess import (
    preprocess_data,
    read_data,
    split_data,
    do_data_loader,
)
from Hefesto.utils.utils import load_model, write_results, save_model
from Hefesto.train_test.train import Train
from Hefesto.train_test.test import Test
from Hefesto.models.diffusion.diffusion import DiffusionModel
from Hefesto.models.GAN.GAN import GANModel
from Hefesto.models.transformers.transformers import TransformersModel
from Hefesto.models.transformers.pre_transformers import PreTransformersModel
from Hefesto.models.VAE.VAE import VAEModel


def main():
    seed = 42
    df = read_data("data\cardio\cardio_train.csv")
    df = preprocess_data(df, seed)

    n = 10000
    m = 10000
    v = 10000

    df_train, df_test, df_val = split_data(
        df, n, m, v
    )  # Assume preprocessed_df is your preprocessed DataFrame
    train_loader = do_data_loader(df_train)
    test_loader = do_data_loader(df_test)
    val_loader = do_data_loader(df_val)

    epochs = 2
    T = 200
    betas = torch.linspace(0.001, 0.2, T)
    tolerance = 0.001
    n_transformers = 2
    input_dim = train_loader.dataset.features.shape[1]
    hidden_dim = 128

    model = DiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        T=T,
        betas=betas,
    )
    # model = VAEModel(
    #     input_dim=train_loader.dataset.features.shape[1], hidden_dim=128, n_steps=20
    # )
    # model = PreTransformersModel(
    #     input_dim=train_loader.dataset.features.shape[1], hidden_dim=128, n_steps=20
    # )
    # model = GANModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = Train(model, device)

    save_model("./save_models/model.pt", train.model)

    train = load_model("./save_models/model.pt", model)

    # train.train_model(train_loader, epochs, T, betas)

    test = Test(train, test_loader, val_loader, seed)
    good_ele, bad_ele = test.evaluate_model()

    write_results(epochs, good_ele, bad_ele, "./results/results.txt", m, model, seed)


if __name__ == "__main__":
    main()
