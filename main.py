import time
from torchviz import make_dot
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from Hefesto.preprocess import PrepExe
from Hefesto.models.diffusion.diffusion import DiffusionModel
from Hefesto.train_test import Test
from Hefesto.train_test import Train
from Hefesto.preprocess.load_data import do_data_loader, read_data, split_data
from Hefesto.utils.utils import load_model, save_model, write_results


def main():
    seed = 42
    load = True
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    prep = True
    hard_prep = False
    view = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    bach_size = 256
    prep_exe = None
    if prep:
        prep_exe = PrepExe(
            device=device,
            seed=seed,
            cant_train=10000,
            cant_test=10000,
            cant_val=10000,
            hard_prep=hard_prep,
        )
        prep_exe.prep_exe()

    df_test = read_data("data/cardio/split/cardio_test.csv")
    df_train = read_data("data/cardio/split/cardio_train.csv")
    df_val = read_data("data/cardio/split/cardio_val.csv")
    columnas = df_test.columns
    size = len(df_test)

    train_loader: DataLoader = do_data_loader(df_train, bach_size, columnas, seed=seed)
    test_loader: DataLoader = do_data_loader(df_test, bach_size, columnas, seed=seed)
    val_loader: DataLoader = do_data_loader(df_val, bach_size, columnas, seed=seed)

    epochs = 300
    T = 400
    # betas = torch.linspace(0.1, 0.9, T)
    # Crear una secuencia de betas donde el incremento no es lineal
    initial_beta = 0.1
    final_beta = 0.9

    # Usar una función exponencial para suavizar la introducción del ruido
    exponential_growth = torch.logspace(
        torch.log10(torch.tensor(initial_beta)),
        torch.log10(torch.tensor(final_beta)),
        steps=T,
    )

    betas = exponential_growth
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

    if load:
        model = load_model(
            "./save_models/model_DiffusionModel_1718476898.2937098.pt",
            model,
        )
    else:
        train = Train(model, device, timestamp, epochs)

        train.train_model(train_loader, val_loader)

        model = train.model

        save_model(f"./save_models/model_{model}_{timestamp}.pt", model)

    # Visualization code
    if view:
        sample_input = torch.randn(1, input_dim).to(device)  # Adjust the shape according to your model's requirement
        y = model(sample_input)
        dot = make_dot(y, params=dict(list(model.named_parameters()) + [('input', sample_input)]))
        dot.render('model_architecture', format='png', view=True)

    test = Test(
        model=model,
        test_loader=test_loader,
        val_loader=val_loader,
        device=device,
        preprocess=prep_exe,
        seed=seed,
        hard_prep=hard_prep,
    )
    test.evaluate_model()


if __name__ == "__main__":
    main()
