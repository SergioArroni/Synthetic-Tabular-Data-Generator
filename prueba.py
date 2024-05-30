import torch
from Hefesto.utils.preprocess import (
    preprocess_data,
    read_data,
    split_data,
    do_data_loader,
)
from Hefesto.utils.utils import load_model, write_results
from TFM.Hefesto.train_test.train.train import Train
from TFM.Hefesto.train_test.test.test import Test
from Hefesto.models.diffusion.diffusion import DiffusionModel
from Hefesto.models.diffusion.diffusion_v2 import DiffusionModel as DiffusionModelV2
from Hefesto.models.GAN.GAN import GANModel
from Hefesto.models.transformers.transformers import TransformersModel
from Hefesto.models.transformers.pre_transformers import PreTransformersModel
from Hefesto.models.VAE.VAE import VAEModel

from sklearn.ensemble import IsolationForest


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

    epochs = 100
    n_steps = 200
    tolerance = 0.001
    data_dim = train_loader.dataset.features.shape[1]
    data_dim_0 = train_loader.dataset.features.shape[0]

    # Configuración del modelo
    T = 1000  # Número total de pasos de difusión
    betas = torch.linspace(0.0001, 0.02, T)  # Ejemplo de secuencia de betas

    # Inicialización del modelo
    diffusion_model = DiffusionModelV2(data_dim, T, betas)

    # Aplica el proceso de difusión a los datos de entrada
    x_T = diffusion_model.forward_diffusion_process(train_loader.dataset.features[:1000])

    # Intenta reconstruir los datos originales a partir del ruido
    x_reconstructed = diffusion_model.reverse_diffusion_process(x_T)
    
    print(f"Reconstrucción: {torch.mean((x_reconstructed - train_loader.dataset.features[:1000])**2)}\n")
    print(f"Original: {torch.mean((train_loader.dataset.features[:1000])**2)}\n")

    clf = IsolationForest(random_state=seed).fit(
        train_loader.dataset.features
    )
    
    good_ele = []
    bad_ele = []

    for ele in x_reconstructed:
        if clf.predict([ele.detach().numpy()]) == 1:
            good_ele.append(ele)
        else:
            bad_ele.append(ele)
            
    print(f"Acierto: {(len(good_ele)/data_dim_0)*100}%\n")


if __name__ == "__main__":
    main()
