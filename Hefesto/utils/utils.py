import torch
from Hefesto.models.model import Model


def save_model(path: str, model: Model) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, shape: int, train_model: Model) -> Model:
    model = train_model(shape, 128, 10)
    model.load_state_dict(torch.load(path))
    return model


def write_results(epochs, df_test, x_gen, good_ele, bad_ele, path: str):
    with open(path, "a") as a:
        a.write(f"Epochs: \n{epochs}\n")
        a.write(f"Array In: \n{df_test.iloc[0].values}\n")
        a.write(f"Array Gen: \n{x_gen.detach().numpy()}\n")
        a.write(f"Good Data Gen: {len(good_ele)}\n")
        a.write(f"Bad Data Gen: {len(bad_ele)}\n")
        a.write(f"Acierto: {(len(good_ele)/df_test.shape[0])*100}%\n")
        a.write("---------------------------------------------------------------\n")
