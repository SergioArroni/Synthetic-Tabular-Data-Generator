import torch
import os
from Hefesto.models.model import Model


def save_model(path, model):
    # Save the model
    torch.save(model.state_dict(), path)


def load_model(path: str, model: Model) -> Model:
    model.load_state_dict(torch.load(path))
    return model


def write_results(
    epochs, good_ele, bad_ele, path: str, size: int, model: Model, seed: int
):
    with open(path, "a") as a:
        a.write(f"Seed: {seed}\n")
        a.write(f"Model: {model}\n")
        a.write(f"Epochs: \n{epochs}\n")
        a.write(f"Good Data Gen: {len(good_ele)}\n")
        a.write(f"Bad Data Gen: {len(bad_ele)}\n")
        a.write(f"Acierto: {(len(good_ele)/size)*100}%\n")
        a.write("---------------------------------------------------------------\n")
