from sklearn.ensemble import IsolationForest
import torch
from torch.utils.data import DataLoader

from Hefesto.models.model import Model


class Test:

    def __init__(
        self, model: Model, test_loader: DataLoader, val_loader: DataLoader, seed: int
    ):
        self.model = model
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.seed = seed

    def evaluate_model(self):
        clf = IsolationForest(random_state=self.seed).fit(
            self.val_loader.dataset.features
        )

        gen_data = self.model.forward(self.test_loader.dataset.features)
        print(self.test_loader.dataset.features[0])
        print(gen_data[0])

        good_ele = []
        bad_ele = []

        for ele in gen_data:
            ele = ele.detach().numpy()
            if clf.predict([ele]) == 1:
                good_ele.append(ele)
            else:
                bad_ele.append(ele)

        return good_ele, bad_ele
