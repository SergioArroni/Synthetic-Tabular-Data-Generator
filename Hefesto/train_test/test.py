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

        good_ele = []
        bad_ele = []

        for ele in self.test_loader.dataset.features:
            if clf.predict([ele]) == 1:
                good_ele.append(ele)
            else:
                bad_ele.append(ele)

        return good_ele, bad_ele
