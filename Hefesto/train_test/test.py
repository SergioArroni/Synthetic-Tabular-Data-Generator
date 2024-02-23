from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
import pandas as pd
from Hefesto.models.model import Model
from Hefesto.utils.utils import save_data, plot_statistics
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class Test:

    def __init__(
        self, model: Model, test_loader: DataLoader, val_loader: DataLoader, seed: int, device: torch.device
    ):
        self.model = model
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.seed = seed
        self.device = device
        self.gen_data = torch.empty(0, 12).to(self.device)

    def generate_data(self):
        self.model.eval()  # Asegúrate de que el modelo esté en modo de evaluación

        with torch.no_grad():  # No calcular gradientes
            for ele in tqdm(self.test_loader.dataset.features):
                ele = ele.to(self.device)
                
                # print(ele)
                
                gen = self.model(ele).round()
                gen = gen.unsqueeze(0)
                # print(gen)
                
                # print(self.gen_data)

                self.gen_data = torch.cat((self.gen_data, gen), 0)

    def evaluate_model(self):
        clf = self.isolation_forest()
        self.generate_data()
        
        # print(self.gen_data)
        
        columns = self.val_loader.dataset.columns
        df = pd.DataFrame(self.gen_data, columns=columns, dtype="int")

        plot_statistics(df, f"./img/stadistics/gendata/boxplot")

        # save the generated data in a .csv with the atricbutes of the original data
        save_data(
            f"./results/gen_data/generated_data_{self.model}_{self.seed}_{time.time()}.csv",
            df,
        )

        good_ele = []
        bad_ele = []

        for ele in self.gen_data:
            ele = ele
            if clf.predict([ele]) == 1:
                good_ele.append(ele)
            else:
                bad_ele.append(ele)

        return good_ele, bad_ele

    def isolation_forest(self):
        clf = IsolationForest(random_state=self.seed).fit(
            self.val_loader.dataset.features
        )
        return clf
