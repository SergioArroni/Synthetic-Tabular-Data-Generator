from torch.utils.data import DataLoader
import pandas as pd
from Hefesto.models.model import Model
from Hefesto.train_test.test.detection import IsolationForestDetection
from Hefesto.utils import save_data, plot_statistics
from Hefesto.preprocess.correlations import matrix_correlation
from Hefesto.train_test.test.utility import TTSTTR, TTSR, TSTR, TRTS
import time
import torch
from tqdm import tqdm


class Test:

    def __init__(
        self,
        model: Model,
        test_loader: DataLoader,
        val_loader: DataLoader,
        seed: int,
        device: torch.device,
    ):
        self.model = model
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.seed = seed
        self.device = device
        tam = self.val_loader.dataset.features.shape[1]
        self.gen_data = torch.empty(0, tam).to(self.device)

    def generate_data(self):

        with torch.no_grad():  # No calcular gradientes
            for feature, _ in tqdm(self.val_loader):
                input = feature.to(self.device)

                gen = self.model.test_model_gen(self.model, input)

                self.gen_data = torch.cat((self.gen_data, gen), 0)

    def evaluate_model(self):
        self.generate_data()

        # print(self.gen_data)
        self.gen_data = self.gen_data.cpu()

        columns = self.val_loader.dataset.columns
        df = pd.DataFrame(self.gen_data.numpy(), columns=columns)
        df = df.round().astype(int)  # Redondear y luego convertir a enteros

        # plot_statistics(df, f"./img/stadistics/gendata/standar/boxplot")

        # prep = Preprocess(df)
        # prep.des_scale()
        # df = prep.df

        plot_statistics(df, f"./img/stadistics/gendata/boxplot")

        matrix_correlation(df, "gen")

        save_data(
            f"./results/gen_data/generated_data_{self.model}_{self.seed}_{time.time()}.csv",
            df,
        )
        df_test = pd.read_csv("data/cardio/split/cardio_test.csv", sep=";")

        ttsttr = TTSTTR(df, df_test, self.seed)
        ttsttr.execute()
        ttsr = TTSR(df, df_test, self.seed)
        ttsr.execute()
        tstr = TSTR(df, df_test, self.seed)
        tstr.execute()
        trts = TRTS(df_test, df, self.seed)
        trts.execute()
        
        isolation_forest = IsolationForestDetection(self.test_loader, self.gen_data, self.seed)
        isolation_forest.execute()
