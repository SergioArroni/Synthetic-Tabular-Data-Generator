import pandas as pd
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Hefesto.models.model import Model
from Hefesto.utils import save_data, plot_statistics
from Hefesto.train_test.test.quality.detection import IsolationForestDetection
from Hefesto.train_test.test.utility.efficiency import TTS, TTSR, TSTR, TRTS
from Hefesto.train_test.test.quality.stadistics import Correlation, Metrics, Tests


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
        self.df_test = None
        self.df_gen_data = None
        self.df_test = None

    def generate_data(self):

        with torch.no_grad():  # No calcular gradientes
            for feature, _ in tqdm(self.val_loader):
                input = feature.to(self.device)

                gen = self.model.test_model_gen(self.model, input)

                self.gen_data = torch.cat((self.gen_data, gen), 0)

    def evaluate_efficiency(self):
        ttsttr = TTS(self.df_gen_data, self.df_test, self.seed)
        ttsttr.execute()
        ttsr = TTSR(self.df_gen_data, self.df_test, self.seed)
        ttsr.execute()
        tstr = TSTR(self.df_gen_data, self.df_test, self.seed)
        tstr.execute()
        trts = TRTS(self.df_test, self.df_gen_data, self.seed)
        trts.execute()

    def evaluate_detection(self):
        isolation_forest = IsolationForestDetection(
            self.test_loader, self.gen_data, self.seed
        )
        isolation_forest.execute()

    def evaluate_stadistics(self, df: pd.DataFrame):
        Correlation(df).matrix_correlation("gen")
        Metrics(df).calculate_metrics()
        tests = Tests(data=self.df_test, data2=self.gen_data)
        tests.test_kl()
        tests.test_ks()

    def evaluate_privacy(self):
        pass

    def evaluate_quality(self, df):
        self.evaluate_stadistics(df=df)
        self.evaluate_detection()

    def evaluate_utility(self):
        self.evaluate_efficiency()

    def evaluate_model(self):
        self.generate_data()

        # print(self.gen_data)
        self.gen_data = self.gen_data.cpu()

        columns = self.val_loader.dataset.columns
        self.df_gen_data = pd.DataFrame(self.gen_data.numpy(), columns=columns)
        self.df_gen_data = self.df_gen_data.round().astype(
            int
        )  # Redondear y luego convertir a enteros

        # plot_statistics(df, f"./img/stadistics/gendata/standar/boxplot")

        # prep = Preprocess(df)
        # prep.des_scale()
        # df = prep.df

        plot_statistics(self.df_gen_data, f"./img/stadistics/gendata/boxplot")

        save_data(
            f"./results/gen_data/generated_data_{self.model}_{self.seed}_{time.time()}.csv",
            self.df_gen_data,
        )
        self.df_test = pd.read_csv("data/cardio/split/cardio_test.csv", sep=";")

        self.evaluate_utility()
        self.evaluate_quality(self.df_gen_data)
        self.evaluate_privacy()
