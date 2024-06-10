import pandas as pd
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Hefesto.models.model import Model
from Hefesto.utils import save_data, plot_statistics
from Hefesto.train_test.test.quality.detection import (
    IsolationForestDetection,
    LOFDetection,
    AEDetection,
)
from Hefesto.train_test.test.utility.efficiency import TTS, TTSR, TSTR, TRTS
from Hefesto.train_test.test.quality.stadistics import Correlation, Metrics, Tests
from Hefesto.train_test.test.privacy import (
    DCR,
    DifferentialPrivacy,
    IdentityAttributeDisclosure,
    MembershipInferenceAttack,
    MMD,
)


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

    def generate_data(self):

        with torch.no_grad():  # No calcular gradientes
            for feature, _ in tqdm(self.val_loader):
                input = feature.to(self.device)

                gen = self.model.test_model_gen(self.model, input)

                self.gen_data = torch.cat((self.gen_data, gen), 0)

    def evaluate_efficiency(self):
        ttsttr = TTS(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TTS.txt",
        )
        ttsttr.execute()
        ttsr = TTSR(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TTSR.txt",
        )
        ttsr.execute()
        tstr = TSTR(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TSTR.txt",
        )
        tstr.execute()
        trts = TRTS(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TRTS.txt",
        )
        trts.execute()

    def evaluate_detection(self):
        IsolationForestDetection(
            test_loader=self.test_loader,
            gen_data=self.gen_data,
            seed=self.seed,
            path="./final_results/quality/detection/isolation.txt",
        ).execute()
        LOFDetection(
            test_loader=self.test_loader,
            gen_data=self.gen_data,
            seed=self.seed,
            path="./final_results/quality/detection/lof.txt",
        ).execute()
        AEDetection(
            test_loader=self.test_loader,
            gen_data=self.gen_data,
            seed=self.seed,
            path="./final_results/quality/detection/ae.txt",
        ).execute()

    def evaluate_stadistics(self, df: pd.DataFrame):
        Correlation(df, "./final_results/quality/statistics/corr.png").execute()
        Metrics(df, "./final_results/quality/statistics/metrics.txt").execute()
        Tests(
            data=self.df_test,
            data2=self.gen_data,
            path="./final_results/quality/statistics/tests.txt",
        ).execute()

    def evaluate_privacy(self):
        IdentityAttributeDisclosure(
            data=self.df_test,
            gen_data=self.gen_data,
            path="./final_results/privacy/identity.txt",
        ).execute()
        DifferentialPrivacy(
            data=self.df_test,
            gen_data=self.gen_data,
            path="./final_results/privacy/differential.txt",
        ).execute()
        DCR(
            data=self.df_test,
            gen_data=self.gen_data,
            path="./final_results/privacy/dcr.txt",
        ).execute()
        MembershipInferenceAttack(
            data=self.df_test,
            gen_data=self.gen_data,
            path="./final_results/privacy/membership.txt",
        ).execute()
        MMD(
            data=self.df_test,
            gen_data=self.gen_data,
            path="./final_results/privacy/mmd.txt",
        ).execute()

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
            f"./final_results/data/generated_data_{self.model}_{self.seed}_{time.time()}.csv",
            self.df_gen_data,
        )
        self.df_test = pd.read_csv("data/cardio/split/cardio_test.csv", sep=";")

        self.evaluate_utility()
        self.evaluate_quality(self.df_gen_data)
        self.evaluate_privacy()
