import pandas as pd
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Hefesto.models.model import Model
from Hefesto.utils import save_data
from Hefesto.train_test.test.quality.detection import (
    IsolationForestDetection,
    LOFDetection,
    AEDetection,
)
from Hefesto.preprocess import PrepExe
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
        device: torch.device,
        preprocess: PrepExe,
        seed: int = 42,
        hard_prep: bool = False,
    ):
        self.model = model
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.seed = seed
        self.preprocess = preprocess
        self.device = device
        tam = self.val_loader.dataset.features.shape[1]
        self.gen_data = torch.empty(0, tam).to(self.device)
        self.df_test = None
        self.df_gen_data = None
        self.hard_prep = hard_prep

    def generate_data(self):

        with torch.no_grad():  # No calcular gradientes
            for feature, _ in tqdm(self.val_loader):
                input = feature.to(self.device)

                gen = self.model.test_model_gen(self.model, input)

                self.gen_data = torch.cat((self.gen_data, gen), 0)

    def evaluate_efficiency(self):
        TTS(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TTS.txt",
        ).execute()

        TTSR(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TTSR.txt",
        ).execute()

        TSTR(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TSTR.txt",
        ).execute()

        TRTS(
            df=self.df_gen_data,
            df_test=self.df_test,
            seed=self.seed,
            path="./final_results/utility/efficiency/TRTS.txt",
        ).execute()

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

    def evaluate_stadistics(self):
        Metrics(
            original_data=self.df_test,
            synthetic_data=self.df_gen_data,
            path="./final_results/quality/statistics/",
        ).execute()
        Correlation(
            data=self.df_gen_data, path="./final_results/quality/statistics/corr.png"
        ).execute()
        Tests(
            data=self.df_test,
            data2=self.df_gen_data,
            path="./final_results/quality/statistics/tests.txt",
        ).execute()

    def evaluate_privacy(self):
        DCR(
            data=self.df_test,
            gen_data=self.df_gen_data,
            path="./final_results/privacy/dcr.txt",
        ).execute()
        MMD(
            data=self.df_test,
            gen_data=self.df_gen_data,
            path="./final_results/privacy/mmd.txt",
        ).execute()
        IdentityAttributeDisclosure(
            data=self.df_test,
            gen_data=self.df_gen_data,
            path="./final_results/privacy/identity.txt",
        ).execute()
        MembershipInferenceAttack(
            data=self.df_test,
            gen_data=self.df_gen_data,
            path="./final_results/privacy/membership.txt",
        ).execute()

    def evaluate_quality(self):
        self.evaluate_stadistics()
        self.evaluate_detection()

    def evaluate_utility(self):
        self.evaluate_efficiency()

    def evaluate_model(self):
        self.generate_data()

        self.gen_data = self.gen_data.cpu()

        columns = self.val_loader.dataset.columns
        self.df_gen_data = pd.DataFrame(self.gen_data.numpy(), columns=columns)
        
        if self.hard_prep:
            self.preprocess.preprocess.des_transformar(self.df_gen_data)
            self.df_gen_data = self.preprocess.preprocess.data_destransformer
            self.df_gen_data = pd.DataFrame(self.gen_data.numpy(), columns=columns)

        self.df_gen_data = self.df_gen_data.round().astype(
            int
        )
        
        save_data(
            f"./final_results/data/generated_data_{self.model}_{self.seed}_{time.time()}.csv",
            self.df_gen_data,
        )
        self.df_test = pd.read_csv("data/cardio/split/cardio_test.csv", sep=";")

        self.evaluate_utility()
        self.evaluate_quality()
        self.evaluate_privacy()
