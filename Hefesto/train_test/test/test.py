from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
import pandas as pd
from Hefesto.models.model import Model
from Hefesto.utils.utils import save_data, plot_statistics
from Hefesto.preprocess.correlations import matrix_correlation
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        clf = self.isolation_forest()
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

        good_ele = []
        bad_ele = []

        for ele in self.gen_data:
            ele = ele.cpu()
            if clf.predict([ele]) == 1:
                good_ele.append(ele)
            else:
                bad_ele.append(ele)

        # Extraer la X y la y de df
        X = df.drop("cardio", axis=1)
        y = df["cardio"]

        metrics = self.utility_TTS_vs_TTR(X, y)
        
        df_test = pd.read_csv("data/cardio/split/cardio_test.csv", sep=";")

        df_cocktel = pd.concat([df, df_test], axis=0)

        X = df_cocktel.drop("cardio", axis=1)
        y = df_cocktel["cardio"]

        metrics_cocktel = self.utility_TSTR(X, y)

        return good_ele, bad_ele, metrics, metrics_cocktel

    def isolation_forest(self):
        clf = IsolationForest(random_state=self.seed).fit(
            self.test_loader.dataset.features
        )
        return clf

    def utility(self, X, y):
        """
        Evalúa el desempeño de un modelo de regresión sobre los datos generados.
        """
        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        # Escalar los datos
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Entrenar el modelo
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(X_train, y_train)

        # Evaluar el modelo
        predictions = model.predict(X_test)

        f1 = f1_score(y_test, predictions, average="weighted")
        accuracy = accuracy_score(y_test, predictions)

        return (f1, accuracy)
    
    def utility_TSTR(self, X, y):
        """
        Evalúa el desempeño de un modelo de regresión sobre los datos generados.
        """
        # Escalar los datos
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Entrenar el modelo
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(X, y)

        # Evaluar el modelo
        predictions = model.predict(X)

        f1 = f1_score(y, predictions, average="weighted")
        accuracy = accuracy_score(y, predictions)

        return (f1, accuracy)
