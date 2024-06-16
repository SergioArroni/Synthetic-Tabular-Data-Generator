import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize, QuantileTransformer
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np


class Preprocess:
    def __init__(self, df: pd.DataFrame, seed: int = 42) -> None:
        self.df = df
        self.scaler = StandardScaler()
        self.seed = seed
        self.qt = QuantileTransformer(
            n_quantiles=500, output_distribution="normal", random_state=self.seed
        )
        self.transformed_data = None
        self.data_destransformer = None

    def normalize(self) -> None:
        # Esta operación no es fácilmente reversible de una manera general
        df_aux = normalize(self.df, norm="l2")
        self.df = pd.DataFrame(df_aux, columns=self.df.columns)

    def scaler_method(self) -> None:
        df_aux = self.scaler.fit_transform(self.df)
        self.df = pd.DataFrame(df_aux, columns=self.df.columns)
        dump(self.scaler, "./scaler/scaler.joblib")

    def outlayer(self):
        Q1 = self.df.quantile(0.25)
        Q4 = self.df.quantile(0.75)
        IQR = Q4 - Q1
        self.df = self.df[
            ~((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q4 + 1.5 * IQR))).any(axis=1)
        ]

    def des_prep(self) -> None:
        self.des_scale()
        # No se incluye des_norm ya que la normalización L2 no tiene un proceso de inversión directo

    def prep(self) -> None:
        self.scaler_method()
        # self.normalize()

    def des_scale(self) -> None:
        # Asegurarse de que el escalador haya sido ajustado previamente
        self.scaler = load("./scaler/scaler.joblib")
        if hasattr(self.scaler, "mean_"):
            df_aux = self.scaler.inverse_transform(
                self.df
            )  # Revertir la transformación
            df_aux = pd.DataFrame(df_aux, columns=self.df.columns)
            self.df = df_aux.round().astype(
                int
            )  # Redondear y luego convertir a enteros
        else:
            print("Scaler has not been fitted yet.")

    def quantile_transform(self):
        # Transformar los datos
        self.transformed_data = self.qt.fit_transform(self.df)

        # Visualizar los datos antes y después de la transformación
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(self.df, bins=30, alpha=0.7, label="Original Data")
        plt.title("Original Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(
            self.transformed_data,
            bins=30,
            alpha=0.7,
            label="Transformed Data",
        )
        plt.title("Transformed Data - Normal Distribution")
        plt.xlabel("Quantile Transformed Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig("./final_results/data/prep/quantile_transform.png")

    def des_transformar(self, generated_data):
        # De-transformar las predicciones
        generated_data = generated_data.to_numpy().reshape(-1, 1)
        generated_data = pd.DataFrame(generated_data, columns=self.df.columns)
        print(generated_data)
        self.data_destransformer = self.qt.inverse_transform(
            generated_data
        )
        print(self.data_destransformer)
