import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from joblib import dump, load


class Preprocess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.scaler = StandardScaler()

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
            self.df = df_aux.round().astype(int)  # Redondear y luego convertir a enteros
        else:
            print("Scaler has not been fitted yet.")
