from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import numpy as np
import optuna


class Effiency:
    """Clase para evaluar la eficiencia de un modelo de clasificación."""

    def __init__(self, path, seed, X=None, y=None):
        self.X = X
        self.y = y
        self.seed = seed
        self.path = path
        self.result = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def process(self):
        if self.y.dtype.kind in "f":  # Check if y is of float type
            print("Converting float labels to integers for classification.")
            self.y = self.y.astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.seed
        )

    def objective(self, trial):
        # Definir el espacio de búsqueda de hiperparámetros
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "leaf_estimation_iterations": trial.suggest_int(
                "leaf_estimation_iterations", 1, 10
            ),
            "random_seed": self.seed,
            "silent": True,
        }

        # Escalar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        # Entrenar el modelo
        model = CatBoostClassifier(**params)
        model.fit(X_train_scaled, self.y_train)

        # Evaluar el modelo
        predictions = model.predict(X_test_scaled)

        f1 = f1_score(self.y_test, predictions, average="weighted")

        return f1

    def run(self):
        # Realizar la optimización de hiperparámetros
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, n_jobs=-1)

        # Obtener los mejores hiperparámetros
        best_params = study.best_params

        # Entrenar y evaluar el modelo en 10 ejecuciones con los mejores hiperparámetros
        for _ in range(10):
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

            model = CatBoostClassifier(**best_params)
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            f1 = f1_score(self.y_test, predictions, average="weighted")
            accuracy = accuracy_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions, average="weighted")
            precision = precision_score(
                self.y_test, predictions, average="weighted", zero_division=0
            )
            roc = roc_auc_score(
                self.y_test, predictions, average="weighted", multi_class="ovo"
            )

            self.result.append((f1, accuracy, recall, precision, roc))

        # Calcular la media de las métricas
        self.result = np.mean(self.result, axis=0)

    def print_result(self):
        # Print the result in a file
        with open(self.path, "w") as f:
            f.write(f"F1: {self.result[0]}\n")
            f.write(f"Accuracy: {self.result[1]}\n")
            f.write(f"Recall: {self.result[2]}\n")
            f.write(f"Precision: {self.result[3]}\n")
            f.write(f"ROC: {self.result[4]}\n")

    def execute(self):
        self.process()
        self.run()
        self.print_result()
