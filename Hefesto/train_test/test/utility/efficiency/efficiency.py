from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Effiency:
    """Clase para evaluar la eficiencia de un modelo de clasificación."""

    def __init__(self, path, seed, X=None, y=None):
        self.X = X
        self.y = y
        self.seed = seed
        self.path = path
        self.result = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def process(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.seed
        )

    def run(self):
        # Escalar los datos
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Entrenar el modelo
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(self.X_train, self.y_train)

        # Evaluar el modelo
        predictions = model.predict(self.X_test)

        f1 = f1_score(self.y_test, predictions, average="weighted")
        accuracy = accuracy_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions, average="weighted")
        precision = precision_score(self.y_test, predictions, average="weighted", zero_division=0)
        if predictions.ndim != 1:
            roc = roc_auc_score(self.y_test, predictions, average="weighted", multi_class='ovo')
        else:
            roc = "NaN"

        self.result = (f1, accuracy, recall, precision, roc)

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
