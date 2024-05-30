from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Utility:
    def __init__(self, file_name, seed, X=None, y=None):
        self.X = X
        self.y = y
        self.seed = seed
        self.file_name = file_name
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

        self.result = (f1, accuracy)

    def print_result(self):
        # Print the result in a file
        with open(self.file_name, "w") as f:
            f.write("f1, accuracy\n")
            f.write(f"{self.result[0]}, {self.result[1]}\n")

    def execute(self):
        self.process()
        self.run()
        self.print_result()
