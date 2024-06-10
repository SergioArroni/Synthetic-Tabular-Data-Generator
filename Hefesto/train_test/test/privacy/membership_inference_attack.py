import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from Hefesto.train_test.test.privacy import Privacy

class MembershipInferenceAttack(Privacy):
    def __init__(self, data, gen_data, path: str):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.attack_model = None
        self.attack_results = None

    def split_data(self, test_size=0.5):
        """Divide los datos en un conjunto de entrenamiento y un conjunto de prueba."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, np.ones(len(self.data)), test_size=test_size, random_state=42
        )
        X_gen_train, X_gen_test, y_gen_train, y_gen_test = train_test_split(
            self.gen_data, np.zeros(len(self.gen_data)), test_size=test_size, random_state=42
        )

        X_combined_train = np.concatenate((X_train, X_gen_train))
        y_combined_train = np.concatenate((y_train, y_gen_train))

        X_combined_test = np.concatenate((X_test, X_gen_test))
        y_combined_test = np.concatenate((y_test, y_gen_test))

        return X_combined_train, X_combined_test, y_combined_train, y_combined_test

    def train_attack_model(self, X_train, y_train):
        """Entrena el modelo de ataque."""
        self.attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.attack_model.fit(X_train, y_train)

    def evaluate_attack(self, X_test, y_test):
        """Evalúa el modelo de ataque."""
        y_pred = self.attack_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def execute_attack(self):
        """Ejecuta el ataque de membresía."""
        X_train, X_test, y_train, y_test = self.split_data()
        self.train_attack_model(X_train, y_train)
        accuracy = self.evaluate_attack(X_test, y_test)
        self.attack_results = accuracy

    def write_results(self):
        with open(self.path, "w") as file:
            file.write(f"Membership Inference Attack Accuracy: {self.attack_results}\n")

    def execute(self):
        self.execute_attack()
        self.write_results()

