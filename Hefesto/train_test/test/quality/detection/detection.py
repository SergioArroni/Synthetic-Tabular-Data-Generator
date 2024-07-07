from sklearn.preprocessing import StandardScaler
import pandas as pd


class Detection:
    """Clase base para detectar anomalías."""

    def __init__(self, original_data, synthetic_data, seed, path):
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.seed = seed
        self.good_ele = []
        self.bad_ele = []
        self.path = path
        self.model = None

    def detection_model(self):
        raise NotImplementedError

    def predict(self):
        for ele in self.synthetic_data:
            ele = ele.cpu()
            if self.model.predict([ele]) == 1:
                self.good_ele.append(ele)
            else:
                self.bad_ele.append(ele)

    def save_results(self):
        # Save the results in a file
        with open(self.path, "w") as file:
            file.write(f"Good elements: {len(self.good_ele)}\n")
            file.write(f"Bad elements: {len(self.bad_ele)}\n")
            file.write(
                f"Percentage of good elements: {len(self.good_ele) / len(self.synthetic_data) * 100}\n"
            )

    def execute(self):
        self.model = self.detection_model()
        self.predict()
        self.save_results()
