class Detection:
    """Clase base para detectar anomalías."""

    def __init__(self, gen_data, seed, file_name):
        self.gen_data = gen_data
        self.seed = seed
        self.good_ele = []
        self.bad_ele = []
        self.file_name = file_name
        self.model = None

    def detection_model(self):
        raise NotImplementedError

    def predict(self):
        for ele in self.gen_data:
            ele = ele.cpu()
            if self.model.predict([ele]) == 1:
                self.good_ele.append(ele)
            else:
                self.bad_ele.append(ele)

    def save_results(self):
        # Save the results in a file
        with open(self.file_name, "w") as file:
            file.write(f"Good elements: {len(self.good_ele)}\n")
            file.write(f"Bad elements: {len(self.bad_ele)}\n")
            file.write(
                f"Percentage of good elements: {len(self.good_ele) / len(self.gen_data) * 100}\n"
            )

    def execute(self):
        self.model = self.detection_model()
        self.predict()
        self.save_results()
