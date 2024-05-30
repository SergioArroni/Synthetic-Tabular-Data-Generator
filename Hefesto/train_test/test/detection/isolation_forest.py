from sklearn.ensemble import IsolationForest

from Hefesto.train_test.test.detection import Detection


class IsolationForestDetection(Detection):
    def __init__(
        self,
        test_loader,
        gen_data,
        seed,
        file_name="./new_results/detection/detection.txt",
    ):
        super().__init__(gen_data=gen_data, seed=seed, file_name=file_name)
        self.test_loader = test_loader

    def detection_model(self):
        return IsolationForest(random_state=self.seed).fit(
            self.test_loader.dataset.features
        )
