from sklearn.ensemble import IsolationForest

from Hefesto.train_test.test.quality.detection import Detection


class IsolationForestDetection(Detection):
    """Clase para detectar anomalías usando Isolation Forest."""
    def __init__(
        self,
        test_loader,
        gen_data,
        seed,
        path,
    ):
        super().__init__(gen_data=gen_data, seed=seed, path=path)
        self.test_loader = test_loader

    def detection_model(self):
        return IsolationForest(random_state=self.seed).fit(
            self.test_loader.dataset.features
        )
