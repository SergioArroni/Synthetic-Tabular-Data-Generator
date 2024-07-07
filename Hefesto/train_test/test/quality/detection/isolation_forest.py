from sklearn.ensemble import IsolationForest

from Hefesto.train_test.test.quality.detection import Detection


class IsolationForestDetection(Detection):
    """Clase para detectar anomalías usando Isolation Forest."""

    def __init__(
        self,
        original_data,
        synthetic_data,
        seed,
        path,
    ):
        super().__init__(
            original_data=original_data,
            synthetic_data=synthetic_data,
            seed=seed,
            path=path,
        )

    def detection_model(self):
        return IsolationForest(random_state=self.seed).fit(
            self.original_data.dataset.features
        )
