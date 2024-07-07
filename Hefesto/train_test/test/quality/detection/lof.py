from sklearn.neighbors import LocalOutlierFactor

from Hefesto.train_test.test.quality.detection import Detection


class LOFDetection(Detection):
    """Clase para detectar anomalías usando Local Outlier Factor."""

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
        # Usamos LocalOutlierFactor para detectar outliers
        lof = LocalOutlierFactor(novelty=True)
        lof.fit(self.original_data.dataset.features)
        return lof
