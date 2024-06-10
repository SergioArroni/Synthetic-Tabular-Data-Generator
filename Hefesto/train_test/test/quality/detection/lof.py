from sklearn.neighbors import LocalOutlierFactor

from Hefesto.train_test.test.quality.detection import Detection


class LOFDetection(Detection):
    """Clase para detectar anomalías usando Local Outlier Factor."""
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
        # Usamos LocalOutlierFactor para detectar outliers
        lof = LocalOutlierFactor(novelty=True, n_neighbors=20)
        lof.fit(self.test_loader.dataset.features)
        return lof
