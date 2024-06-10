import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Hefesto.train_test.test.quality.detection import Detection

class _Autoencoder(nn.Module):
    """Autoencoder simple para detectar anomalías."""

    def __init__(self, input_dim):
        super(_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AEDetection(Detection):
    """Clase para detectar anomalías usando un Autoencoder."""

    def __init__(
        self,
        test_loader,
        gen_data,
        seed,
        path,
        threshold=None
    ):
        super().__init__(gen_data=gen_data, seed=seed, path=path)
        self.test_loader = test_loader
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = self.build_autoencoder(
            input_dim=test_loader.dataset.features.shape[1]
        ).to(self.device)
        self.threshold = threshold

    def build_autoencoder(self, input_dim):
        return _Autoencoder(input_dim)

    def detection_model(self):
        # Entrenamos el Autoencoder
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        features = self.test_loader.dataset.features
        features = torch.tensor(features, dtype=torch.float32).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)

        self.autoencoder.train()
        num_epochs = 50
        batch_size = 32

        for epoch in range(num_epochs):
            permutation = torch.randperm(features.size()[0])
            for i in range(0, features.size()[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_features = features[indices]

                optimizer.zero_grad()
                outputs = self.autoencoder.forward(batch_features)
                loss = criterion(outputs, batch_features)
                loss.backward()
                optimizer.step()

        # Calculate threshold based on training data
        self.autoencoder.eval()
        with torch.no_grad():
            reconstruction = self.autoencoder.forward(features)
        reconstruction_errors = torch.mean((features - reconstruction) ** 2, dim=1).cpu().numpy()
        if self.threshold is None:
            self.threshold = np.percentile(reconstruction_errors, 95)  # Set threshold at the 95th percentile

        return self.autoencoder
    
    def predict(self):
        # Usamos el Autoencoder para detectar anomalías
        self.autoencoder.eval()
        features = torch.tensor(self.gen_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.autoencoder.forward(features)

        predictions = predictions.cpu().numpy()
        features = features.cpu().numpy()

        reconstruction_errors = np.mean(np.power(features - predictions, 2), axis=1)
        for idx, error in enumerate(reconstruction_errors):
            if error > self.threshold:
                self.bad_ele.append(self.gen_data[idx])
            else:
                self.good_ele.append(self.gen_data[idx])
