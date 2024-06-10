import numpy as np
from Hefesto.train_test.test.privacy import Privacy
from sklearn.metrics.pairwise import rbf_kernel

class MMD(Privacy):
    def __init__(self, data, gen_data, path: str, kernel='rbf', gamma=1.0):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.mmd = None
        self.kernel = kernel
        self.gamma = gamma

    def compute_mmd(self, X, Y):
        """Compute Maximum Mean Discrepancy (MMD) between two datasets X and Y using the specified kernel."""
        XX = rbf_kernel(X, X, gamma=self.gamma)
        YY = rbf_kernel(Y, Y, gamma=self.gamma)
        XY = rbf_kernel(X, Y, gamma=self.gamma)

        return XX.mean() + YY.mean() - 2 * XY.mean()

    def calculate_mmd(self):
        """
        Calculate the Maximum Mean Discrepancy (MMD)
        """
        # Convert data to numpy arrays for easier manipulation
        original = np.array(self.data)
        generated = np.array(self.gen_data)

        # Reshape data to ensure it is 2D (required by rbf_kernel)
        if original.ndim == 1:
            original = original.reshape(-1, 1)
        if generated.ndim == 1:
            generated = generated.reshape(-1, 1)

        # Compute MMD using the specified kernel
        self.mmd = self.compute_mmd(original, generated)

    def write_mmd(self):
        with open(self.path, "w") as file:
            file.write(f"MMD: {self.mmd}\n")

    def execute(self):
        self.calculate_mmd()
        self.write_mmd()
