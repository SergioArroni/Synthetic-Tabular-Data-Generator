from Hefesto.train_test.test.privacy import Privacy
import numpy as np


class DCR(Privacy):
    def __init__(self, data, gen_data, path: str):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.dcr = None

    def calculate_dcr(self):
        """
        Calculate the Disclosure Control Ratio (DCR)
        """
        # Convert data to numpy arrays for easier manipulation
        original = np.array(self.data)
        generated = np.array(self.gen_data)

        # Ensure the arrays are the same length
        if len(original) != len(generated):
            raise ValueError(
                "The length of the original data and generated data must be the same."
            )

        # Calculate the frequency of each unique value in the original data
        unique, counts = np.unique(original, return_counts=True)
        original_freq = dict(zip(unique, counts))

        # Calculate the frequency of each unique value in the generated data
        unique_gen, counts_gen = np.unique(generated, return_counts=True)
        generated_freq = dict(zip(unique_gen, counts_gen))

        # Calculate DCR
        dcr_values = []
        for key in original_freq:
            if key in generated_freq:
                dcr = generated_freq[key] / original_freq[key]
                dcr_values.append(dcr)
            else:
                dcr_values.append(0)

        # The DCR is the mean of all calculated DCR values
        self.dcr = np.mean(dcr_values)

    def write_dcr(self):
        with open(self.path, "w") as file:
            file.write(f"DCR: {self.dcr}\n")

    def execute(self):
        self.calculate_dcr()
        self.write_dcr()
