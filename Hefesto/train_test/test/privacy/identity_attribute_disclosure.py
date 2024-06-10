import numpy as np
import torch
import pandas as pd
from Hefesto.train_test.test.privacy import Privacy

class IdentityAttributeDisclosure(Privacy):
    def __init__(self, data, gen_data, path: str):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.attribute_disclosure_rate = None
        self.identity_disclosure_rate = None

    def to_numpy(self, data):
        """
        Convert data to NumPy array if it is not already.
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        else:
            return data.to_numpy()

    def simulate_attribute_attack(self, known_attributes_percentage=0.01):
        """
        Simulate an attribute disclosure attack by revealing a small percentage of attributes
        and trying to guess the rest.
        """
        original_data = self.to_numpy(self.data)
        synthetic_data = self.to_numpy(self.gen_data)

        num_attributes = original_data.shape[1]
        num_known_attributes = int(num_attributes * known_attributes_percentage)
        
        correct_guesses = 0
        total_guesses = 0

        for original_record, synthetic_record in zip(original_data, synthetic_data):
            known_indices = np.random.choice(num_attributes, num_known_attributes, replace=False)
            known_attributes = original_record[known_indices]

            synthetic_known_attributes = synthetic_record[known_indices]
            if np.array_equal(known_attributes, synthetic_known_attributes):
                correct_guesses += 1
            total_guesses += 1

        self.attribute_disclosure_rate = correct_guesses / total_guesses

    def simulate_identity_attack(self):
        """
        Simulate an identity disclosure attack by checking if any synthetic record matches an original record.
        """
        original_data = self.to_numpy(self.data)
        synthetic_data = self.to_numpy(self.gen_data)

        matches = 0
        for original_record in original_data:
            for synthetic_record in synthetic_data:
                if np.array_equal(original_record, synthetic_record):
                    matches += 1
                    break

        self.identity_disclosure_rate = matches / len(original_data)

    def write_results(self):
        with open(self.path, "w") as file:
            file.write(f"Attribute Disclosure Rate: {self.attribute_disclosure_rate}\n")
            file.write(f"Identity Disclosure Rate: {self.identity_disclosure_rate}\n")

    def execute(self):
        self.simulate_attribute_attack()
        self.simulate_identity_attack()
        self.write_results()
