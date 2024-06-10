import numpy as np
from Hefesto.train_test.test.privacy import Privacy

class IdentityAttributeDisclosure(Privacy):
    def __init__(self, data, gen_data, path: str):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.attribute_disclosure_rate = None
        self.identity_disclosure_rate = None

    def simulate_attribute_attack(self, known_attributes_percentage=0.01):
        """
        Simulate an attribute disclosure attack by revealing a small percentage of attributes
        and trying to guess the rest.
        """
        num_attributes = self.data.shape[1]
        num_known_attributes = int(num_attributes * known_attributes_percentage)
        
        correct_guesses = 0
        total_guesses = 0

        for original_record, synthetic_record in zip(self.data, self.gen_data):
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
        matches = 0
        for original_record in self.data:
            for synthetic_record in self.gen_data:
                if np.array_equal(original_record, synthetic_record):
                    matches += 1
                    break

        self.identity_disclosure_rate = matches / len(self.data)

    def write_results(self):
        with open(self.path, "w") as file:
            file.write(f"Attribute Disclosure Rate: {self.attribute_disclosure_rate}\n")
            file.write(f"Identity Disclosure Rate: {self.identity_disclosure_rate}\n")

    def execute(self):
        self.simulate_attribute_attack()
        self.simulate_identity_attack()
        self.write_results()

