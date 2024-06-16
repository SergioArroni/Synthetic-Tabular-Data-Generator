class Privacy:
    def __init__(self, data, gen_data, path: str, seed: int = 42):
        self.data = data
        self.gen_data = gen_data
        self.path = path
        self.seed = seed

    def execute(self):
        raise NotImplementedError
