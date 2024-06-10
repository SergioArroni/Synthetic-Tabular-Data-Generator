class Privacy:
    def __init__(self, data, gen_data, path: str):
        self.data = data
        self.gen_data = gen_data
        self.path = path

    def execute(self):
        raise NotImplementedError
